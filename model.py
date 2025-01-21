# DiT with cross attention

import random
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor as TT


# TODO: cache freqs
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding

class Modulation(nn.Sequential):
    def __init__(self, d: int, n: int):
        super().__init__(nn.SiLU(), nn.Linear(d, n * d, bias=True))
        self.n = n
    def reset_parameters(self):
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)
    def forward(self, c: TT) -> tuple[TT, ...]:
        return super().forward(c).chunk(self.n, dim=-1)

class ModulatedRMSNorm(nn.RMSNorm):
    def forward(self, x: TT, shift: TT, scale: TT) -> TT:
        return super().forward(x) * (1 + scale) + shift

class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_attn_input_size: None | int,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        residual_v=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.residual_v = residual_v

        self.norm1 = ModulatedRMSNorm(hidden_size, eps=1e-6, elementwise_affine=qkv_bias)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if residual_v:
            self.lambda_param = nn.Parameter(torch.tensor(0.5).reshape(1))

        # TODO: reduce footprint
        if cross_attn_input_size is not None:
            self.norm2 = ModulatedRMSNorm(hidden_size, eps=1e-6, elementwise_affine=qkv_bias)
            self.q_cross = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
            self.context_kv = nn.Linear(
                cross_attn_input_size, hidden_size * 2, bias=qkv_bias
            )
            self.cross_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.norm2 = None
            self.q_cross = None
            self.context_kv = None
            self.cross_proj = None

        self.norm3 = ModulatedRMSNorm(hidden_size, eps=1e-6, elementwise_affine=qkv_bias)
        mlp_hidden = int(hidden_size * mlp_ratio)
        # TODO: this probably doesn't need biases
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.adaLN_modulation = Modulation(hidden_size, 6 if cross_attn_input_size is None else 9)
        self.adaLN_modulation.reset_parameters()

    def forward(self, x, context, c, v_0=None, rope=None):
        shift_sa, scale_sa, gate_sa, *mod_ca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c[:,None])

        # Self attention
        norm_x = self.norm1(x, shift_sa, scale_sa)
        qkv = self.qkv(norm_x)
        qkv = rearrange(qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)

        if self.residual_v and v_0 is not None:
            v = self.lambda_param * v + (1 - self.lambda_param) * v_0

        if rope is not None:
            q = apply_rotary_emb(q, rope[0], rope[1])
            k = apply_rotary_emb(k, rope[0], rope[1])

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")
        attn_out = self.attn_proj(attn_out)
        x = x + attn_out * gate_sa

        # Cross attention
        if self.norm2 is not None:
            shift_ca, scale_ca, gate_ca = mod_ca
            norm_x = self.norm2(x, shift_ca, scale_ca)

            q = rearrange(
                self.q_cross(norm_x), "b l (h d) -> b h l d", h=self.num_heads
            )
            context_kv = rearrange(
                self.context_kv(context),
                "b l (k h d) -> k b h l d",
                k=2,
                h=self.num_heads,
            )
            context_k, context_v = context_kv.unbind(0)

            cross_out = F.scaled_dot_product_attention(q, context_k, context_v)
            cross_out = rearrange(cross_out, "b h l d -> b l (h d)")
            cross_out = self.cross_proj(cross_out)
            x = x + cross_out * gate_ca

        # MLP
        norm_x = self.norm3(x, shift_mlp, scale_mlp)
        x = x + self.mlp(norm_x) * gate_mlp

        return x, v


# TODO: simplify me
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768, time_patch_size=16):
        super().__init__()
        self.patch_proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(time_patch_size, patch_size, patch_size),
            stride=(time_patch_size, patch_size, patch_size),
        )
        self.patch_size = patch_size
        self.time_patch_size = time_patch_size

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.patch_proj(x)
        x = rearrange(x, "b c t h w -> b (h w t) c")
        return x


class ThreeDimRotary(torch.nn.Module):
    def __init__(self, dim, base=100, h=128, w=128, t=128):
        super().__init__()
        self.inv_freq_space = 1.0 / (base ** (torch.arange(0, dim, 4).float() / (dim)))
        self.inv_freq_time = 1.0 / (base ** (torch.arange(0, dim, 2).float() / (dim)))
        self.h = h
        self.w = w
        self.t = t

        t_h = torch.arange(h).type_as(self.inv_freq_space)
        t_w = torch.arange(w).type_as(self.inv_freq_space)
        t_t = torch.arange(t).type_as(self.inv_freq_time)

        freqs_h = torch.outer(t_h, self.inv_freq_space).reshape(
            1, h, 1, dim // 4
        )  # 1, h, 1, d / 4
        freqs_w = torch.outer(t_w, self.inv_freq_space).reshape(
            1, 1, w, dim // 4
        )  # 1, 1, w, d / 4
        freqs_t = torch.outer(t_t, self.inv_freq_time).reshape(
            t, 1, 1, dim // 2
        )  # t, 1, 1, d / 2
        freqs_h = freqs_h.repeat(t, 1, w, 1)  # t, h, w, d / 4
        freqs_w = freqs_w.repeat(t, h, 1, 1)  # t, h, w, d / 4
        freqs_t = freqs_t.repeat(1, h, w, 1)  # t, h, w, d / 2
        freqs_hwt = torch.cat([freqs_t, freqs_h, freqs_w], 3)  # t, h, w, d

        self.register_buffer("freqs_hwt_cos", freqs_hwt.cos())
        self.register_buffer("freqs_hwt_sin", freqs_hwt.sin())
        self.rng = random.Random()

    def forward(self, time_height_width, extend_with_register_tokens=0):
        # TODO: write kernel because in principle these are static shapes
        this_t, this_h, this_w = time_height_width

        # randomly, we augment the height and width
        start_t = self.rng.randint(0, self.t - this_t)
        start_h = self.rng.randint(0, self.h - this_h)
        start_w = self.rng.randint(0, self.w - this_w)

        cos = self.freqs_hwt_cos[
            start_t : start_t + this_t,
            start_h : start_h + this_h,
            start_w : start_w + this_w,
        ].flatten(0,-2)
        sin = self.freqs_hwt_sin[
            start_t : start_t + this_t,
            start_h : start_h + this_h,
            start_w : start_w + this_w,
        ].flatten(0,-2)

        # append N of zero-attn tokens
        cos = torch.cat([
            torch.ones(extend_with_register_tokens, cos.shape[-1], device=cos.device),
            cos,
        ])
        sin = torch.cat([
            torch.ones(extend_with_register_tokens, sin.shape[-1], device=sin.device),
            sin,
        ])
        return cos[None, None], sin[None, None]  # [1, 1, T + N, Attn-dim]

# TODO: check efficiency
def apply_rotary_emb(x, cos, sin):
    orig_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(dtype=orig_dtype)


class DiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        patch_size=2,
        time_patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=128,
        residual_v=False,
        train_bias_and_rms=True,
        use_rope=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_patch_size = time_patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        assert use_rope, "rope is required"

        self.patch_embed = PatchEmbed(
            patch_size, in_channels, hidden_size, time_patch_size
        )
        self.rope = ThreeDimRotary(
            hidden_size // (2 * num_heads), h=128, w=128, t=128
        )

        self.register_tokens = nn.Parameter(torch.randn(1, 16, hidden_size))

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cross_attn_input_size=cross_attn_input_size,
                    residual_v=residual_v,
                    qkv_bias=train_bias_and_rms,
                )
                for _ in range(depth)
            ]
        )
        self.depth = depth

        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=train_bias_and_rms)
        self.final_proj = nn.Linear(
            hidden_size, patch_size * patch_size * time_patch_size * self.out_channels
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        self.paramstatus = {}
        for n, p in self.named_parameters():
            self.paramstatus[n] = {
                "shape": p.shape,
                "requires_grad": p.requires_grad,
            }


    # segregated method for 2 reasons:
    # 1. uncompilable
    # 2. needed in train loop for CP API
    def get_rope(self, x_shape):
        b, c, t, h, w = x_shape
        return self.rope(
            extend_with_register_tokens=16,
            time_height_width=(
                t // self.time_patch_size,
                h // self.patch_size,
                w // self.patch_size,
            ),
        )

    def forward(self, x, context, timesteps, rope: tuple[TT,TT]):
        # TODO: document shapes and reduce useless ops here...
        b, c, t, h, w = x.shape
        x = self.patch_embed(x)  # b, T, d

        x = torch.cat([self.register_tokens.repeat(b, 1, 1), x], 1)  # b, T + N, d

        t_emb = timestep_embedding(timesteps, self.hidden_size).to(
            x.device, dtype=x.dtype
        )
        t_emb = self.time_embed(t_emb)

        v_0 = None

        for idx, block in enumerate(self.blocks):
            x, v = block(x, context, t_emb, v_0=v_0, rope=rope)
            if v_0 is None:
                v_0 = v

        x = x[:, 16:, :]
        final_shift, final_scale = self.final_modulation(t_emb).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + final_scale[:, None, :]) + final_shift[:, None, :]
        x = self.final_proj(x)

        x = rearrange(
            x,
            "b (h w t) (p1 p2 p3 c) -> b c (t p3) (h p1) (w p2)",
            t=t // self.time_patch_size,
            h=h // self.patch_size,
            w=w // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.time_patch_size,
        )
        return x

    def get_mup_setup(self, learning_rate, weight_decay, constant_param_classes):

        no_decay_name_list = ["bias", "norm", "lambda"]
        custom_lr_multipliers = {"bias": 0.01, "norm": 0.01, "lambda": 0.01}

        optimizer_grouped_parameters = []
        final_optimizer_settings = {}

        param_groups = defaultdict(
            lambda: {"params": [], "weight_decay": None, "lr": None}
        )

        for n, p in self.named_parameters():
            n = n.replace("_fsdp_wrapped_module.", "")
            status = self.paramstatus[n]
            if status["requires_grad"]:
                # Define learning rate for specific types of params
                if any(ndnl in n for ndnl in no_decay_name_list):
                    for ndnl in no_decay_name_list:
                        if ndnl in n:
                            lr_value = learning_rate * custom_lr_multipliers[ndnl]
                            break
                    per_layer_weight_decay_value = 0.0

                else:
                    hidden_dim = status["shape"][-1]
                    if hidden_dim == 0:
                        print(f"hidden_dim: {hidden_dim}")
                        print(f"n: {n}")
                    lr_value = learning_rate * (32 / hidden_dim)
                    per_layer_weight_decay_value = (
                        weight_decay * hidden_dim / 1024
                    )  # weight decay 0.1 (SP: 1024)

                # in the case of embedding layer, we use higher lr.
                if any(
                    constant_param_class in n
                    for constant_param_class in constant_param_classes
                ):
                    lr_value = learning_rate * 0.01
                    per_layer_weight_decay_value = 0.0

                # modify lr once more for specific params
                if "time" in n:
                    lr_value = learning_rate * 0.1
                if "modulation" in n:
                    lr_value = learning_rate * 0.1

                group_key = (lr_value, per_layer_weight_decay_value)
                param_groups[group_key]["params"].append(p)
                param_groups[group_key]["weight_decay"] = per_layer_weight_decay_value
                param_groups[group_key]["lr"] = lr_value

                final_optimizer_settings[n] = {
                    "lr": lr_value,
                    "wd": per_layer_weight_decay_value,
                    "shape": status["shape"],
                }

        optimizer_grouped_parameters = [v for v in param_groups.values()]

        return optimizer_grouped_parameters, final_optimizer_settings


from functools import reduce

import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh


def get_device_mesh():
    tp_size = 1
    dp_replicate = 1
    dp_shard = dist.get_world_size()

    assert (
        dp_replicate * dp_shard * tp_size == dist.get_world_size()
    ), f"dp_replicate * dp_shard * tp_size ({dp_replicate} * {dp_shard} * {tp_size}) != world_size ({dist.get_world_size()})"

    dims = []
    names = []
    if dp_replicate >= 1:
        dims.append(dp_replicate)
        names.append("dp_replicate")
    if dp_shard > 1:
        dims.append(dp_shard)
        names.append("dp_shard")
    if tp_size > 1:
        dims.append(tp_size)
        names.append("tp")
    dims = tuple(dims)
    names = tuple(names)

    return init_device_mesh("cuda", mesh_shape=dims, mesh_dim_names=names)


def get_module(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def set_module(module, access_string, value):
    names = access_string.split(sep=".")
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], value)


def apply_fsdp(dit_model, param_dtype, reduce_dtype):

    device_mesh = get_device_mesh()
    fsdp_config = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "mesh": device_mesh["dp_shard"],
    }

    for block_idx, block in enumerate(dit_model.blocks):

        reshard_after_forward = block_idx < len(dit_model.blocks) - 1

        set_module(
            dit_model,
            f"blocks.{block_idx}",
            fully_shard(
                block, **fsdp_config, reshard_after_forward=reshard_after_forward
            ),
        )

        # fully_shard(
        #     block, **fsdp_config, reshard_after_forward=False
        # )

        # Wrap the entire block

    dit_model = fully_shard(dit_model, **fsdp_config, reshard_after_forward=True)
    return dit_model


if __name__ == "__main__":
    model = DiT(
        in_channels=4,
        patch_size=2,
        time_patch_size=2,
        hidden_size=512,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=128,
        residual_v=False,
        train_bias_and_rms=True,
        use_rope=True,
    ).cuda()
    print(
        model(
            torch.randn(1, 4, 64, 64, 64).cuda(),
            torch.randn(1, 37, 128).cuda(),
            torch.tensor([1.0]).cuda(),
        ).shape
    )
