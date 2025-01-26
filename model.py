# DiT with cross attention

import random
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed._functional_collectives as funcol
from einops import rearrange, pack
from torch import nn, Tensor as TT
from torch.distributed.tensor import DTensor, Replicate, Shard, device_mesh as tdm
import torch._dynamo as _D


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

class PatchEmbed(nn.Conv3d):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768, time_patch_size=16):
        super().__init__(
            in_channels,
            embed_dim,
            kernel_size=(time_patch_size, patch_size, patch_size),
            stride=(time_patch_size, patch_size, patch_size),
        )
    def forward(self, x: TT):
        return rearrange(super().forward(x), "b c t h w -> b (h w t) c")

# TODO: comprehend 3d RoPE correctly so that I don't mess up CP.
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
        self.rng = torch.Generator('cuda') # <-- TODO: verify RNG dissimilarity across devices

    def forward(self, time_height_width, extend_with_register_tokens=0):
        # TODO: write kernel because in principle these are static shapes
        this_t, this_h, this_w = time_height_width

        # randomly, we augment the height and width
        start_t = torch.randint(0, self.t - this_t + 1, (), generator=self.rng)
        start_h = torch.randint(0, self.h - this_h + 1, (), generator=self.rng)
        start_w = torch.randint(0, self.w - this_w + 1, (), generator=self.rng)

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
            torch.ones(extend_with_register_tokens, cos.shape[-1]),
            cos,
        ])
        sin = torch.cat([
            torch.ones(extend_with_register_tokens, sin.shape[-1]),
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

'''
TODO: it is extremely dumb to do all the work of copying over the large tensor to move some small register buffer.
Ideally, the conv3d would be "register-aware" and allocate some padding space in the output tensor. but can we do that?
'''

class RegisterTokens(nn.Parameter):
    # Sequence-parallelizable register tokens
    @staticmethod
    def calc(self, mesh_cp: tdm.DeviceMesh) -> int:
        cp,rank = mesh_cp.size(0), mesh_cp.get_local_rank()
        regs = self.size(-2)
        extra, remainder = divmod(regs, cp)
        assert remainder == 0, "register token count must be divisible by cp"
        return cp

    ### TODO: more efficient approach:
    # # the entire sequence is "pushed forward" by `regs`, so each rank needs `regs/cp` more space.
    # final_local_slen = x.size(-2) + extra
    # out = torch.empty(x.size(0), final_local_slen, *x.shape[2:], device=x.device, dtype=x.dtype)
    # i_start = regs - extra * rank
    # j_stop = final_local_slen - regs + extra * (rank+1)
    # out[:, i_start:] = x[:, :j_stop]
    # if rank == 0:
    #     out[:, :regs] = self.register_tokens
    # send_len = 
    ### end TODO

    @staticmethod
    def pack(self, x: TT, mesh_cp: tdm.DeviceMesh | None) -> TT:
        # Starting assumption: x was context_parallel split'd and conv'd
        registers = self.repeat(x.size(0), 1, 1)
        if mesh_cp is None or (cp := RegisterTokens.calc(self, mesh_cp)) == 1:
            return torch.cat([registers, x], dim=-2)

        # EXTREMELY LAZY SOLUTION: allgather full seq, then redistribute (slice in fwd, allgather in bwd)
        x_gather = funcol.all_gather_tensor_autograd(x, -2, mesh_cp.get_group())
        x_gather = torch.cat([registers, x_gather], dim=-2)
        if x_gather.requires_grad: # I don't understand why this is mathematically required.
            x_gather.register_hook(lambda x: x / cp)
        x_dt = DTensor.from_local(x_gather, mesh_cp, placements=[Replicate()])
        x_dt = x_dt.redistribute(placements=[Shard(dim=-2)])
        return x_dt.to_local()
    @staticmethod
    def unpack(self, x: TT, mesh_cp: tdm.DeviceMesh | None) -> TT:
        if mesh_cp is None or (cp := RegisterTokens.calc(self, mesh_cp)) == 1:
            return x[:, self.size(-2):]

        # EXTREMELY LAZY SOLUTION: allgather full seq, then redistribute (slice in fwd, allgather in bwd)
        if x.requires_grad: # I don't understand why this is mathematically required.
            x.register_hook(lambda x: x / cp)
        x_gather = funcol.all_gather_tensor_autograd(x, -2, mesh_cp.get_group())
        x_gather = x_gather[:, self.size(-2):]
        x_dt = DTensor.from_local(x_gather, mesh_cp, placements=[Replicate()])
        x_dt = x_dt.redistribute(placements=[Shard(dim=-2)])
        return x_dt.to_local()
    @staticmethod
    def test2(ws: int):
        from debug import printflock, leave, dist, NoZeroInit
        from lovely_tensors import set_config
        set_config(precision=4, sci_mode=True, color=True)
        # Initialization
        mesh_cp = tdm.init_device_mesh("cuda", (ws,), mesh_dim_names=("cp",))
        torch.cuda.set_device(r := mesh_cp.get_local_rank())

        # Models / inputs / outputs
        B, C, S, D, R = 4, 16, 512, 1024, 16
        with torch.device('cuda'), torch.random.fork_rng(['cuda']), NoZeroInit():
            torch.set_default_dtype(torch.bfloat16)
            torch.manual_seed(0)
            m = DiT(
                in_channels=16,
                patch_size=2,
                depth=2,
                num_heads=D // 128,
                mlp_ratio=4.0,
                cross_attn_input_size=4096,
                hidden_size=D,
                residual_v=True,
                train_bias_and_rms=True,
                use_rope=True,
            )
            t = torch.rand(B)
            z_t = torch.randn(B, C, 16, 64, 64)
            v_objective = torch.randn_like(z_t)
            caption_encoded = torch.randn(B, S, 4096)
            rope = m.get_rope(z_t.shape)
            torch.set_default_dtype(torch.float32)
        for n, p in m.named_parameters():
            assert p.nonzero().numel() > 0, f"Parameter {n} ({p}) contains all zeros"

        if dist.get_rank() == 0:
            printflock(t)
            printflock(z_t)
            printflock(v_objective)
            printflock(caption_encoded)
            printflock(rope)
        dist.barrier()

        DiT.mesh_cp = mesh_cp # <-- for register tokens
        from torch.distributed.tensor.experimental import context_parallel, _attention, implicit_replication
        from contextlib import nullcontext
        _attention._cp_options.enable_load_balance = False
        with context_parallel(
            mesh_cp,
            buffers=[z_t, v_objective, caption_encoded, rope[0], rope[1]],
            buffer_seq_dims=[-3, -3, -2, -2, -2],
            no_restore_buffers={z_t, v_objective, caption_encoded, rope[0], rope[1]},
        ) if mesh_cp.size(0) > 1 else nullcontext():
            output = m(z_t, caption_encoded, t, rope=rope)
            diffusion_loss = (v_objective.float() - output.float()).pow(2).mean()
            diffusion_loss.backward()
        for n,p in m.named_parameters():
            if p.grad is None:
                print(f"WARNING: {n} has no grad") if dist.get_rank() == 0 else None
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=mesh_cp.get_group())
            if dist.get_rank() == 0:
                print(p.grad)

    @staticmethod
    def test(ws: int=2):
        """
        Numerical test to demonstrate that all *weights* receive the same gradients for a faux simplified model.

        Specifically, we test that when executing:
            ```
            F.mse_loss(
                Linear_aft(
                    unpack(
                        registers.mean() * Linear_mid(
                            pack(Linear_pre(x))
                        )
                    )
                ), y
            ).backward()
            ```
        The gradients for all weights before/within/after packing, using:
        - randn x,y on 1 GPU
        - *sharded* x,y across `ws` GPUs
        are identical.

        *If* this test passes, *then* the correct way to use CP with our DiT is to:
        - context_parallel shard all inputs *before* forward()
        - calculate losses with *sharded* outputs -- meaning that reported loss is also *sharded*
        """
        from debug import printflock, leave, dist
        # Initialization
        mesh = tdm.init_device_mesh("cuda", (ws,), mesh_dim_names=("cp",))
        torch.cuda.set_device(r := mesh.get_local_rank())

        # Models / inputs / outputs
        B, S, D, R = 4, 512, 1024, 16
        with torch.device('cuda'), torch.random.fork_rng(['cuda']):
            torch.manual_seed(0)
            pre = nn.Linear(D,D,bias=False)
            mid = nn.Linear(D,D,bias=False)
            self = RegisterTokens(torch.randn(1, R, D))
            aft = nn.Linear(D,D,bias=False)
            x = torch.randn(B, S, D, requires_grad=True)
            x_dist = torch.empty(B, S//ws, D, requires_grad=True)
            with torch.no_grad(): x_dist.copy_(x.chunk(ws,dim=-2)[r])
            y_dist = (y := torch.randn_like(x)).chunk(ws, dim=-2)[r]

        # Rank 0 test: x --Linear--> x --packing--> x_r0_packed --Linear*p.mean()--> x_r0_packed --unpack--> x_r0_unpack --Linear--> x_r0_unpack
        x = pre(x)
        x_r0_packed = torch.cat([self.repeat(B,1,1), x], dim=-2)
        x_r0_packed = mid(x_r0_packed) * self.mean()
        x_r0_unpack = x_r0_packed[:,R:]
        x_r0_unpack = aft(x_r0_unpack)
        F.mse_loss(x_r0_unpack, y).backward()

        # Clear/save grads
        pre_r0_grad = pre.weight.grad.clone(); pre.weight.grad = None
        reg_r0_grad = self.grad.clone(); self.grad = None
        lin_r0_grad = mid.weight.grad.clone(); mid.weight.grad = None
        aft_r0_grad = aft.weight.grad.clone(); aft.weight.grad = None

        # CP test: x --Linear--> x --packing--> x_r0_packed --Linear*p.mean()--> x_r0_packed --unpack--> x_r0_unpack --Linear--> x_r0_unpack
        x_dist = pre(x_dist)
        x_dist_packed = RegisterTokens.pack(self, x_dist, mesh)
        x_dist_packed = mid(x_dist_packed) * self.mean()
        x_dist_unpack = RegisterTokens.unpack(self, x_dist_packed, mesh)
        x_dist_unpack = aft(x_dist_unpack)
        F.mse_loss(x_dist_unpack, y_dist).backward()

        # Clear/save/reduce (as ddp would) grads
        pre_dist_grad = pre.weight.grad.clone(); pre.weight.grad = None
        reg_dist_grad = self.grad.clone(); self.grad = None
        lin_dist_grad = mid.weight.grad.clone(); mid.weight.grad = None
        aft_dist_grad = aft.weight.grad.clone(); aft.weight.grad = None
        dist.all_reduce(reg_dist_grad, op=dist.ReduceOp.AVG, group=mesh.get_group())
        dist.all_reduce(lin_dist_grad, op=dist.ReduceOp.AVG, group=mesh.get_group())
        dist.all_reduce(pre_dist_grad, op=dist.ReduceOp.AVG, group=mesh.get_group())
        dist.all_reduce(aft_dist_grad, op=dist.ReduceOp.AVG, group=mesh.get_group())

        assert torch.allclose(x_r0_unpack.chunk(ws, dim=-2)[r], x_dist_unpack)
        assert torch.allclose(pre_r0_grad, pre_dist_grad)
        assert torch.allclose(reg_r0_grad, reg_dist_grad)
        assert torch.allclose(lin_r0_grad, lin_dist_grad)
        assert torch.allclose(aft_r0_grad, aft_dist_grad)

        leave()

class DiTStack(nn.ModuleList):
    # wrapper module containing DiT blocks. useful for input conversion to/from DTensor.
    def __init__(self, d: int, l: int, *, n_heads: int, r_mlp: float, xattn_dim: int, residual_v: bool, qkv_bias: bool):
        super().__init__([
            DiTBlock(
                hidden_size=d,
                num_heads=n_heads,
                mlp_ratio=r_mlp,
                cross_attn_input_size=xattn_dim,
                residual_v=residual_v,
                qkv_bias=qkv_bias,
            )
            for _ in range(l)
        ])
        self.seqlen_bounds = dict(min=8192, max=32768) # inclusive

    @torch.compiler.disable(recursive=False)
    def forward(self, x: TT, c: TT, t: TT, rope: tuple[TT,TT]):
        if _D.config.dynamic_shapes:
            # Unfortunately, dynamo doesn't support negative dim indices,
            # so we use 1/2 instead of -2 (which is reliably seq dim)
            _D.mark_dynamic(x, 1, **self.seqlen_bounds)
            _D.mark_dynamic(rope[0], 2, **self.seqlen_bounds)
            _D.mark_dynamic(rope[1], 2, **self.seqlen_bounds)

        v_0 = None
        for i, block in enumerate(self):
            x, v = block(x, c, t, v_0=v_0, rope=rope)
            if v_0 is None:
                v_0 = v
        return x, v_0

class DiT(nn.Module):
    mesh_cp = None
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

        self.register_tokens = RegisterTokens(torch.randn(1, 16, hidden_size))

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.blocks = DiTStack(
            hidden_size, depth,
            n_heads=num_heads,
            r_mlp=mlp_ratio,
            xattn_dim=cross_attn_input_size,
            residual_v=residual_v,
            qkv_bias=train_bias_and_rms,
        )

        self.final_modulation = Modulation(hidden_size, 2)
        self.final_norm = ModulatedRMSNorm(hidden_size, eps=1e-6, elementwise_affine=train_bias_and_rms)
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

    """A note on compilation:
    The fundamental reason why full torch compilation always takes ridiculously long, is because
    Dynamo *has* to trace the full execution graph for correctness; there is no way to inform
    the compiler that each block repeats the same code in every circumstance
    (and in fact, we do not, because of value residuals!)

    Torchtitan side-steps this problem by doing per-transformer-block compilation, which
    1. allows dynamo to cache work && only do compilation work for 1 fwd/bwd block.
    2. avoids extra work done in attempting to trace FSDP/DDP comms (which happen per block),
       and instead simply lets them execute in eager.
    3. still provides 90% of performance gains from compilation, because >95% of flops are in blocks.

    The above strategy is great for a pure LLM, but a typical diffusion model does significant
    work outside of the pure transformer.
    Leaving timestep emb, encoder, convs, rope, flow objective, etc. out of compile is not a good idea.
    Also, we want to support dynamic shapes without recompilation w/ `mark_dynamic`, but this is
    much harder to pull off with full model compilation, because [slack mentioned poor tracing].

    So, the actual compilation approach currently looks like this:
    - By default, we always compile each DiTBlock to be fullgraph, with dynamic marked inputs.
      We also *try* to compile 3dRoPE, but random shape augmentations are non-fullgraphable.
    - If 'full' compilation is specified, we compile the full DiT module, *but*
      we gate the transformer backbone with a torch.compiler.disable(recursive=False).
      We also compile T5 externally in train.py.
    """
    def apply_compile(self, fullgraph: bool, dynamic: bool | None=None):
        for i, block in enumerate(self.blocks):
            if isinstance(block, DiTBlock):
                self.blocks[i] = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic)
        self.rope = torch.compile(self.rope, fullgraph=False, dynamic=True)

    # segregated method for 2 reasons:
    # 1. not fullgraphable
    # 2. needed in train loop for CP API
    def get_rope(self, x_shape):
        b, c, t, h, w = x_shape
        with torch.device('cuda'):
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
        x = RegisterTokens.pack(self.register_tokens, x, DiT.mesh_cp)
        # TODO: cache timestep freqs
        t_emb = timestep_embedding(timesteps, self.hidden_size).to(
            x.device, dtype=x.dtype
        )
        t_emb = self.time_embed(t_emb)

        x, _ = self.blocks(x, context, t_emb, rope=rope)

        x = RegisterTokens.unpack(self.register_tokens, x, DiT.mesh_cp)
        final_shift, final_scale = self.final_modulation(t_emb[:,None])
        x = self.final_norm(x, final_shift, final_scale)

        x = rearrange(
            self.final_proj(x),
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
            n = n.replace("_orig_mod.", "")
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





### fwd-bwd profiling code (deleteme later) ###
import torch.distributed as dist

def create_dummy(d=1024,l=16): return DiT(
    in_channels=16,
    patch_size=2,
    hidden_size=d,
    depth=l,
    num_heads=d // 128,
    cross_attn_input_size=4096,
    mlp_ratio=4.0,
    residual_v=True,
    train_bias_and_rms=False, #???
    use_rope=True,
)

def fakeinput(bs: int, c: int=16, t: int=16, h: int=64, w: int=64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(bs, c, t, h, w), torch.randn(bs, 512, 4096), torch.rand(bs)

def rank(): return dist.get_rank() if dist.is_initialized() else 0

__import__("lovely_tensors").monkey_patch()
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
from torch.profiler import schedule, profile, ProfilerActivity, record_function

def tqdm_with_step(prof: profile, iters: int, **k):
    for i in tqdm(range(iters)):
        yield i
        prof.step()

@contextmanager
def profiler_setup(path_ct: Path, iters: int):
    """Sets up profiler and yields a generator to iterate over"""
    path_ct.mkdir(parents=True, exist_ok=True)
    sched = schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=3)
    activ = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    # kwarg = dict(record_shapes=True, profile_memory=True, with_stack=True)
    kwarg = dict(with_stack=True)
    def handler(p: profile):
        p.export_chrome_trace(str(path_ct / f"rank-{rank()}_step-{p.step_num}.json"))

    with profile(activities=activ, schedule=sched, on_trace_ready=handler, **kwarg) as prof:
        yield tqdm_with_step(prof, iters)

def bench(comp: bool, dshapes: bool):
    B, D, L = 2, 4096, 4
    # spawn model
    m = create_dummy(D, L)
    if comp:
        m = torch.compile(m)
        m.apply_compile(True, dynamic=None if dshapes else False)

    # if comp: m.apply_compile(True, dynamic=None if dshapes else False)
    _D.config.dynamic_shapes = dshapes
    with profiler_setup(Path(f'./chrometrace-{dshapes=}-{D}-{comp=}'), 30) as g:
        for i in g:
            inp = fakeinput(B, h=64 if i%2 else 128)
            torch.cuda.synchronize()
            with record_function("fwd"): o = m(*inp, rope=m.get_rope(inp[0].shape))
            with record_function("bwd"): o.sum().backward()
            for p in m.parameters(): p.grad = None

if __name__ == "__main__":
    if (ws := int(__import__("os").environ.get("WORLD_SIZE", 1))) > 1:
        RegisterTokens.test2(ws)
        # RegisterTokens.test(ws)
    else:
        RegisterTokens.test2(ws)
        # with torch.device('cuda'), torch.autocast('cuda', torch.bfloat16):
        #     bench(True, True)