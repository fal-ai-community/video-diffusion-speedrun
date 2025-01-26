import gc
import logging
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.device_mesh as tdm
import torch.distributed.checkpoint as dcp
import torch.optim as optim
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor.experimental import context_parallel, _attention, implicit_replication
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.nn.attention import sdpa_kernel, SDPBackend

# Enable TF32 for faster training
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
# torch._dynamo.config.capture_scalar_outputs = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_start_method("spawn")

# As of Dec 2024, non-causal CP is broken-by-default unless this flag is set.
_attention._cp_options.enable_load_balance = False

from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
from model import DiT
from utils import (
    avg_scalar_across_ranks,
    async_allgather_stack,
    create_dataloader,
    encode_prompt_with_t5,
    load_encoders,
    limited_tqdm,
    timeit,
)



def create_mesh(fs: int, cp: int) -> tuple[tdm.DeviceMesh, bool]:
    # Create a 6D mesh of appropriate dims, and also return the logging rank.
    ep = pp = tp = 1 # hardcoded unsupported parallelisms
    minsize = fs * cp * pp * tp * ep
    ws = int(os.environ['WORLD_SIZE'])
    dp, remainder = divmod(ws, minsize)
    assert remainder == 0, f"world size {ws} must be divisible by {minsize}"

    # According to the llama3 paper, mesh hierarchy should be TP->CP->PP->DP,
    # but IMO it makes more sense as TP->(EP)->CP->(FS)DP->PP.
    names = ['pp', 'dp', 'fs', 'cp', 'ep', 'tp']
    sizes = [pp, dp, fs, cp, ep, tp]
    mesh = tdm.init_device_mesh('cuda', sizes, mesh_dim_names=names)

    # We always pass FSDP2 a 2D [dp, fscp] mesh,
    mesh['fs','cp']._flatten(mesh_dim_name="fscp")
    # but the data sharding mesh is different -- only [dp, fs, ep] receives different shards of data.
    mesh['dp', 'fs']._flatten(mesh_dim_name="data_unique")
    mesh['cp']._flatten(mesh_dim_name="data_gather")

    # our logging is based on rlast rather than r0, which is helpful for PP.
    is_rlast = all(mesh[k].get_local_rank() == mesh[k].size(0) - 1 for k in names)
    return mesh, is_rlast

def apply_fsdp(model: DiT, fsdp_mesh: tdm.DeviceMesh):
    conf = dict(
        # use bf16 for param + reduce dtype.
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, cast_forward_inputs=True),
        reshard_after_forward=False, # force zero2 for now
        mesh=fsdp_mesh,
        shard_placement_fn=None, # <-- changeme for MoE later.
    )
    # torch._dynamo.config.skip_fsdp_hooks = True
    for block in model.blocks:
        fully_shard(block, **conf)
    fully_shard(model, **conf)

def load_ckpt(m: DiT, checkpoint_path: Path, master_process: bool, *, load_via_pt = True):
    '''
    Simple rules to avoid wasting time on checkpoint reformatting and etc:
    1. always load and save in identical world size
    2. always load and save with identical parallelism
    3. always load and save in either eager only, or compiled only.
       There is a trick to hide _orig_mod from module entities that I may use in the future.
    '''
    # TODO: reimplement me without wasting time on dense ckpt
    assert load_via_pt, "only pt loading is supported for now"

    if master_process:
        if not os.path.exists(checkpoint_path / "temp.pt"):
            dcp_to_torch_save(checkpoint_path, checkpoint_path / "temp.pt")

    dist.barrier()
    state_dict = torch.load(checkpoint_path / "temp.pt", mmap=True, map_location="cuda")
    status = m.load_state_dict(state_dict, assign=True, strict=False)
    del state_dict

    dist.barrier()
    if master_process:
        print(f"All ranks done loading checkpoint, {status}")

def cleanup():
    dist.destroy_process_group()


def ceil_to_multiple(x, multiple):
    return math.ceil(x / multiple) * multiple


@torch.no_grad()
def prompt2context(text_encoder, tokenizer, captions, device, return_index=-1):
    with record_function("encode_prompt_with_t5"):
        caption_encoded = encode_prompt_with_t5(
            text_encoder,
            tokenizer,
            prompt=captions,
            device=device,
            return_index=return_index,
        )
    caption_encoded = caption_encoded.to(torch.bfloat16)

    do_zero_out = torch.rand(caption_encoded.shape[0], device=device) < 0.01
    caption_encoded[do_zero_out] = 0
    return caption_encoded

def forward_plusmaybe_backward(
    mesh_cp: tdm.DeviceMesh,
    dit_model: DiT,
    vae_latent: torch.Tensor,
    caption_encoded: torch.Tensor,
    timeit_r0: callable,
    backward: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with timeit_r0("preprocess"), torch.device("cuda"):
        batch_size = vae_latent.size(0)
        dtype = vae_latent.dtype
        z = torch.randn(batch_size, dtype=dtype)
        t = torch.nn.functional.sigmoid(z)
        # time shift
        alpha = 8.0
        t = t * alpha / (1 + (alpha - 1) * t)

        noise = torch.randn_like(vae_latent)
        tr = t.reshape(batch_size, 1, 1, 1, 1)
        z_t = vae_latent * (1 - tr) + noise * tr
        v_objective = vae_latent - noise
        rope = dit_model.get_rope(z_t.shape)

    '''print(z_t, caption_encoded, t, rope)
    tensor[2, 16, 16, 64, 64] bf16 n=2097152 (4Mb) x∈[-4.344, 4.562] μ=2.933e-05 σ=0.898 cuda:0
    tensor[2, 512, 4096] bf16 n=4194304 (8Mb) x∈[-0.898, 2.031] μ=0.001 σ=0.070 cuda:0
    tensor[2] bf16 μ=0.891 σ=0.033 cuda:0 [0.914, 0.867]
    (tensor[1, 1, 8208, 64] n=525312 (2.0Mb) x∈[-1.000, 1.000] μ=0.064 σ=0.690 cuda:0, tensor[1, 1, 8208, 64] n=525312 (2.0Mb) x∈[-1.000, 1.000] μ=0.228 σ=0.686 cuda:0)

    Note on context parallelism:
    1. it *must* be wrapped around both fwd + bwd to be correct.
    2. if it is used in training, it is almost certainly desired in inference as well.
    3. it *cannot* be activated during T5 encoding, otherwise it might trip HF usage of sdpa.
    so, we use it here.

    With regards to sequence dim partitioning,
    1. we split noise/velocity on time dimension, -3. in the general case,
       I aim to split on the lastmost effective sequence dim, to ensure sdpa gets valid split chunks
    2. the caption is also used in sdpa (xattn), so it must be split by its own seqlen.
    3. rope is directly applied to q/k, so it must also be seq split'd
    See printed shapes above && buffer_seq_dims below for cross-checking.

    We don't restore any buffers because they aren't needed, generally.
    '''
    DiT.mesh_cp = mesh_cp # <-- for register tokens
    with context_parallel(
        mesh_cp,
        buffers=[z_t, v_objective, caption_encoded, rope[0], rope[1]],
        buffer_seq_dims=[-3, -3, -2, -2, -2],
        no_restore_buffers={z_t, v_objective, caption_encoded, rope[0], rope[1]},
    ) if mesh_cp.size(0) > 1 else nullcontext():
        with timeit_r0("forward"), record_function("forward"):
            output = dit_model(z_t, caption_encoded, t, rope=rope)
            diffusion_loss = (v_objective.float() - output.float()).pow(2).mean()
            total_loss = diffusion_loss
        if backward:
            with timeit_r0("bwd"), record_function("backward"):
                total_loss.backward()

    return total_loss, diffusion_loss


@click.command()
@click.option("--fs", type=int, default=1, help="**extra** fsdp2 (z2) shard factor, on top of what CP already uses.")
@click.option("--cp", type=int, default=1, help="context parallel shard factor, which will be applied together with fsdp2 on the same axis.")
@click.option("--dataset", type=click.Choice(["real", "fake"]), default="real", help="Dataset to use")
@click.option("--num_epochs", type=int, default=2, help="Number of training epochs")
@click.option("--batch_size", type=int, default=64, help="Batch size for training")
@click.option("--learning_rate", type=float, default=1e-4, help="Learning rate")
@click.option("--max_steps", type=int, default=10000, help="Maximum training steps")
@click.option(
    "--evaluate_every", type=int, default=20, help="Steps between evaluations"
)
@click.option("--run_name", type=str, default="diffusion_repa", help="Name of run")
@click.option("--model_width", type=int, default=512, help="Width of the model")
@click.option("--model_depth", type=int, default=9, help="Depth of the model")
@click.option(
    "--model_head_dim", type=int, default=128, help="Head dimension of the model"
)
@click.option(
    "--compile-strat", type=click.Choice(["eager", "block", "full"]), default="block",
    help="""torch.compile strategy. Offers the following options:
    - eager: compile the entire model
    - block: compile DiT blocks only
    - full: compile the entire model && T5 enc
The default option is *block* -- dynamo only needs to trace 1 block once, which
makes compilation relatively fast (<20s). You should use *full* for full real runs,
and use *eager* if actively debugging modelling code.
"""
)
@click.option("--optimizer_type", type=str, default="mup_adam", help="Optimizer type")
@click.option(
    "--lr_scheduler_type",
    type=str,
    default="cosine",
    help="Learning rate scheduler type",
)
@click.option(
    "--train_bias_and_rms",
    type=bool,
    default=False,
    help="Use unlearnable rms and bias",
)
@click.option(
    "--init_std_factor", type=float, default=0.1, help="Factor to scale init std"
)
@click.option(
    "--project_name", type=str, default="test_diffusion_test", help="Project name"
)
@click.option(
    "--profile-dir", type=str, default="", help="enable pytorch profiler (will crash), write to directory"
)
@click.option(
    "--return_index",
    type=int,
    default=-8,
    help="Return index for T5 encoding. Default is -1 which returns the last state.",
)
@click.option(
    "--load_checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint to load",
)
def train_fsdp(
    fs,
    cp,
    dataset,
    num_epochs,
    batch_size,
    learning_rate,
    max_steps,
    evaluate_every,
    run_name,
    model_width,
    model_depth,
    model_head_dim,
    compile_strat,
    optimizer_type,
    lr_scheduler_type,
    train_bias_and_rms,
    init_std_factor,
    project_name,
    profile_dir,
    return_index,
    load_checkpoint,
):
    # Initialize distributed training
    assert int(os.environ["WORLD_SIZE"]) > 1, "trainer doesn't handle single GPU"
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(device := f"cuda:{ddp_local_rank}")
    mesh, master_process = create_mesh(fs=fs, cp=cp)

    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if master_process:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    print_fn = (lambda x: logger.info(x)) if master_process else (lambda _:0)
    timeit_r0 = lambda s: timeit(s, print_fn=print_fn)

    tokenizer, text_encoder = load_encoders(device=device, compile_models=compile_strat == "full")
    model_conf = dict(
        in_channels=16,
        patch_size=2,
        depth=model_depth,
        num_heads=model_width // model_head_dim,
        mlp_ratio=4.0,
        cross_attn_input_size=4096,
        hidden_size=model_width,
        residual_v=True,
        train_bias_and_rms=train_bias_and_rms,
        use_rope=True,
    )

    with torch.device('cuda'), torch.random.fork_rng(['cuda']), timeit_r0("init model"):
        # *All ranks* receive the same starting seed for param initialization.
        torch.manual_seed(0)

        # initialize single-device on cuda. this is cheap and shouldn't be done at larger scale.
        dit_model = DiT(**model_conf).train()
        # TODO: ask simo how this is supposed to work. isn't default kaiming uniform?
        # initialize 2d params with normal fan_in
        for _, param in dit_model.named_parameters():
            if len(param.shape) == 2:
                fan_in = param.shape[1]
                param.data.mul_(init_std_factor)

        model_size = sum(p.numel() for p in dit_model.parameters() if p.requires_grad)
        print(f"Number of parameters (pre-sharded): {model_size}, {model_size / 1e6}M")

        # torch._dynamo.config.cache_size_limit = 8
        if compile_strat != "eager":
            if compile_strat == "full": dit_model = torch.compile(dit_model)
            # Don't bother using dynamic shapes, because we have constant shapes.
            torch._dynamo.config.dynamic_shapes = False
            dit_model.apply_compile(True, False)

        if load_checkpoint is not None:
            load_ckpt(dit_model, Path(load_checkpoint), master_process)

        apply_fsdp(dit_model, mesh["dp", "fscp"])


    if master_process:
        print(f"batch_size: {batch_size}")
        print(f"model_width: {model_width}")
        print(f"model_depth: {model_depth}")
        print(f"model_head_dim: {model_head_dim}")
        print(f"train_bias_and_rms: {train_bias_and_rms}")
        print(f"init_std_factor: {init_std_factor}")
        print(f"optimizer_type: {optimizer_type}")
        print(f"learning_rate: {learning_rate}")
        print(f"lr_scheduler_type: {lr_scheduler_type}")
        print(f"return_index: {return_index}")
        print(f"project_name: {project_name}")

    if master_process:
        if os.getenv("WANDB_API_KEY") is not None:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "model_parameters": model_size / 1e6,
                "model_width": model_width,
                "model_depth": model_depth,
                "model_head_dim": model_head_dim,
                "train_bias_and_rms": train_bias_and_rms,
            },
        )

    # Wrap model in FSDP
    constant_param_name = ["patch_embed", "context_kv", "positional_embedding"]

    # Initialize optimizer and scheduler
    if optimizer_type == "mup_adam":
        optimizer_grouped_parameters, final_optimizer_settings = (
            dit_model.get_mup_setup(learning_rate, 1e-1, constant_param_name)
        )

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.95, 0.99),
            fused=True,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    num_warmup_steps = 20

    if lr_scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, max_steps
        )
    elif lr_scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, max_steps
        )
    elif lr_scheduler_type == "constant":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, 10000000000
        )
    else:
        raise ValueError(f"Unknown lr scheduler type: {lr_scheduler_type}")

    train_loader = create_dataloader(
        dataset,
        "train",
        batch_size,
        num_workers=8,
        do_shuffle=True,
        prefetch_factor=4,
        device=device,
    )

    test_loader = create_dataloader(
        dataset,
        "train", # test
        batch_size,
        num_workers=1,
        do_shuffle=False,
        device=device,
    )

    # Training loop
    dit_model.train()

    # (TODO) do this every k steps like torchtitan
    # clear up memory
    torch.cuda.empty_cache()
    gc.collect()

    if profile_dir:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sched = schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=3)
        def handler(p: profile, s=[0]):
            p.export_chrome_trace(f"{profile_dir}/chrometrace-{s[0]}.json")
            s[0] += 1
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        prof_ctx = profile(
            activities=activities, schedule=sched, on_trace_ready=handler, with_stack=True
        ) if master_process and profile else nullcontext()
        prof = prof_ctx.__enter__()
    else:
        prof = type('',(),dict(step=lambda:0))

    '''sdp cudnn backend not working:
    [rank0]:   File "/tmp/torchinductor_ubuntu/m4/cm4svcidkqacmkfbaxf7wnrenhwjmcknezqjleka54t3tpp4wydk.py", line 1439, in call
    [rank0]:     assert_size_stride(getitem_23, (1, 4, 8208, 128), (4202496, 1050624, 128, 1))
    [rank0]: AssertionError: expected size 4==4, stride 128==1050624 at dim=1; expected size 8208==8208, stride 512==128 at dim=2
    # with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    '''
    def multiepoch_loop(*, step = 0):
        for e in range(num_epochs):
            for i, d in enumerate(train_loader):
                x, c = d['latent'].cuda(), d['prompt']

                # TODO: have a replication PG in LavenderStreamingDataset.
                if mesh['data_gather'].size(0) != 1:
                    gather_pg = mesh['data_gather'].get_group()
                    x = async_allgather_stack(x, gather_pg)()
                    prompts = [None] * gather_pg.size()
                    dist.all_gather_object(prompts, c, group=gather_pg)
                    c = [p for ls in prompts for p in ls]
                # gathering this way defeats the entire point of CP, it's just a demonstration.

                yield (e,step), (x,c)
                step += 1
                if step >= max_steps: return

    # For training purposes: all processes with *different* batches must have *different* random seeding,
    data_seed = mesh['data_unique'].get_local_rank()
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    # and all processes that have the same microbatch **must** have the same seed.

    # This is also a warning to implementers: be CERTAIN you aren't rank-conditionally
    # executing any global random methods in your model's forward pass. Some easy ways to fuck this up:
    # * executing any random methods at all, in rank0/master_process-only code.
    # * using double-random conditions [e.g. if random() > 0.5: x = random(...)]
    # You can also protect yourself against this kind of trouble with `torch.random.fork_rng(...)`.

    time_for_10_steps = time.time()
    for (epoch, step), (x, c) in multiepoch_loop():
        e = prompt2context(text_encoder, tokenizer, c, device, return_index=-1)
        total_loss, diffusion_loss = forward_plusmaybe_backward(mesh['cp'], dit_model, x, e, timeit_r0, backward=True)
        with timeit_r0("optim/lr"):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % 10 == 0:
            step_time = (time.time() - time_for_10_steps) / 10
            time_for_10_steps = time.time()

            dist.all_reduce(diffusion_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            diffusion_loss, total_loss = diffusion_loss.item(), total_loss.item()

            if master_process:
                print(f"Avg fwdbwd steps: {step_time*1000:.2f}ms")
                wandb.log({
                    "train/diffusion_loss": diffusion_loss,
                    "train/total_loss": total_loss,
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": step,
                })
                logger.info(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{step}/{max_steps}] "
                    f"Loss: {total_loss:.4f} "
                    f"(Diffusion: {diffusion_loss:.4f}) "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.4f}"
                )

        if step % evaluate_every == 0:
            ### Evaluation ###
            dit_model.eval()

            ## (Evaluation loop) ##
            total_losses = []
            diffusion_losses = []
            with torch.random.fork_rng(['cuda']), torch.no_grad():
                # TODO: we don't account for CP here. Instead we:
                # 1. assume the test loader yields unique data per rank,
                # 2. assume mesh['pp'] will be of size 1 and so CP won't trigger,
                # 3. assume we have enough vram to do inference without CP.
                # this makes the loss computation equivalent to the non-CP case, which is conveinent
                # because our train loss actually represents a different distribution per-CP-rank
                torch.manual_seed(dist.get_rank())
                for batch_idx, batch in limited_tqdm(enumerate(test_loader), total=9):
                    e = prompt2context(text_encoder, tokenizer, batch["prompt"], device, return_index=-1)
                    total_loss, diffusion_loss = forward_plusmaybe_backward(
                        mesh['pp'], dit_model, batch["latent"], e, timeit_r0, backward=False
                    )
                    total_losses.append(total_loss.item())
                    diffusion_losses.append(diffusion_loss.item())
                    print(f"Eval, Batch {batch_idx} done, {total_losses[-1]}, {diffusion_losses[-1]}")

            dit_model.train()
            ## (End of evaluation loop) ##

            ## log evals
            dist.barrier()
            total_loss = avg_scalar_across_ranks(np.mean(total_losses).item())
            diffusion_loss = avg_scalar_across_ranks(
                np.mean(diffusion_losses).item()
            )
            if master_process:
                wandb.log({
                    "test/total_loss": total_loss,
                    "test/diffusion_loss": diffusion_loss,
                })

            ## Checkpointing ##
            state_dict = get_model_state_dict(dit_model)

            # TODO: clean up execution environment assumptions
            if master_process:
                os.makedirs("checkpoints", exist_ok=True)
                os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                os.makedirs(f"checkpoints/{run_name}/{step}", exist_ok=True)
                # Save model state dict
                print(f"Saving model state dict to checkpoints/{run_name}/{step}")

            dcp.save(
                state_dict,
                checkpoint_id=Path(f"checkpoints/{run_name}/{step}"),
            )
            print(f"Epoch {epoch} done")
            print(f"Global step {step}")

    # Cleanup
    if master_process:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    train_fsdp()
