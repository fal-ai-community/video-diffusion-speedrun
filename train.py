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
    create_dataloader,
    encode_prompt_with_t5,
    load_encoders,
    limited_tqdm,
)

CAPTURE_INPUT = False


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


def forward(
    dit_model: DiT,
    batch,
    text_encoder,
    tokenizer,
    device,
    global_step,
    master_process,
    generator=None,
    batch_size=None,
    return_index=-1,
):
    logger = logging.getLogger(__name__)

    captions = batch["prompt"]

    images_vae = batch["latent"]

    # print(captions, images_vae.shape)

    preprocess_start = time.time()
    vae_latent = images_vae.to(device).to(torch.bfloat16)

    with torch.no_grad():
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

    batch_size = vae_latent.size(0)
    z = torch.randn(
        batch_size, device=device, dtype=torch.bfloat16, generator=generator
    )
    t = torch.nn.Sigmoid()(z)
    # time shift
    alpha = 8.0
    t = t * alpha / (1 + (alpha - 1) * t)

    if CAPTURE_INPUT and master_process and global_step == 0:
        torch.save(vae_latent, f"test_data/vae_latent_{global_step}.pt")
        torch.save(caption_encoded, f"test_data/caption_encoded_{global_step}.pt")
        torch.save(t, f"test_data/timesteps_{global_step}.pt")

    noise = torch.randn(
        vae_latent.shape, device=device, dtype=torch.bfloat16, generator=generator
    )

    preprocess_time = time.time() - preprocess_start

    if master_process:
        logger.info(f"Preprocessing took {preprocess_time*1000:.2f}ms")

    forward_start = time.time()

    # Forward pass
    tr = t.reshape(batch_size, 1, 1, 1, 1)
    z_t = vae_latent * (1 - tr) + noise * tr
    v_objective = vae_latent - noise
    rope = dit_model.get_rope(z_t.shape)

    with record_function("dit_model"):
        output = dit_model(z_t, caption_encoded, t, rope=rope)

    diffusion_loss_batchwise = (
        (v_objective.float() - output.float()).pow(2).mean(dim=(1, 2, 3, 4))
    )

    diffusion_loss = diffusion_loss_batchwise.mean()
    total_loss = diffusion_loss

    forward_time = time.time() - forward_start
    if master_process:
        logger.info(f"Forward pass took {forward_time*1000:.2f}ms")

    return total_loss, diffusion_loss


@click.command()
@click.option("--fs", type=int, default=1, help="fsdp2 (z2) shard factor")
@click.option("--cp", type=int, default=1, help="context parallel shard factor")
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

    with torch.device('cuda'), torch.random.fork_rng(['cuda']):
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

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if master_process:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Initialize step counter
    global_step = 0

    # Training loop
    dit_model.train()

    diffusion_loss_binning = {k: 0 for k in range(10)}
    diffusion_loss_binning_count = {k: 0 for k in range(10)}

    time_for_10_steps = time.time()

    # clear up memory
    torch.cuda.empty_cache()
    gc.collect()

    if profile_dir:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sched = schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=3)
        handler = lambda p: p.export_chrome_trace(f"{profile_dir}/chrometrace-{global_step}.json")
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        prof_ctx = profile(
            activities=activities, schedule=sched, on_trace_ready=handler, with_stack=True
        ) if master_process and profile else nullcontext()
        prof = prof_ctx.__enter__()
    else:
        prof = type('',(),dict(step=lambda:0))


    for epoch in range(num_epochs):
        if global_step >= max_steps: break

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps: break

            total_loss, diffusion_loss = forward(
                dit_model,
                batch,
                text_encoder,
                tokenizer,
                device,
                global_step,
                master_process,
                generator=None,
                binnings=(
                    diffusion_loss_binning,
                    diffusion_loss_binning_count,
                ),
                batch_size=batch_size,
                return_index=return_index,
            )

            # Optimization step
            backward_start = time.time()
            with record_function("backward"):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                lr_scheduler.step()
            backward_time = time.time() - backward_start

            if master_process:
                logger.info(f"Backward pass took {backward_time*1000:.2f}ms")
                prof.step()

            # Logging
            if global_step % 10 == 0:

                time_for_10_steps = time.time() - time_for_10_steps
                time_for_10_steps = time_for_10_steps / 10

                diffusion_loss = avg_scalar_across_ranks(diffusion_loss.item())
                total_loss = avg_scalar_across_ranks(total_loss.item())

                if master_process:
                    # Calculate average losses per timestep bin
                    print(f"Avg fwdbwd steps: {time_for_10_steps*1000:.2f}ms")

                    def get_binned_averages(loss_dict, count_dict):
                        return {
                            k: v / max(c, 1)
                            for k, v, c in zip(
                                loss_dict.keys(),
                                loss_dict.values(),
                                count_dict.values(),
                            )
                        }

                    diffusion_binned = get_binned_averages(
                        diffusion_loss_binning, diffusion_loss_binning_count
                    )

                    # Log metrics to wandb
                    wandb.log(
                        {
                            "train/diffusion_loss": diffusion_loss,
                            "train/total_loss": total_loss,
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "train_binning/diffusion_loss_binning": diffusion_binned,
                        }
                    )

                    # Format per-timestep losses for logging
                    def format_timestep_losses(binned_losses):
                        return "\n\t".join(
                            f"{k}: {v:.4f}" for k, v in binned_losses.items()
                        )

                    diffusion_per_timestep = format_timestep_losses(diffusion_binned)

                    logger.info(
                        f"Epoch [{epoch}/{num_epochs}] "
                        f"Step [{global_step}/{max_steps}] "
                        f"Loss: {total_loss:.4f} "
                        f"(Diffusion: {diffusion_loss:.4f}) "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.4f}"
                        f"\nDiffusion Per-timestep-binned:\n{diffusion_per_timestep}"
                    )

                    # Reset binning counters
                    diffusion_loss_binning = {k: 0 for k in range(10)}
                    diffusion_loss_binning_count = {k: 0 for k in range(10)}

                time_for_10_steps = time.time()

            global_step += 1

            if global_step % evaluate_every == 1:

                generator = torch.Generator(device=device).manual_seed(ddp_rank)

                val_diffusion_loss_binning = {k: 0 for k in range(10)}
                val_diffusion_loss_binning_count = {k: 0 for k in range(10)}

                dit_model.eval()

                total_losses = []
                diffusion_losses = []
                for batch_idx, batch in enumerate(test_loader):
                    with torch.no_grad():
                        # torch.compiler.cudagraph_mark_step_begin()

                        total_loss, diffusion_loss = forward(
                            dit_model,
                            batch,
                            text_encoder,
                            tokenizer,
                            device,
                            global_step,
                            master_process,
                            generator,
                            binnings=(
                                val_diffusion_loss_binning,
                                val_diffusion_loss_binning_count,
                            ),
                            return_index=return_index,
                        )

                        total_losses.append(total_loss.item())
                        diffusion_losses.append(diffusion_loss.item())

                        print(
                            f"Eval, Batch {batch_idx} done, {total_loss.item()}, {diffusion_loss.item()}"
                        )

                    if batch_idx == 8:
                        break

                dit_model.train()

                dist.barrier()
                total_loss = avg_scalar_across_ranks(np.mean(total_losses).item())
                diffusion_loss = avg_scalar_across_ranks(
                    np.mean(diffusion_losses).item()
                )

                state_dict = get_model_state_dict(dit_model)

                if master_process:
                    os.makedirs("checkpoints", exist_ok=True)
                    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                    os.makedirs(f"checkpoints/{run_name}/{global_step}", exist_ok=True)
                    # Save model state dict
                    print(
                        f"Saving model state dict to checkpoints/{run_name}/{global_step}"
                    )

                    stats = {
                        k: v / max(c, 1)
                        for k, v, c in zip(
                            val_diffusion_loss_binning.keys(),
                            val_diffusion_loss_binning.values(),
                            val_diffusion_loss_binning_count.values(),
                        )
                    }
                    wandb.log(
                        {
                            "test/total_loss": total_loss,
                            "test/diffusion_loss": diffusion_loss,
                            "test_binning/diffusion_loss_binning": stats,
                        }
                    )
                    print(f"Binned Losses: {stats}")

                dcp.save(
                    state_dict,
                    checkpoint_id=Path(f"checkpoints/{run_name}/{global_step}"),
                )
                print(f"Epoch {epoch} done")
                print(f"Global step {global_step}")

        # Cleanup
    if master_process:
        wandb.finish()
    cleanup()


if __name__ == "__main__":

    train_fsdp()
