import gc
import logging
import math
import os
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.optim as optim
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import get_model_state_dict

# Enable TF32 for faster training
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
# torch._dynamo.config.capture_scalar_outputs = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
from model import DiT, apply_fsdp
from utils import (
    avg_scalar_across_ranks,
    create_dataloader,
    encode_prompt_with_t5,
    load_encoders,
)

CAPTURE_INPUT = False


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
    binnings=None,
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

    output = dit_model(z_t, caption_encoded, t)

    diffusion_loss_batchwise = (
        (v_objective.float() - output.float()).pow(2).mean(dim=(1, 2, 3, 4))
    )

    diffusion_loss = diffusion_loss_batchwise.mean()

    # # timestep binning
    # tbins = [int(_t * 10) for _t in t]

    # if binnings is not None:
    #     (
    #         diffusion_loss_binning,
    #         diffusion_loss_binning_count,
    #     ) = binnings
    #     for idx, tb in enumerate(tbins):
    #         diffusion_loss_binning[tb] += diffusion_loss_batchwise[idx].item()
    #         diffusion_loss_binning_count[tb] += 1

    total_loss = diffusion_loss

    forward_time = time.time() - forward_start
    if master_process:
        logger.info(f"Forward pass took {forward_time*1000:.2f}ms")

    return total_loss, diffusion_loss


@click.command()
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
@click.option("--compile_models", type=bool, default=False, help="Compile models")
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
    num_epochs,
    batch_size,
    learning_rate,
    max_steps,
    evaluate_every,
    run_name,
    model_width,
    model_depth,
    model_head_dim,
    compile_models,
    optimizer_type,
    lr_scheduler_type,
    train_bias_and_rms,
    init_std_factor,
    project_name,
    return_index,
    load_checkpoint,
):
    # Initialize distributed training
    assert torch.cuda.is_available(), "CUDA is required for training"
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # Set random seeds
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)

    # Initialize wandb for the master process

    tokenizer, text_encoder = load_encoders(device=device, compile_models=False)

    # with torch.device("meta" if load_checkpoint is not None else device):
    dit_model = DiT(
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

    # initialize 2d params with normal fan_in
    for name, param in dit_model.named_parameters():
        # check if param is 2d
        if len(param.shape) == 2:
            fan_in = param.shape[1]
            param.data.mul_(init_std_factor)

    param_count = sum(p.numel() for p in dit_model.parameters())

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
        print(f"param_count: {param_count / 1e6}M")

    if master_process:

        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "model_parameters": param_count / 1e6,
                "model_width": model_width,
                "model_depth": model_depth,
                "model_head_dim": model_head_dim,
                "train_bias_and_rms": train_bias_and_rms,
            },
        )

    # Wrap model in FSDP
    constant_param_name = ["patch_proj", "context_kv", "positional_embedding"]

    # Move model to CUDA device

    torch.cuda.set_device(device)
    load_via_pt = False
    if load_checkpoint is not None:
        checkpoint_path = Path(load_checkpoint)
        load_via_pt = True

        if load_via_pt:
            if master_process:
                if not os.path.exists(checkpoint_path / "temp.pt"):
                    dcp_to_torch_save(checkpoint_path, checkpoint_path / "temp.pt")

            dist.barrier()
            state_dict = torch.load(checkpoint_path / "temp.pt", map_location="cpu")

            state_dict = {
                k.replace("module.", "")
                .replace("_orig_mod.", ""): v.clone()
                .to(device, dtype=torch.float32)
                for k, v in state_dict.items()
            }

            status = dit_model.load_state_dict(state_dict, assign=True, strict=False)

            print(f"Rank {ddp_rank} done loading checkpoint, {status}")

            del state_dict
            dit_model = dit_model.to(device)

            dist.barrier()
            print(f"Rank {ddp_rank} done loading checkpoint, {status}")

    # print(f"Loaded checkpoint from {load_checkpoint}")
    dit_model = apply_fsdp(
        dit_model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    if compile_models:
        torch._dynamo.config.cache_size_limit = 8
        dit_model = torch.compile(dit_model)

    dist.barrier()
    # state_dict = {}

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
        "train",
        batch_size,
        num_workers=8,
        do_shuffle=True,
        prefetch_factor=4,
    )

    test_loader = create_dataloader("test", batch_size, num_workers=1, do_shuffle=False)

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

    for epoch in range(num_epochs):
        if global_step >= max_steps:
            break

        for batch_idx, batch in enumerate(train_loader):

            if global_step >= max_steps:
                break

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
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            backward_time = time.time() - backward_start

            if master_process:
                logger.info(f"Backward pass took {backward_time*1000:.2f}ms")

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
