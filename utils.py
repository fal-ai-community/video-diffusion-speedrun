import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5TokenizerFast

from sharded_dataset import LatentDataset
from torch.utils.data import DistributedSampler

torch.set_float32_matmul_precision("high")


def avg_scalar_across_ranks(scalar):
    world_size = dist.get_world_size()
    scalar_tensor = torch.tensor(scalar, device="cuda")
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.AVG)
    return scalar_tensor.item()


def create_dataloader(split, batch_size, num_workers, do_shuffle, prefetch_factor=8):
    dset = LatentDataset(split=split)

    def collate_fn(batch):
        return {
            "latent": torch.stack([item["latent"] for item in batch]),
            "prompt": [item["prompt"] for item in batch],
        }

    dl = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        sampler=DistributedSampler(dset, shuffle=do_shuffle),
    )
    return dl


def encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=256,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
    return_index=-1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(
        text_input_ids.to(device), return_dict=True, output_hidden_states=True
    )

    prompt_embeds = prompt_embeds.hidden_states[return_index]
    if return_index != -1:
        prompt_embeds = text_encoder.encoder.final_layer_norm(prompt_embeds)
        prompt_embeds = text_encoder.encoder.dropout(prompt_embeds)

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def load_encoders(
    vae_path="black-forest-labs/FLUX.1-dev",
    text_encoder_path="black-forest-labs/FLUX.1-dev",
    device="cuda",
    compile_models=True,
):

    tokenizer = T5TokenizerFast.from_pretrained(
        text_encoder_path, subfolder="tokenizer_2"
    )

    text_encoder = (
        T5EncoderModel.from_pretrained(
            text_encoder_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    text_encoder.requires_grad_(False)

    if compile_models:
        text_encoder.forward = torch.compile(
            text_encoder.forward, mode="reduce-overhead"
        )

    return tokenizer, text_encoder
