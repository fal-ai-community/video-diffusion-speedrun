import os
import time
from contextlib import contextmanager
from functools import reduce
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5TokenizerFast

from sharded_dataset import LatentDataset, FakeDataset

torch.set_float32_matmul_precision("high")

@contextmanager
def timeit(name: str="", print_fn = print):
    """Context manager for timing code blocks.
    
    Args:
        name (str): Optional name for the timed block
        print_fn (callable): Function used for printing the timing output

    Example:
        with timeit("my op", lambda x:0) as t:
            # code to time
            ...
        my_op_time = t[0]
    """
    start = [time.time()]
    yield start
    start[0] = time.time() - start[0]
    print_fn(f"[{name}] {start[0]*1000:.2f}ms")

def async_allgather_stack(t: torch.Tensor, gather_pg: dist.ProcessGroup) -> torch.Tensor:
    out = torch.empty(gather_pg.size(), *t.shape, dtype=t.dtype, device=t.device).flatten(end_dim=1)
    handle = dist.all_gather_into_tensor(out, t, group=gather_pg, async_op=True)
    return lambda: (handle.wait(),out)[-1]

def limited_tqdm(it, total=9999999):
    for i, x in enumerate(tqdm(it, total=total)):
        yield x
        if i >= total: return
    print(f"WARNING: iterator {it} exhausted before {total=} steps.")

def get_module(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def set_module(module, access_string, value):
    names = access_string.split(sep=".")
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], value)


def avg_scalar_across_ranks(scalar):
    scalar_tensor = torch.tensor(scalar, device="cuda")
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.AVG)
    return scalar_tensor.item()


def create_dataloader(dataset, split, batch_size, num_workers, do_shuffle, device, prefetch_factor=8):
    cls = dict(real=LatentDataset, fake=FakeDataset)[dataset]
    dset = cls(split=split, shuffle=do_shuffle, num_workers=num_workers, device=device)

    def collate_fn(batch):
        return {
            "latent": torch.stack([item["latent"] for item in batch]),
            "prompt": [item["prompt"] for item in batch],
        }

    dl = DataLoader(
        dset,
        batch_size=batch_size,
        # prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    return dl


def encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
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
    hf_token=os.getenv("HF_HUB_TOKEN"),
):

    tokenizer = T5TokenizerFast.from_pretrained(
        text_encoder_path,
        subfolder="tokenizer_2",
        token=hf_token,
    )

    text_encoder = (
        T5EncoderModel.from_pretrained(
            text_encoder_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            token=hf_token,
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
