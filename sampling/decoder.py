from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from huggingface_hub import snapshot_download
import os
from datasets import load_dataset

import torch
import numpy as np
import imageio
import io


def deserialize_tensor(serialized_tensor: bytes, device=None) -> torch.Tensor:
    return torch.load(
        io.BytesIO(serialized_tensor),
        weights_only=True,
        map_location=torch.device(device) if device else None,
    )


def get_decoder(model_name: str = "Cosmos-Tokenizer-CV4x8x8"):
    """Get the decoder for the given model name.
    model_name can be "Cosmos-Tokenizer-DV4x8x8", "Cosmos-Tokenizer-DV8x8x8", or "Cosmos-Tokenizer-DV8x16x16".
    """

    local_dir = f"./pretrained_ckpts/{model_name}"
    if not os.path.exists(local_dir):
        hf_repo = "nvidia/" + model_name
        snapshot_download(repo_id=hf_repo, local_dir=local_dir)
    decoder = CausalVideoTokenizer(checkpoint_dec=f"{local_dir}/decoder.jit")
    return decoder


_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)


def unclamp_video(input_tensor: torch.Tensor) -> torch.Tensor:
    """Unclamps tensor in [-1,1] to video(dtype=np.uint8) in range [0..255]."""
    tensor = (input_tensor.float() + 1.0) / 2.0
    tensor = tensor.clamp(0, 1).cpu()
    return (tensor * _UINT8_MAX_F + 0.5).to(dtype=torch.uint8)


def save_tensor_to_mp4(tensor, decoder, path, name):
    decoded_video = decoder.decode(tensor.unsqueeze(0)).squeeze(0)

    # [C, T, H, W] -> [T, H, W, C]
    video = decoded_video.permute(1, 2, 3, 0)
    video = unclamp_video(video)

    os.makedirs(path, exist_ok=True)
    video_np = video.cpu().numpy()

    imageio.mimsave(os.path.join(path, f"{name}.mp4"), video_np, fps=30, codec="h264")


if __name__ == "__main__":
    decoder = get_decoder()
    # print(decoder)

    dataset = load_dataset("fal/cosmos-openvid-1m", num_proc=32, cache_dir="./cache")
    # test the decoder
    tensor = deserialize_tensor(dataset["train"][0]["serialized_latent"])

    # int4 quantize and dequantize back.
    MAX_VAL = 6.0
    tensor = (tensor.float().clamp(-MAX_VAL, MAX_VAL) + MAX_VAL) / (
        2 * MAX_VAL
    )  # this is [0, 1]
    tensor = (tensor * 15.0).long()

    # dequantize back
    tensor = tensor.float() / 15.0
    tensor = tensor * 2 * MAX_VAL - MAX_VAL
    tensor = tensor.bfloat16()

    print(tensor.shape)
    save_tensor_to_mp4(tensor, decoder, "./output", "test_raw")
