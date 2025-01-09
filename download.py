from datasets import load_dataset

print("Downloading preprocessed openvid-1m dataset...")

ds = load_dataset("fal/cosmos-openvid-1m", num_proc=32, cache_dir="./cache")

print(
    "Note: Install hf-transfer and set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads"
)
print("Finished downloading preprocessed dataset")

rows = 1979810

import io

import torch


def deserialize_tensor(serialized_tensor: bytes, device=None) -> torch.Tensor:
    return torch.load(
        io.BytesIO(serialized_tensor),
        weights_only=True,
        map_location=torch.device(device) if device else None,
    )


for i in range(10):
    print(deserialize_tensor(ds["train"][rows // 2 - 9 + i]["serialized_latent"]).shape)
    print(ds["train"][rows // 2 - 9 + i]["caption"])
print("---")

for i in range(10):
    print(deserialize_tensor(ds["train"][rows - 9 + i]["serialized_latent"]).shape)
