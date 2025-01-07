import io

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def deserialize_tensor(serialized_tensor: bytes, device=None) -> torch.Tensor:
    return torch.load(
        io.BytesIO(serialized_tensor),
        weights_only=True,
        map_location=torch.device(device) if device else None,
    )


class LatentDataset(Dataset):
    def __init__(self, split="train", cache_dir="./cache"):
        MS = 1979810 // 2
        RANGE = range(0, MS - 1000) if split == "train" else range(MS - 1000, MS)

        self.dataset = load_dataset(
            "fal/cosmos-openvid-1m", split="train", cache_dir=cache_dir
        ).select(RANGE)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        latent = deserialize_tensor(item["serialized_latent"], "cpu")

        return {"latent": latent, "prompt": item["caption"]}


if __name__ == "__main__":
    dset = LatentDataset(split="train", device="cuda")
    print(f"Length: {len(dset)}")
    print(dset[0])
    # iterate and check the length of the latent tensor
    for i in range(len(dset)):
        print(dset[i]["latent"].shape)
        # print stats.
        print(
            dset[i]["latent"].min(),
            dset[i]["latent"].max(),
            dset[i]["latent"].mean(),
            dset[i]["latent"].std(),
        )
