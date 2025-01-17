import io
from string import ascii_letters
from random import randint, choice

import torch
from torch.utils.data import IterableDataset

from lavender_data import StreamingLavenderDataset


def deserialize_tensor(serialized_tensor: bytes, device=None) -> torch.Tensor:
    return torch.load(
        io.BytesIO(serialized_tensor),
        weights_only=True,
        map_location=torch.device(device) if device else None,
    )


class LatentDataset(IterableDataset):
    def __init__(
        self,
        split="train",
        cache_dir="./cache",
        shuffle=False,
        device=None,
        num_workers=1,
    ):
        self.device = device
        self.dataset = StreamingLavenderDataset(
            local=cache_dir,
            remote="hf://huggingface.co/fal/cosmos-openvid-1m/continuous",
            split=split,
            queue_size=16,
            cache_size=256,
            num_workers=num_workers,
            shuffle_shards=shuffle,
            shuffle_samples=shuffle,
        )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for item in self.dataset:
            latent = deserialize_tensor(item["serialized_latent"], self.device)
            yield {"latent": latent, "prompt": item["caption"]}

class FakeDataset(IterableDataset):
    def __init__(self, *a, **k): pass
    def __len__(self): return 9999999
    def __iter__(self):
        for _ in range(len(self)):
            yield dict(
                latent=torch.randn(16,16,64,64,device='cuda'),
                prompt=''.join(choice(ascii_letters) for _ in range(randint(16,64))),
            )

if __name__ == "__main__":
    dset = LatentDataset(split="train", device="cuda")
    print(f"Length: {len(dset)}")
    # iterate and check the length of the latent tensor
    for data in dset:
        print(data["latent"].shape)
        # print stats.
        print(
            data["latent"].min(),
            data["latent"].max(),
            data["latent"].mean(),
            data["latent"].std(),
        )
