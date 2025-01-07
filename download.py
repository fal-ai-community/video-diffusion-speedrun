import click
from huggingface_hub import login, snapshot_download
from datasets import load_dataset
import os

from datasets import load_dataset


print("Downloading preprocessed openvid-1m dataset...")

ds = load_dataset("fal/cosmos-openvid-1m", num_proc=32, cache_dir="./cache")

print(
    "Note: Install hf-transfer and set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads"
)
print("Finished downloading preprocessed dataset")
