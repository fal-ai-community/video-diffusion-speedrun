import torch
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    def __init__(self, size=1000, channels=16, frames=16, height=32, width=32, max_prompt_length=77):
        self.size = size
        self.channels = channels
        self.frames = frames 
        self.height = height
        self.width = width
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random latent tensor
        latent = torch.randn(self.channels, self.frames, self.height, self.width)
        
        prompt = "a cat"
        return {
            "latent": latent,
            "prompt": prompt
        }
