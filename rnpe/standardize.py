
import torch
from torch import nn

class Standardize(nn.Module):
    """Standardizing "embedding network" with no trainable parameters."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor):
        return (tensor - self._mean) / self._std
