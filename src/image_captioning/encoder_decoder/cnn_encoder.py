import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass
