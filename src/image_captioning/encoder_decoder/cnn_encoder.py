import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    CNN Encoder for extracting the latent features in an image.

    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == '__main__':
    new_tensor = torch.arange(5)
    new_tensor = (new_tensor, torch.ones(5))
    new_tensor = torch.stack(new_tensor, 0)
    print(f"Tensor shape: {new_tensor.shape}")

