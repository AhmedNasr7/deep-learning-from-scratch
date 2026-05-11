"""
UNet for diffusion models.

Encoder-decoder with skip connections. Takes noisy image + timestep
embedding, outputs predicted noise.
"""

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (same idea as positional encoding).

    Maps scalar timestep t → vector of dimension embed_dim.

    Args:
        embed_dim: embedding dimension

    Input:  (batch,) integer timesteps
    Output: (batch, embed_dim)
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Sinusoidal embedding of t (like positional encoding)
        # 2. Pass through MLP
        raise NotImplementedError


class UNet(nn.Module):
    """
    UNet for noise prediction in diffusion.

    Args:
        in_channels:  image channels (1 or 3)
        base_channels: base channel width (doubles at each downsample)
        time_dim:     timestep embedding dimension
        num_classes:  if set, enables class-conditional generation

    Input:  x (batch, C, H, W), t (batch,) timesteps
    Output: (batch, C, H, W) predicted noise
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 time_dim: int = 256, num_classes: int = None):
        super().__init__()

        self.time_emb = TimeEmbedding(time_dim)

        # your code here
        # Encoder: downsample blocks with residual connections
        # Bottleneck: middle block
        # Decoder: upsample blocks with skip connections from encoder
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError
