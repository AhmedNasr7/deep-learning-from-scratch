"""
NeRF — Neural Radiance Fields (Mildenhall et al., 2020).

Maps 5D input (3D position + 2D viewing direction) to color + density.
Uses positional Fourier encoding to capture high-frequency detail.
"""

import torch
import torch.nn as nn

from fourier_features import PositionalFourierFeatures


class NeRF(nn.Module):
    """
    Neural Radiance Field MLP.

    Architecture (from paper):
        - 8 FC layers (256 units, ReLU) for density
        - Skip connection at layer 4 (concatenate input)
        - 1 FC layer (128 units) for color, conditioned on view direction
        - Output: (density σ, color RGB)

    Args:
        pos_dim:        raw position dim (3)
        dir_dim:        raw direction dim (3)
        pos_freq:       number of frequency bands for position encoding
        dir_freq:       number of frequency bands for direction encoding
        hidden_dim:     hidden layer width
        num_layers:     number of density layers (skip at num_layers // 2)

    Input:
        positions:  (batch, 3) — 3D coordinates
        directions: (batch, 3) — viewing directions

    Output:
        rgb:     (batch, 3) — predicted color
        density: (batch, 1) — volume density (σ)
    """

    def __init__(self, pos_dim: int = 3, dir_dim: int = 3,
                 pos_freq: int = 10, dir_freq: int = 4,
                 hidden_dim: int = 256, num_layers: int = 8):
        super().__init__()

        self.pos_enc = PositionalFourierFeatures(pos_dim, pos_freq)
        self.dir_enc = PositionalFourierFeatures(dir_dim, dir_freq)
        self.skip_layer = num_layers // 2

        # your code here
        # 1. Density MLP: pos_enc_dim → hidden_dim × num_layers (with skip connection)
        # 2. Density output: hidden_dim → 1 (σ, with ReLU for non-negative)
        # 3. Feature layer: hidden_dim → hidden_dim
        # 4. Color MLP: (hidden_dim + dir_enc_dim) → 128 → 3 (RGB, with sigmoid)
        raise NotImplementedError

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> tuple:
        # your code here
        # 1. Encode position and direction with Fourier features
        # 2. Pass through density MLP (with skip connection)
        # 3. Output density
        # 4. Concatenate feature + direction encoding
        # 5. Pass through color MLP
        # Returns: (rgb, density)
        raise NotImplementedError
