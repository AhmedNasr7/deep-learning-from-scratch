"""
Gaussian Fourier Features (Tancik et al., 2020 — "Fourier Features Let Networks Learn
High Frequency Functions in Low Dimensional Domains").

Maps low-dimensional inputs (e.g. 2D/3D coordinates) to a higher-dimensional
space using random Fourier projections, enabling MLPs to learn high-frequency
patterns that they'd otherwise struggle with.
"""

import torch
import torch.nn as nn
import math


class GaussianFourierFeatures(nn.Module):
    """
    Random Fourier feature mapping.

    γ(x) = [cos(2π·Bx), sin(2π·Bx)]

    B is a fixed random matrix sampled from N(0, σ²).
    Higher σ → higher frequencies captured.

    Args:
        in_dim:       input dimensionality (e.g. 2 for images, 3 for 3D)
        num_features: number of random frequencies (output dim = 2 * num_features)
        sigma:        std of the Gaussian for frequency sampling

    Input:  (batch, in_dim) — coordinates
    Output: (batch, 2 * num_features) — fourier-mapped features
    """

    def __init__(self, in_dim: int = 2, num_features: int = 256, sigma: float = 10.0):
        super().__init__()
        self.out_dim = 2 * num_features

        # Fixed random projection matrix — not learned
        B = torch.randn(in_dim, num_features) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Project: x_proj = 2π · x @ B
        # 2. Return [cos(x_proj), sin(x_proj)]
        raise NotImplementedError
