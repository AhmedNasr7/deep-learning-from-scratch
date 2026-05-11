"""
INR — Implicit Neural Representation (basic MLP for signal fitting).

The simplest neural field: an MLP that maps coordinates → signal.
Great starting point before SIREN/NeRF — fit an image pixel-by-pixel.

Usage:
    model = INR(in_features=2, out_features=3)  # (x, y) → RGB
    Train with:
      coords = grid of (x, y) in [-1, 1]
      targets = pixel values at those coords
      loss = MSE(model(coords), targets)
"""

import torch
import torch.nn as nn


class INR(nn.Module):
    """
    Basic MLP for implicit neural representation.

    Maps coordinates → signal values using ReLU activations.
    Simple but struggles with high-frequency details
    (which is why SIREN and Fourier features exist).

    Args:
        in_features:     input dim (2 for images, 3 for 3D)
        hidden_features: hidden layer width
        hidden_layers:   number of hidden layers
        out_features:    output dim (1 for grayscale, 3 for RGB)

    Input:  (batch, in_features) — coordinates
    Output: (batch, out_features) — predicted signal
    """

    def __init__(self, in_features: int = 2, hidden_features: int = 256,
                 hidden_layers: int = 4, out_features: int = 3):
        super().__init__()

        # your code here
        # Build MLP: Linear → ReLU → ... → Linear (no activation on output)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError
