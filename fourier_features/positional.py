"""
Positional Fourier Features (NeRF-style, Mildenhall et al., 2020).

Deterministic frequency mapping using powers of 2:
  γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]

Used in NeRF and other neural implicit representations to encode
spatial coordinates at multiple frequency scales.
"""

import torch
import torch.nn as nn
import math


class PositionalFourierFeatures(nn.Module):
    """
    NeRF-style positional encoding with log-linear frequency bands.

    Args:
        in_dim:     input dimensionality
        num_freqs:  number of frequency bands (L in the paper)
        include_input: if True, prepend the raw input to the output

    Input:  (batch, in_dim)
    Output: (batch, in_dim * (2 * num_freqs) [+ in_dim if include_input])
    """

    def __init__(self, in_dim: int = 3, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim = in_dim * (2 * num_freqs) + (in_dim if include_input else 0)

        # Precompute frequency bands: [2⁰, 2¹, ..., 2^(L-1)]
        freq_bands = torch.pow(2.0, torch.arange(num_freqs).float())
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # For each frequency f in freq_bands:
        #   append sin(f * π * x) and cos(f * π * x)
        # Optionally prepend raw input x
        raise NotImplementedError
