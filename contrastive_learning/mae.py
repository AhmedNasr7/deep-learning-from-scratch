"""
Masked Autoencoder scaffold.

This module is intended for implementing masked transformer-style pretraining
for images or patches.
"""

import torch
import torch.nn as nn


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder model scaffold."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass placeholder for masked reconstruction."""
        raise NotImplementedError("MAE forward pass is not implemented yet.")
