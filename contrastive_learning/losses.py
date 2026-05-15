"""
Contrastive learning losses.

This module holds from-scratch loss functions for SimCLR, BYOL, DINO, and MAE.
"""

import torch


def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Placeholder for a SimCLR-style normalized temperature-scaled loss."""
    raise NotImplementedError("Contrastive loss is not implemented yet.")


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Placeholder for MAE or masked reconstruction loss."""
    raise NotImplementedError("Reconstruction loss is not implemented yet.")
