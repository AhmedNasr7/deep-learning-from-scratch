"""
Regression Losses: MSE, MAE, Huber, etc.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Mean Squared Error (L2 Loss).

    Loss = (1/N) * Σ (y_pred - y_true)^2

    Args:
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MAELoss(nn.Module):
    """
    Mean Absolute Error (L1 Loss).

    Loss = (1/N) * Σ |y_pred - y_true|

    More robust to outliers than MSE.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1) — L2 for small errors, L1 for large errors.

    Loss = 0.5 * (y-ŷ)^2              if |y-ŷ| < delta
           delta * (|y-ŷ| - delta/2)  otherwise

    Args:
        delta: threshold between L2 and L1 regime (default: 1.0)
    """
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
