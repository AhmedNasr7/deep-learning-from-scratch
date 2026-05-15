"""
Normalization Layers — BatchNorm1d and BatchNorm2d.

Normalizes activations per mini-batch to stabilize training.
Note: LayerNorm is in transformer/normalization.py (preferred for transformers).

Reference:
  - Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training", 2015
"""

import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    """
    Batch Normalization for 2D inputs (B, C) or 3D (B, C, L).

    Normalizes each feature channel across the batch:
        x_hat = (x - mean_B) / sqrt(var_B + eps)
        out   = gamma * x_hat + beta

    At inference: uses running statistics (exponential moving average).

    Args:
        num_features: number of channels C
        eps:          numerical stability constant
        momentum:     running stats update rate (default: 0.1)
        affine:       if True, learn gamma (scale) and beta (shift)
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = self.beta = None

        # Running statistics (not learned — updated via EMA)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Steps:
            Training:
                1. mean = x.mean(dim=0) over batch (and L if 3D)
                2. var  = x.var(dim=0, unbiased=False)
                3. Update running_mean, running_var (EMA)
                4. x_hat = (x - mean) / sqrt(var + eps)
            Inference:
                1. x_hat = (x - running_mean) / sqrt(running_var + eps)
            Both:
                5. return gamma * x_hat + beta
        """
        raise NotImplementedError


class BatchNorm2d(nn.Module):
    """
    Batch Normalization for 4D inputs (B, C, H, W).

    Same as BatchNorm1d but normalizes over (B, H, W) for each channel C.

    Args:
        num_features: number of channels C
        eps:          numerical stability constant
        momentum:     running stats update rate
        affine:       if True, learn gamma and beta (shape: C)
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(num_features, 1, 1))
        else:
            self.gamma = self.beta = None

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Steps (same as BatchNorm1d but reduce over dims 0, 2, 3):
            1. mean = x.mean(dim=[0, 2, 3])  → (C,)
            2. var  = x.var(dim=[0, 2, 3], unbiased=False)
            3. Normalize and affine-transform
        """
        raise NotImplementedError
