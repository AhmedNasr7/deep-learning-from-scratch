"""
Activation Functions.

References:
  - Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines", 2010
  - Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", 2016
  - Ramachandran et al., "Searching for Activation Functions (Swish/SiLU)", 2017
  - Misra, "Mish: A Self Regularized Non-Monotonic Activation Function", 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReLU(nn.Module):
    """
    Rectified Linear Unit: f(x) = max(0, x)

    Fast, sparse (many zeros), but suffers from dying ReLU problem
    (neurons can get permanently stuck at 0).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LeakyReLU(nn.Module):
    """
    Leaky ReLU: f(x) = x if x > 0 else negative_slope * x

    Fixes dying ReLU by allowing small gradient for x < 0.

    Args:
        negative_slope: slope for x < 0 (default: 0.01)
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.

    f(x) = x * Φ(x)  where Φ is the Gaussian CDF

    Approximation (Hendrycks & Gimpel):
        f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

    Used in: BERT, GPT-2, ViT, and most modern transformers.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (Swish): f(x) = x * sigmoid(x)

    Smooth, non-monotonic, bounded below. Self-gating.
    Used in: EfficientNet, LLaMA FFN (SwiGLU variant).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Mish(nn.Module):
    """
    Mish: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    Smooth, non-monotonic, no upper bound, small negative values allowed.
    Often outperforms ReLU and Swish on image tasks.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
