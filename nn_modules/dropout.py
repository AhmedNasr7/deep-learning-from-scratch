"""
Dropout and Stochastic Depth (DropPath).

References:
  - Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", 2014
  - Huang et al., "Deep Networks with Stochastic Depth", 2016
"""

import torch
import torch.nn as nn


class Dropout(nn.Module):
    """
    Standard Dropout.

    Training: randomly zeroes elements with probability p, then scales by 1/(1-p)
              (inverted dropout — keeps expected activation magnitude constant)
    Inference: identity (no zeroing, no scaling)

    Args:
        p: probability of zeroing an element (default: 0.5)
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Steps (training only):
            1. mask = Bernoulli(1 - p) → same shape as x
            2. x = x * mask / (1 - p)   (inverted scaling)
        """
        raise NotImplementedError


class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath.

    Drops entire residual branches (not individual elements) with probability p.
    Used in: DeiT, Swin Transformer, ConvNeXt.

    Training:  randomly skips entire samples' residual path
    Inference: scales output by (1-p) to match expected training output

    Args:
        drop_prob: probability of dropping a sample's path (default: 0.0)
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ...) residual tensor

        Steps (training only):
            1. Create per-sample Bernoulli mask: shape (B, 1, 1, ...)
            2. Zero out entire samples with probability drop_prob
            3. Scale by 1 / (1 - drop_prob)
        """
        raise NotImplementedError
