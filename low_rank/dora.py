"""
DoRA: Weight-Decomposed Low-Rank Adaptaptation placeholder.

This file holds the DoRA layer scaffold for later implementation.

Reference:
  - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", 2024
"""

import torch
import torch.nn as nn


class DoRALinear(nn.Module):
    """DoRA low-rank adapter layer placeholder.

    This class is a scaffold only; the exact decomposition and forward pass
    are deferred until implementation time.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.bias = bias

        self.base_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.dora_left = nn.Parameter(torch.zeros(rank, in_features))
        self.dora_right = nn.Parameter(torch.zeros(out_features, rank))

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder forward pass for DoRA.

        Expected behavior after implementation:
            DoRA decomposes W = m * (V / ||V||) where:
              - m: learnable per-column magnitude vector (replaces norm)
              - V = W_0 + dora_right @ dora_left  (LoRA-style directional update)
              - The output normalizes the direction then scales by magnitude:
                out = x @ (m * normalize(V)).T + bias
        """
        raise NotImplementedError("DoRA forward pass is not implemented yet.")
