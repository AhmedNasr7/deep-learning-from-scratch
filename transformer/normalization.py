"""
Layer Normalization (Ba et al., 2016).

Implement from scratch using nn.Parameter — no torch.nn.LayerNorm.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization: normalize across the last dimension (features).

    Learnable parameters: gamma (scale) and beta (shift).

    Args:
        d_model: feature dimension
        eps:     small constant for numerical stability

    Input:  (..., d_model)
    Output: (..., d_model), same shape, normalized
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Compute mean and variance across last dim
        # 2. Normalize: (x - mean) / sqrt(var + eps)
        # 3. Scale and shift: gamma * x_norm + beta

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.gamma * x_norm + self.beta



    
        
