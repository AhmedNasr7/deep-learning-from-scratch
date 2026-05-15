"""
Linear Layer — fully connected (dense) layer.

y = x @ W.T + b

Reference:
  - Rosenblatt, "The Perceptron", 1958
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Fully connected linear layer from scratch.

    Args:
        in_features:  input dimension D_in
        out_features: output dimension D_out
        bias:         whether to add a learnable bias (default: True)

    Input:  (..., D_in)
    Output: (..., D_out)

    Weight init: Kaiming uniform (default in PyTorch)
    Bias init:   uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Steps:
            1. out = x @ self.weight.T
            2. If bias: out = out + self.bias
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"
