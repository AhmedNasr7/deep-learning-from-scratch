"""
Activation functions.

Implement from scratch — no torch.nn.functional.relu / gelu.
"""

import torch
import torch.nn as nn
import math

class ReLU(nn.Module):
    """
    Rectified Linear Unit.

    Input:  x of any shape
    Output: same shape, with negative values zeroed out
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, 0)


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (used in GPT-2 / ViT).

    Input:  x of any shape
    Output: same shape, smooth approximation of ReLU

    Hint: exact form uses torch.erf, approximate form uses tanh.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))



# Functional aliases for convenience
def relu(x: torch.Tensor) -> torch.Tensor:
    """Functional ReLU — see ReLU class."""
    return torch.where(x > 0, x, 0)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Functional GELU — see GELU class."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))
