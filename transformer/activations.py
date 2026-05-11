"""
Activation functions.

Implement from scratch — no torch.nn.functional.relu / gelu.
"""

import torch
import torch.nn as nn


class ReLU(nn.Module):
    """
    Rectified Linear Unit.

    Input:  x of any shape
    Output: same shape, with negative values zeroed out
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (used in GPT-2 / ViT).

    Input:  x of any shape
    Output: same shape, smooth approximation of ReLU

    Hint: exact form uses torch.erf, approximate form uses tanh.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError


# Functional aliases for convenience
def relu(x: torch.Tensor) -> torch.Tensor:
    """Functional ReLU — see ReLU class."""
    # your code here
    raise NotImplementedError


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Functional GELU — see GELU class."""
    # your code here
    raise NotImplementedError
