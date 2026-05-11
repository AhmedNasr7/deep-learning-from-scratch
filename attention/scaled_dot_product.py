"""
Scaled Dot-Product Attention (Vaswani et al., 2017).

The fundamental attention operation. All other attention variants
build on top of or modify this.
"""

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.

    Input:
        Q: (batch, ..., seq_q, d_k)
        K: (batch, ..., seq_k, d_k)
        V: (batch, ..., seq_k, d_v)
        mask: optional, broadcastable to (batch, ..., seq_q, seq_k)
              positions with True / 1 are masked (set to -inf before softmax)

    Output: (batch, ..., seq_q, d_v)
    """

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # your code here
        raise NotImplementedError
