"""
ALiBi — Attention with Linear Biases.

Instead of adding positional encodings to token embeddings, ALiBi adds a
static, non-learned bias directly to the attention scores.

The bias is proportional to the query-key distance:
    bias(i, j) = -m * |i - j|

where m is a head-specific slope (different per head, fixed, not learned).
This makes ALiBi naturally handle sequences longer than seen at training.

Used in: BLOOM, MPT.

Reference:
  - Press et al., "Train Short, Test Long: Attention with Linear Biases
    Enables Input Length Extrapolation", ICLR 2022
"""

import torch
import torch.nn as nn
import math


class ALiBi(nn.Module):
    """
    ALiBi — Attention with Linear Biases.

    Applied as an additive bias to raw attention logits (before softmax).
    No parameters — completely static. Not added to token embeddings.

    Args:
        n_heads:     number of attention heads
        max_seq_len: max sequence length for pre-computing bias table
    """

    def __init__(self, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        # Slopes: m_h = 2^{-8h/n_heads} for h=1..n_heads
        # Register as buffer — not a learned parameter

    def _compute_slopes(self) -> torch.Tensor:
        """
        Compute per-head slope values.

        Steps:
            1. start = 2^{-8/n_heads}
            2. slopes = [start^1, start^2, ..., start^n_heads]
        """
        raise NotImplementedError

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Build ALiBi bias matrix.

        Args:
            seq_len: current sequence length T

        Returns:
            bias: (n_heads, T, T) — to be added to attention logits
                  bias[h, i, j] = -slope_h * |i - j|
        """
        # your code here
        raise NotImplementedError
