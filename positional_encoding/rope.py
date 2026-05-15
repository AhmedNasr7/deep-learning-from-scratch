"""
RoPE — Rotary Position Embedding.

Instead of adding positional information to token embeddings, RoPE *rotates*
Q and K vectors by a position-dependent angle before computing attention.
This gives the attention score a natural relative-position dependence:
    <R_m q, R_n k> depends only on (m - n), not absolute positions.

Used in: LLaMA, GPT-NeoX, PaLM, Mistral, Gemma.

Reference:
  - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
"""

import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """
    RoPE — Rotary Position Embedding.

    Applied directly to Q and K inside attention, NOT added to token embeddings.
    Operates pair-wise on feature dimensions: (x_1, x_2, ..., x_{d-1}, x_d) →
    rotates each consecutive pair (x_{2i-1}, x_{2i}) by angle m * theta_i.

    Frequency per dimension:
        theta_i = 10000^{-2i/d}  for i = 0, 1, ..., d//2 - 1

    Rotation of pair (x, y) at position m:
        [x * cos(m*theta) - y * sin(m*theta),
         x * sin(m*theta) + y * cos(m*theta)]

    Args:
        d_model:  feature dimension (must be even)
        max_seq_len: maximum sequence length to pre-compute frequencies for
        base:     frequency base (default 10000, as in original paper)
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        # Pre-compute cos/sin tables (register as buffers — not learned parameters)
        # Shape: (max_seq_len, d_model // 2)

    def _build_cache(self) -> tuple:
        """
        Build cos and sin look-up tables.

        Steps:
            1. theta_i = base^{-2i/d} for i in range(d//2)
            2. m = position indices [0, 1, ..., max_seq_len-1]
            3. freqs = outer(m, theta)  → (max_seq_len, d//2)
            4. cos_table = cos(freqs), sin_table = sin(freqs)
        """
        raise NotImplementedError

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate pairs: [x1, x2, ..., x_{d-1}, x_d] → [-x2, x1, ..., -x_d, x_{d-1}]

        Args:
            x: (..., d)
        Returns:
            x_rot: (..., d)
        """
        raise NotImplementedError

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Apply RoPE to Q and K tensors.

        Args:
            q: (B, n_heads, T, d_k)
            k: (B, n_heads, T, d_k)
        Returns:
            (q_rotated, k_rotated) — same shapes as input
        """
        # your code here
        # 1. Slice cos/sin tables to [:T]
        # 2. q_rot = q * cos + rotate_half(q) * sin
        # 3. k_rot = k * cos + rotate_half(k) * sin
        raise NotImplementedError
