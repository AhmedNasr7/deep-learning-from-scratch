"""
GPT-2 building blocks.

Key differences from the encoder-only Transformer:
  - Causal (masked) self-attention — can only attend to past tokens
  - Pre-norm — LayerNorm before attention/FFN, not after
  - GELU activation instead of ReLU
  - Learned positional embeddings instead of sinusoidal
"""

import torch
import torch.nn as nn

from transformer.normalization import LayerNorm
from transformer.activations import GELU


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) multi-head self-attention for autoregressive models.

    Generates a causal mask internally — tokens can only attend to
    previous positions (and themselves).

    Args:
        d_model:     model dimension
        n_heads:     number of attention heads
        max_seq_len: maximum sequence length (for mask buffer)
        dropout:     dropout rate

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Combined QKV projection (more efficient than 3 separate)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Pre-compute causal mask as buffer (not a parameter)
        # Upper triangular = True means "block this position"
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Project to Q, K, V via self.qkv
        # 2. Reshape to (B, n_heads, T, d_k)
        # 3. Scaled dot-product attention with causal mask
        #    - scores = Q @ K^T / sqrt(d_k)
        #    - apply self.causal_mask[:T, :T] (mask future positions)
        #    - softmax, dropout
        # 4. Attend: attn @ V
        # 5. Concatenate heads, project with self.proj
        raise NotImplementedError


class GPTFeedForward(nn.Module):
    """
    GPT-2 feed-forward network with GELU activation.

    Args:
        d_model: model dimension
        d_ff:    hidden dimension (typically 4 * d_model)
        dropout: dropout rate

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # fc1 → GELU → fc2 → dropout
        raise NotImplementedError
