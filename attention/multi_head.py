"""
Multi-Head Attention.

Projects Q, K, V into multiple heads, runs ScaledDotProductAttention
per head in parallel, concatenates and projects back.

This file is designed to grow — add variants here:
  - MultiQueryAttention (MQA)
  - GroupedQueryAttention (GQA)
  - etc.
"""

import torch
import torch.nn as nn

from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention.

    Args:
        d_model: model dimension
        n_heads: number of attention heads (must divide d_model)
        dropout: dropout rate on attention weights

    Input:
        Q: (batch, seq_q, d_model)
        K: (batch, seq_k, d_model)
        V: (batch, seq_k, d_model)
        mask: optional (batch, 1, seq_q, seq_k) or broadcastable

    Output: (batch, seq_q, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Projection layers: Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Project Q, K, V
        # 2. Reshape to (batch, n_heads, seq, d_k)
        # 3. Apply ScaledDotProductAttention per head
        # 4. Concatenate heads, project with W_o

        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        B, seq_q, d_model = Q.shape
        _, seq_k, _ = K.shape
        Q = Q.reshape(B, seq_q, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.reshape(B, seq_k, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.reshape(B, seq_k, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        output = self.attention(Q, K, V, mask)

        output = output.permute(0, 2, 1, 3).reshape(B, seq_q, self.d_model) 
        output = self.W_o(output)
        output = self.dropout(output)

        return output




