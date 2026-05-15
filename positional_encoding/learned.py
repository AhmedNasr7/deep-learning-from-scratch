"""
Learned Positional Embeddings.

Simplest alternative to sinusoidal: map each position index to a learned vector.
Used in: GPT-2, BERT (with absolute positions), DeiT.

Limitation: cannot generalize beyond max_seq_len seen at training
(unlike RoPE, ALiBi, or sinusoidal which extrapolate).

Reference:
  - Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2019
"""

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned absolute positional embeddings.

    Simply wraps nn.Embedding — every position index maps to a
    trainable d_model-dimensional vector. Added to token embeddings.

    Args:
        max_seq_len: maximum sequence length
        d_model:     embedding dimension
        dropout:     dropout rate applied after adding positional embeddings
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings to input tensor.

        Args:
            x: (B, T, D) token embeddings

        Returns:
            x + pos_emb: (B, T, D)

        Steps:
            1. positions = [0, 1, ..., T-1]
            2. pos_emb = self.emb(positions)  → (T, D) → broadcast to (B, T, D)
            3. return dropout(x + pos_emb)
        """
        raise NotImplementedError


class RelativePositionalBias(nn.Module):
    """
    Relative positional bias (T5-style).

    Instead of absolute positions, adds a learned scalar bias to each
    attention head based on the RELATIVE distance between query and key.

    Used in: T5, Longformer.

    Args:
        n_heads:     number of attention heads
        n_buckets:   number of distance buckets (positions mapped to buckets)
        max_distance: max relative distance before clamping to last bucket
    """

    def __init__(self, n_heads: int, n_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(n_buckets, n_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Map relative distances to bucket indices (log-scale bucketing)."""
        raise NotImplementedError

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position bias matrix.

        Args:
            seq_len: sequence length T

        Returns:
            bias: (1, n_heads, T, T) — to be added to attention logits
        """
        raise NotImplementedError
