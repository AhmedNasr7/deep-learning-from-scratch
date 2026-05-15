"""
Embedding Layer — maps discrete token indices to dense vectors.

Used everywhere: GPT token embeddings, ViT patch projection (conceptually),
positional embeddings, word2vec.
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Lookup table that maps integer indices to dense embedding vectors.

    Equivalent to: a Linear layer applied to one-hot vectors, but implemented
    as an efficient index-select for memory efficiency.

    Args:
        num_embeddings: vocabulary / dictionary size
        embedding_dim:  dimensionality of each embedding vector
        padding_idx:    if given, this index outputs the zero vector (no grad)
        max_norm:       if given, embeddings with norm > max_norm are renormalized
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int = None, max_norm: float = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, T) or (B,) integer token indices

        Returns:
            embeddings: (B, T, D) or (B, D)

        Steps:
            1. self.weight[indices]  — simple index select
            2. If padding_idx: zero out those rows
            3. If max_norm: clamp embedding norms
        """
        raise NotImplementedError
