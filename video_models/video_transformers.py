"""
Video transformer modules.

This module is intended for video self-attention, temporal attention,
and spatiotemporal transformer blocks.
"""

import torch
import torch.nn as nn


class VideoSelfAttention(nn.Module):
    """Spatiotemporal self-attention scaffold."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute video self-attention across space and time."""
        raise NotImplementedError("VideoSelfAttention forward pass is not implemented yet.")


class VideoTransformer(nn.Module):
    """Video transformer scaffold using tubelet embeddings."""

    def __init__(self, embed_dim: int, depth: int, num_heads: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([
            VideoSelfAttention(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass for video classification or feature extraction."""
        raise NotImplementedError("VideoTransformer forward pass is not implemented yet.")
