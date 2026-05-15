"""
Tubelet embedding for video transformers.

This scaffold converts `B, C, T, H, W` video tensors into token sequences.
"""

import torch
import torch.nn as nn


class TubeletEmbedding(nn.Module):
    """Extracts tubelets from video input and projects them into embeddings."""

    def __init__(self, in_channels: int, embed_dim: int, tubelet_size: tuple[int, int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.proj = nn.Linear(in_channels * tubelet_size[0] * tubelet_size[1] * tubelet_size[2], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert video input into a sequence of tubelet embeddings."""
        raise NotImplementedError("Tubelet embedding forward pass is not implemented yet.")
