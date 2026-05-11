"""
Patch Embedding — educational, from-scratch version.

Uses manual reshape (not Conv2d) so you see exactly how an image
is broken into patches and projected.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and linearly project.

    Args:
        img_size:    input image size (square)
        patch_size:  patch size (must divide img_size)
        in_channels: input channels (3 for RGB)
        embed_dim:   output embedding dimension

    Input:  (batch, in_channels, img_size, img_size)
    Output: (batch, num_patches, embed_dim)

    num_patches = (img_size / patch_size) ^ 2
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Reshape (B, C, H, W) into patches: (B, num_patches, patch_dim)
        #    - Rearrange spatial dims into grid of patches
        #    - Flatten each patch into a vector of size patch_dim
        # 2. Linear projection: (B, num_patches, embed_dim)
        raise NotImplementedError
