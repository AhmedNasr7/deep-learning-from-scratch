"""
Patch Embedding — educational, from-scratch version.

Uses manual reshape (not Conv2d) so you see exactly how an image
is broken into patches and projected.
"""

from _pytest import fixtures
from _pytest import fixtures
from _pytest import fixtures
from sympy.printing.pretty.pretty_symbology import B
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

    def __init__(self, img_size: int = 64, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
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
        # (B, C, H, W)
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 1, 3, 5) # (B, H // P, W // P, C, P , P)
        x = x.reshape(B, H // P, W // P, self.patch_dim) # collect patch_dim = C * patch_size * patch_size
        x = self.proj(x) # project x to (B, H // P, W // P, embed_dim)
        
        # Flatten the grid dimensions to create a sequence of patches
        # (B, H // P, W // P, embed_dim) -> (B, num_patches, embed_dim)
        x = x.flatten(1, 2)
        return x


class PatchEmbeddingConv(nn.Module):
    """
    Standard production implementation using a 2D Convolution.
    This is mathematically identical to the manual reshape + linear layer,
    but highly optimized in PyTorch using CUDNN.
    """
    def __init__(self, img_size: int = 64, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        # A convolution with kernel_size == stride == patch_size 
        # exactly extracts non-overlapping patches!
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H // patch_size, W // patch_size)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Swap so sequence is in the middle: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class PatchEmbeddingEinops(nn.Module):
    """
    Elegant implementation using the 'einops' library.
    Highly readable and very popular in modern research code.
    """
    def __init__(self, img_size: int = 64, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import inside to avoid crashing if einops isn't installed
        from einops import rearrange
        
        # This one string replaces all the reshape and permute magic!
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                      p1=self.patch_size, p2=self.patch_size)
        
        return self.proj(x)
