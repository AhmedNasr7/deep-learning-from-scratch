"""
Hybrid ViT scaffold.

Hybrid ViT combines CNN feature extraction with transformer patch encoding.
"""

import torch
import torch.nn as nn


class HybridViT(nn.Module):
    """Hybrid CNN-ViT model scaffold."""

    def __init__(self, cnn_backbone: nn.Module, embed_dim: int, depth: int, num_heads: int, num_classes: int):
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.token_proj = nn.Linear(cnn_backbone.output_dim, embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method placeholder for hybrid ViT."""
        raise NotImplementedError("HybridViT forward pass is not implemented yet.")
