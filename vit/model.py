"""
Vision Transformer (ViT) — educational, from-scratch version.

Reuses shared building blocks from the transformer module
(attention, normalization, feed_forward) to show how the same
components power both NLP and vision.

Separate from transformer/ because ViT is its own architecture,
not a seq2seq model.
"""

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from transformer.feed_forward import PositionWiseFeedForward
from transformer.normalization import LayerNorm

from .patch_embedding import PatchEmbedding


class ViTEncoderBlock(nn.Module):
    """
    ViT encoder block with pre-norm (LayerNorm before attention/FFN).

    Args:
        embed_dim: embedding dimension
        n_heads:   number of attention heads
        mlp_ratio: FFN hidden dim = embed_dim * mlp_ratio
        dropout:   dropout rate

    Input:  (batch, seq_len, embed_dim)
    Output: (batch, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = PositionWiseFeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # Pre-norm residual:
        # 1. x = x + attn(norm1(x), norm1(x), norm1(x))   (self-attention: Q=K=V)
        # 2. x = x + ffn(norm2(x))
        raise NotImplementedError


class VisionTransformer(nn.Module):
    """
    Full Vision Transformer (Dosovitskiy et al., 2020).

    patch_embed → prepend [CLS] token → add position embeddings →
    N × ViTEncoderBlock → LayerNorm → classification head

    Args:
        img_size, patch_size, in_channels: input image specs
        num_classes: number of output classes
        embed_dim:   embedding dimension
        depth:       number of encoder blocks
        n_heads:     number of attention heads
        mlp_ratio:   FFN expansion ratio
        dropout:     dropout rate

    Input:  (batch, in_channels, img_size, img_size)
    Output: (batch, num_classes) — classification logits
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for CLS
        self.pos_drop = nn.Dropout(dropout)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            ViTEncoderBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final norm + classification head
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Patch embedding: (B, C, H, W) → (B, N, D)
        # 2. Prepend [CLS] token: (B, N+1, D)
        # 3. Add position embeddings, apply dropout
        # 4. Pass through encoder blocks
        # 5. LayerNorm on [CLS] token output (index 0)
        # 6. Classification head → logits
        raise NotImplementedError


def build_vit(arch: str, num_classes: int, img_size: int = 32) -> VisionTransformer:
    """
    Factory for common ViT configurations.

    Supported:
        'vit_tiny':  embed=192,  depth=12, heads=3   → CIFAR-10 (32×32, patch=4)
        'vit_small': embed=384,  depth=12, heads=6   → Tiny ImageNet (64×64, patch=8)
        'vit_base':  embed=768,  depth=12, heads=12  → ImageNet (224×224, patch=16)
    """
    configs = {
        "vit_tiny":  dict(embed_dim=192,  depth=12, n_heads=3,  patch_size=4),
        "vit_small": dict(embed_dim=384,  depth=12, n_heads=6,  patch_size=8),
        "vit_base":  dict(embed_dim=768,  depth=12, n_heads=12, patch_size=16),
    }
    assert arch in configs, f"Unknown arch '{arch}'. Choose from: {list(configs.keys())}"

    return VisionTransformer(img_size=img_size, num_classes=num_classes, **configs[arch])
