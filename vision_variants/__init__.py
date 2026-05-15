"""
Vision Variants — ViT extensions and alternatives.

Modules:
  swin.py      — Swin Transformer: hierarchical windowed attention (Liu et al., 2021)
  hybrid_vit.py — Hybrid ViT: CNN stem + ViT encoder
  deit.py      — DeiT: Data-efficient ViT with distillation token (Touvron et al., 2021)
"""

from .swin import SwinTransformer, SwinTransformerBlock
from .deit import DeiT, deit_distillation_loss

__all__ = [
    "SwinTransformer", "SwinTransformerBlock",
    "DeiT", "deit_distillation_loss",
]
