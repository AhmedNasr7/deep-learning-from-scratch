"""
Contrastive and Self-Supervised Learning methods from scratch.

Modules:
  simclr.py — SimCLR: pairwise contrastive learning with NT-Xent loss
  mae.py    — MAE: Masked Autoencoder (self-supervised ViT)
  clip.py   — CLIP: image-text contrastive pretraining (Radford et al., 2021)
  losses.py — Shared loss functions (NT-Xent / InfoNCE)
  train.py  — Training loop scaffold
"""

from .simclr import SimCLR
from .mae import MaskedAutoencoder
from .clip import CLIP, CLIPImageEncoder, CLIPTextEncoder, clip_loss

__all__ = [
    "SimCLR",
    "MaskedAutoencoder",
    "CLIP", "CLIPImageEncoder", "CLIPTextEncoder", "clip_loss",
]
