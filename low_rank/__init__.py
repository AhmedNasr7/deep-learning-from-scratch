"""
Low-Rank Methods for efficient fine-tuning and model compression.

Modules:
  lora.py        — LoRA: Low-Rank Adaptation (Hu et al., 2021)
  dora.py        — DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)
  svd_compress.py — SVD-based post-training weight compression
  apply.py       — Utilities to inject LoRA into any model (GPT, ViT, Transformer)

Usage:
    from low_rank import LoRALinear, DoRALinear
    from low_rank.apply import apply_lora, freeze_base_model

    model = build_vit("vit_tiny", num_classes=10)
    # Target real attribute names: qkv_proj and out_proj inside FusedMultiHeadSelfAttention
    apply_lora(model, rank=4, target_modules=["qkv_proj", "out_proj"])
    freeze_base_model(model)
"""

from .lora import LoRALinear
from .dora import DoRALinear
from .apply import apply_lora, freeze_base_model

__all__ = ["LoRALinear", "DoRALinear", "apply_lora", "freeze_base_model"]
