"""
Utilities to inject low-rank adapters into model architectures.

This module is intended to support GPT, Transformer, and ViT models by
replacing or augmenting selected linear layers with low-rank alternatives.
"""

from typing import List, Optional
import torch.nn as nn


def apply_lora(model: nn.Module, rank: int = 4, target_modules: Optional[List[str]] = None):
    """Scaffold for applying LoRA adapters to a model.

    Args:
        model: target model to modify
        rank: low-rank adaptation rank
        target_modules: list of module names to patch (e.g. ["W_q", "W_k", "W_v"])
    """
    raise NotImplementedError("apply_lora is not implemented yet.")


def freeze_base_model(model: nn.Module):
    """Scaffold to freeze base model parameters while keeping adapters trainable."""
    raise NotImplementedError("freeze_base_model is not implemented yet.")
