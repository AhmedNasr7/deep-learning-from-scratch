"""
DeiT — Data-efficient Image Transformers.

A ViT trained on ImageNet-1K *without* JFT-300M pre-training, made possible by:
  1. Knowledge distillation from a CNN teacher (RegNet/EfficientNet)
  2. A distillation token (alongside the CLS token) that learns from teacher soft labels
  3. Strong data augmentation (Mixup, CutMix, RandAugment)

Key difference from vanilla ViT:
  - Adds a learnable distillation token alongside [CLS]
  - Hard distillation: distillation token supervised by argmax of teacher output
  - Soft distillation: distillation token supervised by teacher soft probabilities (KL divergence)

Reference:
  - Touvron et al., "Training data-efficient image transformers &
    distillation through attention", ICML 2021
"""

import torch
import torch.nn as nn

from vit.model import ViTEncoderBlock, build_vit


class DeiT(nn.Module):
    """
    DeiT model: ViT + distillation token + teacher distillation loss.

    Architecture:
        patch_embed → [CLS] + [DIST] + patches → pos_embed →
        N × ViTEncoderBlock → LayerNorm →
        cls_head (classification) + dist_head (distillation)

    At inference: average cls_head and dist_head outputs.

    Args:
        img_size:      input image resolution
        patch_size:    patch size
        in_channels:   number of input channels
        num_classes:   number of output classes
        embed_dim:     embedding dimension
        depth:         number of encoder blocks
        n_heads:       number of attention heads
        mlp_ratio:     FFN expansion factor
        dropout:       dropout rate
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 192, depth: int = 12, n_heads: int = 3,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        # TODO: patch embedding (reuse from vit.patch_embedding)
        # TODO: CLS token (same as ViT)
        # TODO: distillation token (same shape as CLS token)
        # TODO: position embedding (num_patches + 2 for CLS + DIST)
        # TODO: encoder blocks
        # TODO: final LayerNorm
        # TODO: cls_head — classification head
        # TODO: dist_head — distillation head (same size as cls_head)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (cls_logits, dist_logits) during training
            averaged logits during inference
        """
        # your code here
        # 1. Patch embed
        # 2. Prepend [CLS] and [DIST] tokens → (B, N+2, D)
        # 3. Add position embeddings
        # 4. Pass through encoder blocks
        # 5. Extract cls_output = x[:, 0], dist_output = x[:, 1]
        # 6. cls_logits = cls_head(cls_output), dist_logits = dist_head(dist_output)
        # 7. If training: return (cls_logits, dist_logits); else: return avg
        raise NotImplementedError


def deit_distillation_loss(cls_logits: torch.Tensor, dist_logits: torch.Tensor,
                           labels: torch.Tensor, teacher_logits: torch.Tensor,
                           alpha: float = 0.5, tau: float = 3.0) -> torch.Tensor:
    """
    Combined DeiT loss.

    total_loss = (1 - alpha) * CE(cls_logits, labels)
               + alpha * KL(softmax(dist_logits/tau), softmax(teacher_logits/tau))

    Args:
        cls_logits:      (B, num_classes) from CLS head
        dist_logits:     (B, num_classes) from DIST head
        labels:          (B,) ground truth labels
        teacher_logits:  (B, num_classes) from frozen teacher model
        alpha:           weight for distillation loss (default 0.5)
        tau:             temperature for soft labels (default 3.0)
    """
    # your code here
    raise NotImplementedError
