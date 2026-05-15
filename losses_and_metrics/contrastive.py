"""
Contrastive and Metric Learning Losses.

References:
  - Hadsell et al., "Dimensionality Reduction by Learning an Invariant Mapping", 2006
  - Schroff et al., "FaceNet: A Unified Embedding for Face Recognition", 2015
  - Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Siamese / pairwise contrastive loss (Hadsell et al.).

    Loss = (1-y) * d^2 + y * max(0, margin - d)^2

    where d = ||f(x1) - f(x2)||_2, y=0 for similar pairs, y=1 for dissimilar.

    Args:
        margin: minimum distance between dissimilar pairs (default: 1.0)
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1:   (B, D) embeddings for first items
            emb2:   (B, D) embeddings for second items
            labels: (B,) 0=similar, 1=dissimilar
        """
        raise NotImplementedError


class TripletLoss(nn.Module):
    """
    Triplet Loss (Schroff et al.).

    Loss = max(0, d(a, p) - d(a, n) + margin)

    Pulls anchor-positive pairs together, pushes anchor-negative pairs apart.

    Args:
        margin: minimum gap between positive and negative distances (default: 0.3)
        distance: 'euclidean' or 'cosine'
    """
    def __init__(self, margin: float = 0.3, distance: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor:   (B, D)
            positive: (B, D) — same class as anchor
            negative: (B, D) — different class from anchor
        """
        raise NotImplementedError


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent Loss — used in SimCLR, CLIP, MoCo.

    For a batch of B augmented pairs: treats the i-th pair as positive,
    all other 2(B-1) views as negatives.

    Loss = -log(exp(sim(q, k+)/τ) / Σ_j exp(sim(q, k_j)/τ))

    Args:
        temperature: scaling factor τ (default: 0.07)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) L2-normalized embeddings (view 1)
            z2: (B, D) L2-normalized embeddings (view 2)
        Returns:
            scalar loss
        """
        raise NotImplementedError
