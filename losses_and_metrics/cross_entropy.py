"""
Cross-Entropy and Classification Losses.

Reference:
  - Bishop, "Pattern Recognition and Machine Learning", Ch. 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss for multi-class classification.

    Combines LogSoftmax + NLLLoss in one numerically stable step.

    Loss = -Σ y_true * log(softmax(logits))
         = -log(exp(logits[true_class]) / Σ exp(logits))

    Args:
        reduction:   'mean', 'sum', or 'none'
        label_smoothing: smoothing factor (0 = standard CE)
    """

    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw unnormalized scores
            targets: (B,) integer class indices

        Steps:
            1. Compute log_softmax(logits) for numerical stability
            2. If label_smoothing > 0: mix true label with uniform (1/C)
            3. Gather log-prob at true class index
            4. Apply reduction
        """
        raise NotImplementedError


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross-Entropy for binary or multi-label classification.

    Loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]

    Args:
        reduction: 'mean', 'sum', or 'none'
        from_logits: if True, applies sigmoid internally (numerically stable)
    """

    def __init__(self, reduction: str = "mean", from_logits: bool = True):
        super().__init__()
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, ...) — raw scores (or probabilities if from_logits=False)
            targets: (B, ...) — binary labels in {0, 1}
        """
        raise NotImplementedError


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples to focus on hard ones.

    Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Used in: object detection (RetinaNet), class-imbalanced problems.

    Reference:
      - Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        gamma:     focusing parameter (0 = standard CE, typical: 2)
        alpha:     class weighting factor (optional)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError
