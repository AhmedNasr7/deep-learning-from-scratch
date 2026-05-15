"""
Losses and Metrics — Common loss functions and evaluation metrics from scratch.

Modules:
  cross_entropy.py — CrossEntropyLoss, BinaryCrossEntropyLoss, FocalLoss
  mse.py           — MSELoss, MAELoss, HuberLoss
  contrastive.py   — ContrastiveLoss, TripletLoss, InfoNCELoss
  metrics.py       — accuracy, precision_recall_f1, perplexity, bleu_score, mAP
"""

from .cross_entropy import CrossEntropyLoss, BinaryCrossEntropyLoss, FocalLoss
from .mse import MSELoss, MAELoss, HuberLoss
from .contrastive import ContrastiveLoss, TripletLoss, InfoNCELoss
from .metrics import accuracy, precision_recall_f1, perplexity, bleu_score, mean_average_precision

__all__ = [
    "CrossEntropyLoss", "BinaryCrossEntropyLoss", "FocalLoss",
    "MSELoss", "MAELoss", "HuberLoss",
    "ContrastiveLoss", "TripletLoss", "InfoNCELoss",
    "accuracy", "precision_recall_f1", "perplexity", "bleu_score",
    "mean_average_precision",
]
