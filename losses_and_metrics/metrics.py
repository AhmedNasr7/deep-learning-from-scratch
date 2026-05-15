"""
Evaluation Metrics — classification, generation, and retrieval.
"""

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: int = 1) -> float:
    """
    Top-k accuracy.

    Args:
        logits:  (B, C) raw scores or probabilities
        targets: (B,) integer class indices
        topk:    k for top-k accuracy (1 = standard accuracy)

    Returns:
        accuracy as a float in [0, 1]
    """
    raise NotImplementedError


def precision_recall_f1(preds: torch.Tensor, targets: torch.Tensor,
                        num_classes: int, average: str = "macro") -> dict:
    """
    Precision, Recall, and F1 score.

    Args:
        preds:       (B,) predicted class indices
        targets:     (B,) ground truth class indices
        num_classes: total number of classes
        average:     'macro' (unweighted mean) or 'micro' (global counts)

    Returns:
        dict with keys: 'precision', 'recall', 'f1'
    """
    raise NotImplementedError


def perplexity(log_probs: torch.Tensor) -> float:
    """
    Perplexity for language model evaluation.

    PP = exp(-1/T * Σ log p(w_t))

    Args:
        log_probs: (T,) per-token log probabilities from the model

    Returns:
        perplexity as a scalar float
    """
    raise NotImplementedError


def bleu_score(hypothesis: list, references: list, max_n: int = 4) -> float:
    """
    BLEU score for machine translation / text generation.

    Geometric mean of n-gram precisions * brevity penalty.

    Args:
        hypothesis:  list of tokens (the generated sentence)
        references:  list of list of tokens (reference sentences)
        max_n:       maximum n-gram order (default 4 → BLEU-4)

    Returns:
        BLEU score in [0, 1]
    """
    raise NotImplementedError


def mean_average_precision(embeddings: torch.Tensor, labels: torch.Tensor,
                           k: int = 10) -> float:
    """
    Mean Average Precision @ k for retrieval tasks.

    For each query, rank all other samples by cosine similarity,
    compute AP @ k, average over queries.

    Args:
        embeddings: (N, D) L2-normalized feature vectors
        labels:     (N,) class labels
        k:          number of retrieved results to consider

    Returns:
        mAP@k as a float
    """
    raise NotImplementedError
