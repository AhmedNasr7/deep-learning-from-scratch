"""
LR Schedulers — Learning Rate Schedule Strategies.

References:
  - Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", 2017
  - Smith, "Cyclical Learning Rates for Training Neural Networks", 2017
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    Used in: ViT, GPT-2, BERT pre-training — arguably the default for transformers.

    Phase 1 (warmup): lr linearly increases from 0 to base_lr over warmup_steps
    Phase 2 (cosine): lr decays via cosine from base_lr → min_lr over remaining steps

    Args:
        optimizer:     wrapped optimizer
        warmup_steps:  number of warmup steps
        total_steps:   total training steps
        min_lr:        minimum lr at end of cosine decay
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 total_steps: int, min_lr: float = 0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        # your code here
        # If step < warmup_steps: lr = base_lr * (step / warmup_steps)
        # Else: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        raise NotImplementedError


class CyclicalLR(_LRScheduler):
    """
    Cyclical Learning Rate (CLR) — triangular or cosine cycle.

    Oscillates lr between base_lr and max_lr. Helps escape local minima
    and often achieves better results without careful lr tuning.

    Args:
        optimizer:   wrapped optimizer
        base_lr:     minimum lr
        max_lr:      maximum lr
        step_size:   half-cycle length in steps
        mode:        'triangular', 'triangular2', or 'exp_range'
    """

    def __init__(self, optimizer: Optimizer, base_lr: float, max_lr: float,
                 step_size: int = 2000, mode: str = "triangular"):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        super().__init__(optimizer)

    def get_lr(self):
        # your code here
        raise NotImplementedError
