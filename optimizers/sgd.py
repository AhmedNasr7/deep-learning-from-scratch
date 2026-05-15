"""
SGD — Stochastic Gradient Descent.

Variants:
  - Vanilla SGD
  - SGD with Momentum (Polyak, 1964)
  - Nesterov Accelerated Gradient (NAG)

Reference:
  - Robbins & Monro, "A Stochastic Approximation Method", 1951
  - Polyak, "Some methods of speeding up the convergence of iteration methods", 1964
  - Sutskever et al., "On the importance of initialization and momentum in deep learning", 2013
"""

import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov acceleration.

    Update rule (with momentum):
        v_t = momentum * v_{t-1} + grad
        theta = theta - lr * v_t

    Update rule (Nesterov):
        v_t = momentum * v_{t-1} + grad(theta - momentum * v_{t-1})
        theta = theta - lr * v_t

    Args:
        params:   model parameters
        lr:       learning rate
        momentum: momentum factor (0 = vanilla SGD)
        nesterov: use Nesterov accelerated gradient
        weight_decay: L2 regularization coefficient
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # your code here
        # For each parameter group and each parameter:
        #   1. Apply weight decay: grad += weight_decay * param
        #   2. If momentum > 0: update velocity buffer
        #   3. If nesterov: use lookahead gradient
        #   4. Update param: param -= lr * grad (or velocity)
        raise NotImplementedError
