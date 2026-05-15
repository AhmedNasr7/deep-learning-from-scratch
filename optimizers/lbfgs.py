"""
L-BFGS — Limited-memory Broyden–Fletcher–Goldfarb–Shanno.

A quasi-Newton second-order optimizer that approximates the Hessian inverse
using a limited history of gradient vectors (m most recent steps).
Much more efficient than full Newton's method — O(md) storage vs O(d^2).

Commonly used for:
  - Small datasets / final fine-tuning passes
  - SIREN / neural implicit representations
  - Physics-informed neural networks (PINNs)

Reference:
  - Liu & Nocedal, "On the limited memory BFGS method for large scale optimization", 1989
"""

import torch
from torch.optim import Optimizer


class LBFGS(Optimizer):
    """
    L-BFGS optimizer.

    Unlike first-order methods, L-BFGS requires the full loss value
    at each step (via a closure), because it performs a line search.

    Key concepts:
        - history_size: how many (s, y) pairs to store (s = param diff, y = grad diff)
        - two-loop recursion: efficient H^{-1} * grad computation without forming H
        - Wolfe conditions: line search acceptance criteria

    Args:
        params:       model parameters
        lr:           initial step size for line search
        history_size: number of (grad, param) pairs to remember (default: 10)
        max_iter:     max line-search iterations per step
        tolerance_grad:  gradient convergence threshold
        tolerance_change: parameter change convergence threshold
    """

    def __init__(self, params, lr: float = 1.0, history_size: int = 10,
                 max_iter: int = 20, tolerance_grad: float = 1e-7,
                 tolerance_change: float = 1e-9):
        defaults = dict(lr=lr, history_size=history_size, max_iter=max_iter,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change)
        super().__init__(params, defaults)

    def step(self, closure):
        """
        NOTE: L-BFGS requires a closure that re-evaluates the loss.

        Args:
            closure: callable that computes and returns the loss

        Steps:
            1. Evaluate loss and gradients via closure
            2. Compute search direction: d = -H^{-1} * grad (two-loop recursion)
            3. Line search along d satisfying Wolfe conditions
            4. Update params, store (s, y) pair in history
            5. If history full, discard oldest pair
        """
        raise NotImplementedError
