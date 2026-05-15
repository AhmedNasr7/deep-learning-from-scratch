"""
Second-Order Optimization Methods.

Unlike first-order methods (SGD, Adam) that only use gradient information,
second-order methods use curvature (Hessian) to scale updates per-direction.

Methods:
  - Newton's Method:    theta -= H^{-1} * grad  (exact, O(d^3) per step)
  - Gauss-Newton:       approximates Hessian as J^T @ J for least-squares
  - Natural Gradient:   uses Fisher Information Matrix instead of Hessian
  - Shampoo:            structured preconditioning per layer (practical 2nd order)

References:
  - Nocedal & Wright, "Numerical Optimization", Ch. 6-7
  - Amari, "Natural Gradient Works Efficiently in Learning", 1998
  - Gupta et al., "Shampoo: Preconditioned Stochastic Tensor Optimization", 2018
"""

import torch
from torch.optim import Optimizer


class NewtonOptimizer(Optimizer):
    """
    Full Newton's method — uses exact Hessian inverse.

    WARNING: Only feasible for very small models (d < ~1000) due to O(d^2)
    memory and O(d^3) inversion cost. Educational purposes only.

    Update rule:
        H = d^2 L / d theta^2   (Hessian)
        theta = theta - lr * H^{-1} * grad

    Args:
        params: model parameters
        lr:     step size (1.0 = full Newton step)
    """

    def __init__(self, params, lr: float = 1.0):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure):
        """
        NOTE: Requires a closure for double-backward (Hessian computation).

        Steps:
            1. Compute grad and Hessian via torch.autograd.functional.hessian
            2. Compute H^{-1} via torch.linalg.solve or torch.inverse
            3. Update: param -= lr * H^{-1} @ grad
        """
        raise NotImplementedError


class NaturalGradient(Optimizer):
    """
    Natural Gradient Descent — uses Fisher Information Matrix as metric.

    Equivalent to steepest descent in the space of distributions (KL geometry),
    not Euclidean parameter space. Invariant to reparameterization.

    Update rule:
        F = E[grad log p(x;theta) @ grad log p(x;theta)^T]  # Fisher matrix
        theta = theta - lr * F^{-1} * grad

    Args:
        params:    model parameters
        lr:        learning rate
        damping:   regularization added to F diagonal (F + damping * I) for stability
    """

    def __init__(self, params, lr: float = 1e-3, damping: float = 1e-4):
        defaults = dict(lr=lr, damping=damping)
        super().__init__(params, defaults)

    def step(self, closure=None):
        # your code here
        # 1. Compute (approximate) Fisher Information Matrix
        # 2. Invert (F + damping * I)
        # 3. Multiply by gradient
        # 4. Update params
        raise NotImplementedError
