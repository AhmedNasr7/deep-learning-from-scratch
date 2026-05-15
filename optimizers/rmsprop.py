"""
RMSProp — Root Mean Square Propagation.

Adapts the learning rate per-parameter by dividing by a running average
of squared gradients, preventing the learning rate from growing too large
for frequently updated parameters.

Reference:
  - Hinton, unpublished lecture notes, 2012
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
"""

import torch
from torch.optim import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    Update rule:
        v_t = rho * v_{t-1} + (1 - rho) * grad^2
        theta = theta - lr * grad / (sqrt(v_t) + eps)

    Args:
        params:      model parameters
        lr:          learning rate
        rho:         decay factor for squared gradient moving average (default: 0.99)
        eps:         numerical stability constant
        weight_decay: L2 regularization coefficient
        momentum:    optional momentum factor
    """

    def __init__(self, params, lr: float = 1e-3, rho: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # your code here
        # For each parameter:
        #   1. Apply weight decay: grad += weight_decay * param
        #   2. Update squared grad running average: v = rho * v + (1-rho) * grad^2
        #   3. Optionally update momentum buffer
        #   4. Update: param -= lr * grad / (sqrt(v) + eps)
        raise NotImplementedError
