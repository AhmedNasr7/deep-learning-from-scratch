"""
Adam & AdamW — Adaptive Moment Estimation.

Adam combines momentum (1st moment) with RMSProp (2nd moment) and adds
bias correction for the first few steps.

AdamW fixes the weight decay bug in Adam by decoupling L2 regularization
from the gradient update (Loshchilov & Hutter, 2019).

References:
  - Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
  - Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
"""

import torch
from torch.optim import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad         # 1st moment
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2       # 2nd moment
        m̂_t = m_t / (1 - beta1^t)                          # bias correction
        v̂_t = v_t / (1 - beta2^t)
        theta = theta - lr * m̂_t / (sqrt(v̂_t) + eps)

    Args:
        params:       model parameters
        lr:           learning rate (default: 1e-3)
        betas:        (beta1, beta2) moment decay rates
        eps:          numerical stability constant
        weight_decay: L2 regularization (applied to gradient, not param directly)
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # your code here
        # For each parameter:
        #   1. Apply weight decay to gradient
        #   2. Update 1st moment: m = beta1 * m + (1-beta1) * grad
        #   3. Update 2nd moment: v = beta2 * v + (1-beta2) * grad^2
        #   4. Bias-correct: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
        #   5. Update: param -= lr * m_hat / (sqrt(v_hat) + eps)
        raise NotImplementedError


class AdamW(Optimizer):
    """
    AdamW optimizer — Adam with decoupled weight decay.

    Key difference from Adam: weight decay is applied directly to the parameter
    (not mixed into the gradient), so the adaptive learning rate doesn't scale it.

    Update rule:
        theta = theta - lr * weight_decay * theta            # decoupled decay
        (then apply same Adam moment updates as above)

    Args:
        params:       model parameters
        lr:           learning rate (default: 1e-3)
        betas:        (beta1, beta2) moment decay rates
        eps:          numerical stability constant
        weight_decay: decoupled weight decay (default: 0.01)
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # your code here
        # For each parameter:
        #   1. Apply DECOUPLED weight decay: param -= lr * weight_decay * param
        #   2. Update 1st and 2nd moments (same as Adam)
        #   3. Bias correction
        #   4. Update: param -= lr * m_hat / (sqrt(v_hat) + eps)
        raise NotImplementedError
