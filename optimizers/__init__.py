"""
Optimizers — Common optimization algorithms from scratch.

All optimizers subclass torch.optim.Optimizer — they are drop-in compatible
with any PyTorch training loop and can be tested with the optimizer test suite.

Modules:
  sgd.py          — SGD, SGD+Momentum, Nesterov Accelerated Gradient
  rmsprop.py      — RMSProp (Hinton, 2012)
  adam.py         — Adam (Kingma & Ba, 2015), AdamW (Loshchilov & Hutter, 2019)
  lbfgs.py        — L-BFGS quasi-Newton (Liu & Nocedal, 1989)
  second_order.py — Newton's Method, Natural Gradient
  lr_scheduler.py — WarmupCosine, CyclicalLR

Test:
  python -m tests.optimizers.test_optimizers
"""

from .sgd import SGD
from .rmsprop import RMSProp
from .adam import Adam, AdamW
from .lbfgs import LBFGS
from .second_order import NewtonOptimizer, NaturalGradient

__all__ = ["SGD", "RMSProp", "Adam", "AdamW", "LBFGS",
           "NewtonOptimizer", "NaturalGradient"]
