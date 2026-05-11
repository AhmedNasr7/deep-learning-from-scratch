"""
Autograd — Automatic differentiation engine from scratch.

Build a minimal computation graph with forward/backward passes
to understand how PyTorch's autograd works under the hood.

Planned files:
  tensor.py     — Custom Tensor class with grad tracking
  operators.py  — Add, Mul, MatMul, ReLU as graph nodes
  engine.py     — Topological sort + backward pass
  test.py       — Verify gradients against PyTorch autograd
"""

# Future implementations:
# from .tensor import Tensor
# from .engine import backward
