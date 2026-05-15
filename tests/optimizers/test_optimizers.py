"""
Optimizer test suite — validates each custom optimizer against a simple MLP.

Since all optimizers subclass torch.optim.Optimizer, they are drop-in
compatible with any PyTorch training loop.

Test strategy: train a tiny 2-layer MLP on a sin(x) regression task.
A correct optimizer should reduce loss to < 0.01 within 500 steps.

Usage:
    python -m tests.optimizers.test_optimizers
    python -m pytest tests/optimizers/test_optimizers.py -v
"""

import torch
import torch.nn as nn
import pytest

# Import our custom optimizers
from optimizers.sgd import SGD
from optimizers.adam import Adam, AdamW
from optimizers.rmsprop import RMSProp

# ─── Shared Fixtures ──────────────────────────────────────────────────────────

def make_mlp() -> nn.Module:
    """Tiny 2-layer MLP: 1 → 32 → 32 → 1."""
    return nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )

def make_data(n: int = 256, device: str = "cpu"):
    """Simple sin(x) regression dataset."""
    x = torch.linspace(-3.14, 3.14, n, device=device).unsqueeze(1)
    y = torch.sin(x)
    return x, y

def train_loop(optimizer: torch.optim.Optimizer, model: nn.Module,
               steps: int = 500, device: str = "cpu") -> float:
    """Run training loop and return final loss."""
    x, y = make_data(device=device)
    criterion = nn.MSELoss()
    model.to(device).train()

    for _ in range(steps):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_sgd():
    model = make_mlp()
    opt = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    final_loss = train_loop(opt, model)
    assert final_loss < 0.01, f"SGD did not converge: loss={final_loss:.4f}"
    print(f"✅ SGD final loss: {final_loss:.6f}")


def test_adam():
    model = make_mlp()
    opt = Adam(model.parameters(), lr=1e-3)
    final_loss = train_loop(opt, model)
    assert final_loss < 0.01, f"Adam did not converge: loss={final_loss:.4f}"
    print(f"✅ Adam final loss: {final_loss:.6f}")


def test_adamw():
    model = make_mlp()
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    final_loss = train_loop(opt, model)
    assert final_loss < 0.01, f"AdamW did not converge: loss={final_loss:.4f}"
    print(f"✅ AdamW final loss: {final_loss:.6f}")


def test_rmsprop():
    model = make_mlp()
    opt = RMSProp(model.parameters(), lr=1e-3)
    final_loss = train_loop(opt, model)
    assert final_loss < 0.01, f"RMSProp did not converge: loss={final_loss:.4f}"
    print(f"✅ RMSProp final loss: {final_loss:.6f}")


if __name__ == "__main__":
    test_sgd()
    test_adam()
    test_adamw()
    test_rmsprop()
    print("\n✅ All optimizer tests passed!")
