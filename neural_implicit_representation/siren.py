"""
SIREN — Sinusoidal Representation Networks (Sitzmann et al., 2020).

Uses periodic sine activations instead of ReLU, enabling the network
to represent signals and their derivatives accurately. Key insight:
the initialization scheme (ω₀ scaling) is critical for training.
"""

import torch
import torch.nn as nn
import math


class SineLayer(nn.Module):
    """
    Single SIREN layer: Linear → sin(ω₀ · x)

    Args:
        in_features:  input dimension
        out_features: output dimension
        omega_0:      frequency scaling factor (30.0 is standard)
        is_first:     if True, use first-layer initialization (uniform ±1/in)
                      if False, use hidden-layer init (uniform ±√(6/in)/ω₀)

    Input:  (batch, in_features)
    Output: (batch, out_features)
    """

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        # your code here
        # Initialize weights according to SIREN paper:
        #   first layer:  uniform(-1/in, 1/in)
        #   hidden layers: uniform(-√(6/in)/ω₀, √(6/in)/ω₀)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # sin(ω₀ · linear(x))
        raise NotImplementedError


class SIREN(nn.Module):
    """
    Full SIREN network for implicit neural representations.

    Maps coordinates (x, y[, z]) → signal value (RGB, SDF, etc.)

    Args:
        in_features:    input dim (2 for images, 3 for 3D)
        hidden_features: hidden layer width
        hidden_layers:   number of hidden SineLayer
        out_features:    output dim (1 for grayscale, 3 for RGB)
        omega_0:         frequency factor

    Input:  (batch, in_features) — coordinates in [-1, 1]
    Output: (batch, out_features) — predicted signal
    """

    def __init__(self, in_features: int = 2, hidden_features: int = 256,
                 hidden_layers: int = 3, out_features: int = 1, omega_0: float = 30.0):
        super().__init__()

        # your code here
        # 1. First SineLayer (is_first=True)
        # 2. N hidden SineLayers
        # 3. Final nn.Linear (no sine activation)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError
