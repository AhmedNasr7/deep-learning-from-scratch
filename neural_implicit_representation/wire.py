"""
WIRE — Wavelet Implicit Neural Representations (Saragadam et al., 2023).

Uses Gabor wavelet activations instead of sin (SIREN) or ReLU.
Gabor wavelets = Gaussian envelope × sinusoid — provides both
spatial localization AND frequency selectivity (best of both worlds).

Advantages over SIREN:
  - Better at representing spatially-varying frequencies
  - More stable training (Gaussian envelope prevents blow-up)
  - Naturally handles multi-scale signals
"""

import torch
import torch.nn as nn
import math


class GaborWaveletLayer(nn.Module):
    """
    Gabor wavelet activation layer.

    activation(x) = exp(-|s · Wx|²) · sin(ω₀ · Wx + b)

    s controls the spatial extent (envelope), ω₀ controls frequency.

    Args:
        in_features:  input dimension
        out_features: output dimension
        omega_0:      frequency scaling
        sigma_0:      Gaussian envelope width
        is_first:     affects initialization

    Input:  (batch, in_features)
    Output: (batch, out_features)
    """

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 20.0, sigma_0: float = 40.0,
                 is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        # your code here
        # Initialize weights similar to SIREN but adapted for Gabor
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Compute linear projection: h = Wx + b
        # 2. Gaussian envelope: exp(-(sigma_0 * h)²)
        # 3. Sinusoidal: sin(omega_0 * h)
        # 4. Return envelope * sinusoidal
        raise NotImplementedError


class WIRE(nn.Module):
    """
    Full WIRE network for implicit neural representations.

    Same structure as SIREN but with GaborWaveletLayer activations.

    Args:
        in_features:     input dim (2 for images, 3 for 3D)
        hidden_features: hidden layer width
        hidden_layers:   number of hidden layers
        out_features:    output dim
        omega_0:         frequency factor
        sigma_0:         envelope width

    Input:  (batch, in_features) — coordinates in [-1, 1]
    Output: (batch, out_features) — predicted signal
    """

    def __init__(self, in_features: int = 2, hidden_features: int = 256,
                 hidden_layers: int = 3, out_features: int = 1,
                 omega_0: float = 20.0, sigma_0: float = 40.0):
        super().__init__()

        # your code here
        # 1. First GaborWaveletLayer (is_first=True)
        # 2. N hidden GaborWaveletLayers
        # 3. Final nn.Linear (no wavelet activation)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        raise NotImplementedError
