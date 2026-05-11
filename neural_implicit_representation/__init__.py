"""
Neural Implicit Representations — coordinate-based neural networks
that represent signals (images, 3D shapes, radiance fields) as
continuous functions parameterized by neural networks.

Contents:
  inr.py       — Basic MLP (ReLU baseline)
  siren.py     — SIREN (sinusoidal activations)
  nerf.py      — NeRF (radiance fields)
  deep_sdf.py  — DeepSDF (signed distance functions)
  wire.py      — WIRE (wavelet implicit representation)
"""

from .siren import SIREN, SineLayer
from .nerf import NeRF
from .inr import INR
from .deep_sdf import DeepSDF
from .wire import WIRE, GaborWaveletLayer
