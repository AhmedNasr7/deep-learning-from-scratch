"""
DeepSDF — Learning Continuous Signed Distance Functions (Park et al., 2019).

Represents 3D geometry as the zero-level set of a learned SDF:
  - SDF(x) < 0 → inside the shape
  - SDF(x) = 0 → on the surface
  - SDF(x) > 0 → outside the shape

Key idea: condition the MLP on a learned latent code z per shape,
so one network can represent many shapes.
"""

import torch
import torch.nn as nn


class DeepSDF(nn.Module):
    """
    DeepSDF network: (latent_code, 3D_point) → signed distance.

    Architecture:
      - Input: concatenate latent code z and point x → [z; x]
      - 8 FC layers (512 units, ReLU) with skip connection at layer 4
      - Output: scalar SDF value (clamped during training)

    Args:
        latent_dim:  latent code dimension per shape
        hidden_dim:  hidden layer width
        num_layers:  number of FC layers
        skip_layer:  which layer gets the skip connection

    Input:
        z: (batch, latent_dim) — shape latent code
        x: (batch, 3) — query 3D point

    Output: (batch, 1) — signed distance value
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 8, skip_layer: int = 4):
        super().__init__()
        self.skip_layer = skip_layer
        in_dim = latent_dim + 3  # z concatenated with xyz

        # your code here
        # 1. Build FC layers with skip connection at skip_layer
        #    (at skip_layer, concatenate original input again)
        # 2. Final layer outputs scalar SDF
        raise NotImplementedError

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Concatenate [z, x]
        # 2. Pass through FC layers (with skip connection)
        # 3. Output SDF value
        raise NotImplementedError
