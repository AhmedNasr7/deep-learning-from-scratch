"""
SimCLR from scratch.

Classic image contrastive learning with projection heads and pairwise loss.
"""

import torch
import torch.nn as nn


class SimCLR(nn.Module):
    """SimCLR model scaffold.

    This class should implement a backbone encoder + projection head.
    """

    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder.output_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images and return contrastive embeddings."""
        raise NotImplementedError("SimCLR forward pass is not implemented yet.")
