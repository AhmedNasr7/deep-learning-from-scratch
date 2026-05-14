"""
Position-wise Feed-Forward Network.

Two linear layers with an activation in between.
Applied independently to each position in the sequence.
"""

import torch
import torch.nn as nn
from transformer.activations import GELU

class PositionWiseFeedForward(nn.Module):
    """
    FFN(x) = Linear(activation(Linear(x)))

    Args:
        d_model: input and output dimension
        d_ff:    hidden dimension (typically 4 * d_model)
        dropout: dropout rate

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fc1 → activation (ReLU for transformer, GELU for GPT/ViT) → dropout → fc2
        
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


