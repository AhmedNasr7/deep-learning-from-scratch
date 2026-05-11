"""
Sinusoidal Positional Encoding (Vaswani et al., 2017).

Fixed (not learned) encoding using sin/cos at different frequencies.

This file is designed to grow — add variants here:
  - Learned positional embeddings
  - Rotary Position Embedding (RoPE)
  - ALiBi (Attention with Linear Biases)
  - etc.
"""

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Precomputes a (max_seq_len, d_model) matrix of sin/cos values
    and adds it to the input embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model:     embedding dimension
        max_seq_len: maximum sequence length to precompute
        dropout:     dropout rate

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model), with positional encoding added
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # your code here
        # 1. Create a (max_seq_len, d_model) tensor of zeros
        # 2. Compute position indices and dimension indices
        # 3. Fill even columns with sin, odd columns with cos
        # 4. Register as a buffer (not a parameter — not learned)
        #    self.register_buffer('pe', pe)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to x and apply dropout.

        Input:  (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)
        """
        # your code here
        # Slice self.pe to match seq_len, add to x, apply dropout
        raise NotImplementedError
