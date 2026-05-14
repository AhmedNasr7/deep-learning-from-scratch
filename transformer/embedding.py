"""
Token + Positional Embedding.

You can either:
  - Use nn.Embedding (PyTorch built-in)
  - Implement a lookup table from scratch with nn.Parameter

Both are valid — nn.Embedding is just a thin wrapper around an nn.Parameter
weight matrix with an indexing operation.
"""

import torch
import torch.nn as nn
import math

from positional_encoding import SinusoidalPositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    Token embedding + positional encoding for the Transformer.

    token_ids → embedding lookup → scale by √d_model → add positional encoding → dropout

    Args:
        vocab_size:  number of tokens in vocabulary
        d_model:     embedding dimension
        max_seq_len: maximum sequence length
        dropout:     dropout rate

    Input:  (batch, seq_len) of token IDs (long)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 1. Create token embedding (nn.Embedding or nn.Parameter)
        # 2. Create positional encoding (reuse SinusoidalPositionalEncoding)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(max_seq_len=max_seq_len, dropout=dropout, d_model=d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Input:  (batch, seq_len) of token IDs
        Output: (batch, seq_len, d_model) — embedded and position-encoded
        """
        # 1. Lookup token embeddings
        # 2. Scale by sqrt(d_model) — "Attention Is All You Need" Sec 3.4
        # 3. Add positional encoding

        embeds = self.embedding(tokens) * math.sqrt(self.d_model)
        return self.pos_encoding(embeds)




