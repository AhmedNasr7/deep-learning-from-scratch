"""
GPT-2 model.

Decoder-only transformer: token_emb + pos_emb → N × GPTBlock → LayerNorm → lm_head

Key design:
  - Pre-norm (LayerNorm before attention/FFN)
  - Learned positional embeddings
  - Weight tying: lm_head shares weights with token embedding
"""

import torch
import torch.nn as nn

from .config import GPT2Config
from .layers import CausalSelfAttention, GPTFeedForward
from transformer.normalization import LayerNorm


class GPTBlock(nn.Module):
    """
    Single GPT-2 transformer block.

    Pre-norm residual: x = x + attn(norm(x)), x = x + ffn(norm(x))

    Args:
        config: GPT2Config

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.norm1 = LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.max_seq_len, config.dropout)
        self.norm2 = LayerNorm(config.d_model)
        self.ffn = GPTFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # your code here
        # Pre-norm residual:
        # 1. x = x + attn(norm1(x))
        # 2. x = x + ffn(norm2(x))
        raise NotImplementedError


class GPT2(nn.Module):
    """
    GPT-2 language model.

    Args:
        config: GPT2Config

    Input:  (batch, seq_len) of token IDs
    Output: (batch, seq_len, vocab_size) — logits over vocabulary

    Architecture:
        token_embedding + position_embedding → dropout →
        N × GPTBlock → LayerNorm → lm_head (linear to vocab)

    Weight tying: lm_head.weight = token_embedding.weight
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Embeddings (learned, not sinusoidal)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.norm = LayerNorm(config.d_model)

        # Language model head (weight-tied with token_emb)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # your code here
        # 1. Token embeddings: (B, T) → (B, T, D)
        # 2. Position embeddings: create position indices [0, 1, ..., T-1]
        # 3. Add token + position embeddings, apply dropout
        # 4. Pass through all GPTBlocks
        # 5. Final LayerNorm
        # 6. Project to vocab with lm_head → logits (B, T, vocab_size)
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt_ids:     (1, prompt_len) starting token IDs
            max_new_tokens: how many tokens to generate
            temperature:    sampling temperature (1.0 = normal, <1 = greedy, >1 = creative)

        Returns:
            (1, prompt_len + max_new_tokens) generated token IDs
        """
        # your code here
        # Loop max_new_tokens times:
        #   1. Forward pass on current sequence (truncate to max_seq_len if needed)
        #   2. Get logits for last position, apply temperature
        #   3. Sample from softmax distribution (or argmax for greedy)
        #   4. Append sampled token to sequence
        raise NotImplementedError
