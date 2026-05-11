"""
Transformer model — encoder-decoder architecture (Vaswani et al., 2017).

Composes the building blocks from attention, feed_forward, normalization,
and embedding modules into a full seq2seq model for translation.
"""

import torch
import torch.nn as nn

from .config import TransformerConfig
from attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .normalization import LayerNorm
from .embedding import TransformerEmbedding


class TransformerEncoderBlock(nn.Module):
    """
    Single encoder block: self-attention → add & norm → FFN → add & norm.

    Args:
        d_model, n_heads, d_ff, dropout: architecture params

    Input:  x (batch, seq_len, d_model), src_mask (optional)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention + residual + norm (post-norm style)
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TransformerDecoderBlock(nn.Module):
    """
    Single decoder block: masked self-attention → add & norm →
                          cross-attention → add & norm → FFN → add & norm.

    Args:
        d_model, n_heads, d_ff, dropout: architecture params

    Input:
        x:          (batch, tgt_len, d_model) — decoder input
        enc_output: (batch, src_len, d_model) — encoder output
        src_mask:   (optional) mask for encoder output
        tgt_mask:   (optional) causal mask for decoder self-attention

    Output: (batch, tgt_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Masked self-attention + residual + norm
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention (Q=decoder, K=V=encoder) + residual + norm
        cross_attn_out = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Feed-forward + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N encoder blocks with input embedding.

    Args:
        config: TransformerConfig

    Input:  src (batch, src_len) token IDs, src_mask (optional)
    Output: (batch, src_len, d_model) — encoded representation
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embedding = TransformerEmbedding(
            config.src_vocab_size, config.d_model, config.max_seq_len, config.dropout
        )
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_encoder_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(src)
        for block in self.blocks:
            x = block(x, src_mask)
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of N decoder blocks with input embedding.

    Args:
        config: TransformerConfig

    Input:
        tgt:        (batch, tgt_len) token IDs
        enc_output: (batch, src_len, d_model)
        src_mask, tgt_mask: optional masks

    Output: (batch, tgt_len, d_model)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embedding = TransformerEmbedding(
            config.tgt_vocab_size, config.d_model, config.max_seq_len, config.dropout
        )
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_decoder_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt)
        for block in self.blocks:
            x = block(x, enc_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer for sequence-to-sequence tasks.

    Args:
        config: TransformerConfig

    Input:
        src: (batch, src_len)  — source token IDs
        tgt: (batch, tgt_len)  — target token IDs
        src_mask, tgt_mask: optional masks

    Output: (batch, tgt_len, tgt_vocab_size) — logits over target vocabulary
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.output_proj = nn.Linear(config.d_model, config.tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.output_proj(dec_output)
        return logits

    # -------------------------------------------------------------------------
    # MASKS GUIDE
    #
    # Attention scores have shape (batch, n_heads, seq_q, seq_k).
    # Before softmax, we set certain positions to -inf so they get zero
    # probability. Two reasons to mask:
    #
    # 1. CAUSAL MASK (decoder self-attention)
    #    The decoder generates tokens one at a time, left-to-right.
    #    At position t, it should only see positions [0, 1, ..., t],
    #    NOT future tokens [t+1, t+2, ...].
    #    Without this: the model cheats by looking at the answer during training,
    #    learns nothing, and can't generate at inference time.
    #
    #    Shape: (T, T) upper-triangular = True (blocked)
    #
    #    Example for T=4:
    #        [[False,  True,  True,  True],    token 0 sees [0]
    #         [False, False,  True,  True],    token 1 sees [0, 1]
    #         [False, False, False,  True],    token 2 sees [0, 1, 2]
    #         [False, False, False, False]]    token 3 sees [0, 1, 2, 3]
    #
    # 2. PADDING MASK (encoder + decoder cross-attention)
    #    Sequences in a batch have different lengths, padded with PAD tokens.
    #    Attention should ignore PAD positions — they carry no information.
    #    Without this: the model attends to garbage PAD values, hurting quality.
    #
    #    Shape: (batch, 1, 1, seq_len) — broadcasts across heads and query positions
    #
    # Usage in train.py:
    #    src_mask = Transformer.generate_padding_mask(src, pad_id)     # encoder
    #    tgt_mask = Transformer.generate_causal_mask(tgt_len)          # decoder
    #    Both are passed to forward() and routed to the attention layers.
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_causal_mask(size: int) -> torch.Tensor:
        """
        Generate an upper-triangular causal mask for decoder self-attention.

        Prevents the decoder from attending to future positions.
        True = blocked (will be set to -inf before softmax).

        Args:
            size: target sequence length

        Returns:
            (size, size) boolean mask
        """
        return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    @staticmethod
    def generate_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        Generate a padding mask from token IDs.

        Prevents attention from attending to PAD tokens.
        True = blocked (will be set to -inf before softmax).

        Shape is (batch, 1, 1, seq_len) so it broadcasts correctly
        with attention scores of shape (batch, n_heads, seq_q, seq_k).

        Args:
            tokens: (batch, seq_len) token IDs
            pad_id: the padding token ID

        Returns:
            (batch, 1, 1, seq_len) boolean mask
        """
        return (tokens == pad_id).unsqueeze(1).unsqueeze(2)
