from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the encoder-decoder Transformer."""

    # --- Vocabulary (set by your tokenizer after training) ---
    src_vocab_size: int = 10_000
    tgt_vocab_size: int = 10_000
    pad_id: int = 0

    # --- Architecture ---
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024            # typically 4 * d_model
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 20
    warmup_steps: int = 4000
