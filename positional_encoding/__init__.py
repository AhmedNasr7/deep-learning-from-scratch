"""
Positional Encoding methods.

Modules:
  sinusoidal.py — Fixed sin/cos encoding (Vaswani et al., 2017)
  rope.py       — Rotary Position Embedding (Su et al., 2021) — used in LLaMA, Gemma
  alibi.py      — ALiBi linear bias (Press et al., 2022) — used in BLOOM
  learned.py    — Learned absolute embeddings (BERT/GPT-2) + T5 relative bias
"""

from .sinusoidal import SinusoidalPositionalEncoding
from .rope import RotaryPositionalEncoding
from .alibi import ALiBi
from .learned import LearnedPositionalEmbedding, RelativePositionalBias

__all__ = [
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEncoding",
    "ALiBi",
    "LearnedPositionalEmbedding",
    "RelativePositionalBias",
]
