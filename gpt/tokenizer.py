"""
Tokenizer adapter for GPT-2.

Wraps EfficientBPETokenizer and adds special token handling
(PAD, BOS, EOS) needed for language modeling.
"""

from typing import List

from tokenizers import EfficientBPETokenizer


class GPTTokenizer:
    """
    Wrapper around EfficientBPETokenizer for GPT-2 usage.

    Adds special tokens on top of the BPE vocab:
      PAD = bpe_vocab_size
      BOS = bpe_vocab_size + 1
      EOS = bpe_vocab_size + 2

    Usage:
        tokenizer = GPTTokenizer()
        tokenizer.train(corpus, vocab_size=10_000)
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
    """

    def __init__(self):
        self.bpe = EfficientBPETokenizer()
        self._pad_id = None
        self._bos_id = None
        self._eos_id = None

    @property
    def vocab_size(self) -> int:
        """Total vocab including special tokens."""
        return len(self.bpe.vocab) + 3  # PAD, BOS, EOS

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def train(self, text: str, target_vocab_size: int) -> "GPTTokenizer":
        """
        Train BPE on text, then assign special token IDs.

        Args:
            text:              raw training corpus
            target_vocab_size: desired BPE vocab size (before adding special tokens)
        """
        self.bpe.train(text, target_vocab_size)

        bpe_size = len(self.bpe.vocab)
        self._pad_id = bpe_size
        self._bos_id = bpe_size + 1
        self._eos_id = bpe_size + 2

        return self

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """
        Encode text to token IDs using trained BPE.

        Args:
            text:        raw text string
            add_special: if True, wraps with [BOS] ... [EOS]
        """
        ids = self.bpe.encode(text)
        if add_special:
            ids = [self._bos_id] + ids + [self._eos_id]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text, filtering special tokens."""
        special = {self._pad_id, self._bos_id, self._eos_id}
        filtered = [t for t in token_ids if t not in special]
        return self.bpe.decode(filtered)
