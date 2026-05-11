"""
Tokenizer adapter for GPT-2.

Wraps the BPE tokenizers from the tokenizers/ module, adding
special token handling (PAD, BOS, EOS) needed for language modeling.

Plug in either NaiveBPETokenizer or EfficientBPETokenizer.
"""

from typing import List

# Import your tokenizer of choice:
# from tokenizers import NaiveBPETokenizer, EfficientBPETokenizer


class GPTTokenizer:
    """
    Wrapper around your BPE tokenizer for GPT-2 usage.

    Adds special tokens on top of the BPE vocab:
      PAD = vocab_size
      BOS = vocab_size + 1
      EOS = vocab_size + 2

    Usage:
        tokenizer = GPTTokenizer()
        tokenizer.train(corpus, vocab_size=10_000)
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
    """

    def __init__(self):
        self.bpe = None  # Set after train() or load()
        self._pad_id = None
        self._bos_id = None
        self._eos_id = None

    @property
    def vocab_size(self) -> int:
        """Total vocab including special tokens."""
        if self.bpe is None:
            return 0
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
        # your code here
        # 1. Instantiate and train your BPE tokenizer
        # 2. Set special token IDs (after BPE vocab)
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using trained BPE."""
        # your code here
        raise NotImplementedError

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text, filtering special tokens."""
        # your code here
        raise NotImplementedError
