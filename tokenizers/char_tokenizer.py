"""
Naive character-level tokenizer.

Builds a vocabulary from unique characters in the training text.
Useful as a simple baseline before BPE — no merging, no subword units,
just one token per character.
"""

from typing import Dict, List


class CharTokenizer:
    """Character-level tokenizer with special tokens."""

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0

    @property
    def pad_id(self) -> int:
        return self.char_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.char_to_id["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.char_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.char_to_id["<EOS>"]

    def train(self, text: str) -> "CharTokenizer":
        """
        Build vocabulary from all unique characters in text.

        Adds special tokens first, then each unique character.
        Sets self.char_to_id, self.id_to_char, self.vocab_size.

        Args:
            text: raw training text

        Returns:
            self (for chaining)
        """
        # your code here
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """
        Convert a string to a list of token IDs (one per character).

        Unknown characters → UNK id.

        Args:
            text: input string

        Returns:
            List[int] of token IDs
        """
        # your code here
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a string.

        Args:
            ids: list of token IDs

        Returns:
            decoded string
        """
        # your code here
        raise NotImplementedError
