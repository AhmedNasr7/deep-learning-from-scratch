"""
Naive BPE (Byte Pair Encoding) tokenizer.

Starts from byte-level tokens (256 base vocab), iteratively merges
the most frequent adjacent pair until reaching target vocab size.
O(n * m) per merge step where n = sequence length, m = number of merges.
"""

from typing import Dict, List, Tuple
from collections import Counter

Pair = Tuple[int, int]


class NaiveBPETokenizer:
    def __init__(self):
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge_rules: List[Pair] = []
        self.pair_to_token: Dict[Pair, int] = {}

    def train(self, text: str, target_vocab_size: int) -> "NaiveBPETokenizer":
        self._reset()

        tokens = list(text.encode("utf-8"))
        next_id = 256

        while len(self.vocab) < target_vocab_size:
            pair_counts = self._count_pairs(tokens)

            if not pair_counts:
                break

            best_pair = self._select_best_pair(pair_counts)

            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules.append(best_pair)
            self.pair_to_token[best_pair] = next_id

            tokens = self._replace_pair(tokens, best_pair, next_id)
            next_id += 1

        return self

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))

        for new_id, pair in enumerate(self.merge_rules, start=256):
            tokens = self._replace_pair(tokens, pair, new_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        raw = b"".join(self.vocab[token_id] for token_id in token_ids)
        return raw.decode("utf-8", errors="replace")

    def _reset(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merge_rules = []
        self.pair_to_token = {}

    @staticmethod
    def _count_pairs(tokens: List[int]) -> Counter:
        return Counter(zip(tokens, tokens[1:]))

    @staticmethod
    def _select_best_pair(pair_counts: Counter) -> Pair:
        best_count = max(pair_counts.values())

        # deterministic tie-breaking
        return min(
            pair for pair, count in pair_counts.items()
            if count == best_count
        )

    @staticmethod
    def _replace_pair(tokens: List[int], pair: Pair, new_id: int) -> List[int]:
        output = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                output.append(new_id)
                i += 2
            else:
                output.append(tokens[i])
                i += 1

        return output
