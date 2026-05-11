"""
Efficient BPE tokenizer using indexed linked list + lazy max-heap.

Same algorithm as NaiveBPETokenizer but avoids rescanning the full
sequence on every merge. Training is O(n + m * k) where m = merges,
k = average positions per best pair.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .data_structures import IndexedLinkedList, MaxPairHeap, PairStats

Pair = Tuple[int, int]


class EfficientBPETokenizer:
    def __init__(self):
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge_rules: List[Pair] = []
        self.pair_to_token: Dict[Pair, int] = {}

    def train(self, text: str, target_vocab_size: int) -> "EfficientBPETokenizer":
        self._reset()

        tokens = list(text.encode("utf-8"))

        if target_vocab_size <= 256 or len(tokens) < 2:
            return self

        ll = IndexedLinkedList(tokens)
        heap = MaxPairHeap()
        stats = PairStats(ll, heap)
        stats.initialize()

        next_id = 256

        while len(self.vocab) < target_vocab_size:
            best = stats.pop_best_pair()

            if best is None:
                break

            best_pair, count = best

            new_id = next_id
            a, b = best_pair

            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merge_rules.append(best_pair)
            self.pair_to_token[best_pair] = new_id

            positions = stats.get_positions(best_pair)

            for left_idx in positions:
                if ll.get_pair_at(left_idx) != best_pair:
                    continue

                right_idx = ll.nodes[left_idx].next

                if right_idx is None:
                    continue

                prev_idx = ll.nodes[left_idx].prev
                next_idx = ll.nodes[right_idx].next

                # remove old local pairs
                stats.remove_at(prev_idx)
                stats.remove_at(left_idx)
                stats.remove_at(right_idx)

                # linked-list replacement
                ll.replace_pair_at(left_idx, new_id)

                # add new local pairs
                stats.add_at(prev_idx)
                stats.add_at(left_idx)

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
