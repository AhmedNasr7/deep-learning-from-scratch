"""
Data structures for the efficient BPE tokenizer.

IndexedLinkedList — O(1) pair replacement without rescanning.
MaxPairHeap      — lazy max-heap for efficient best-pair lookup.
PairStats        — tracks pair counts and positions, wired to both above.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import heapq

Pair = Tuple[int, int]


@dataclass
class Node:
    token: int
    prev: Optional[int]
    next: Optional[int]
    alive: bool = True


class IndexedLinkedList:
    def __init__(self, tokens: List[int]):
        self.nodes: List[Node] = [
            Node(
                token=token,
                prev=i - 1 if i > 0 else None,
                next=i + 1 if i < len(tokens) - 1 else None,
            )
            for i, token in enumerate(tokens)
        ]

    def get_pair_at(self, left_idx: Optional[int]) -> Optional[Pair]:
        if left_idx is None:
            return None

        if not self.nodes[left_idx].alive:
            return None

        right_idx = self.nodes[left_idx].next

        if right_idx is None or not self.nodes[right_idx].alive:
            return None

        return self.nodes[left_idx].token, self.nodes[right_idx].token

    def replace_pair_at(self, left_idx: int, new_id: int) -> None:
        right_idx = self.nodes[left_idx].next

        if right_idx is None:
            return

        next_idx = self.nodes[right_idx].next

        self.nodes[left_idx].token = new_id
        self.nodes[right_idx].alive = False
        self.nodes[left_idx].next = next_idx

        if next_idx is not None:
            self.nodes[next_idx].prev = left_idx

    def iter_indices(self):
        if not self.nodes:
            return

        idx = 0
        while idx is not None:
            if self.nodes[idx].alive:
                yield idx
            idx = self.nodes[idx].next

    def current_tokens(self) -> List[int]:
        return [self.nodes[i].token for i in self.iter_indices()]


class MaxPairHeap:
    def __init__(self):
        self.heap: List[Tuple[int, Pair]] = []

    def push(self, pair: Pair, count: int) -> None:
        if count > 0:
            heapq.heappush(self.heap, (-count, pair))

    def pop_best_valid(self, counts: Counter) -> Optional[Tuple[Pair, int]]:
        while self.heap:
            neg_count, pair = heapq.heappop(self.heap)
            heap_count = -neg_count

            if counts.get(pair, 0) == heap_count:
                return pair, heap_count

        return None

    def clear(self) -> None:
        self.heap.clear()


class PairStats:
    def __init__(self, linked_list: IndexedLinkedList, heap: MaxPairHeap):
        self.ll = linked_list
        self.heap = heap
        self.counts: Counter[Pair] = Counter()
        self.positions: Dict[Pair, Set[int]] = defaultdict(set)

    def initialize(self) -> None:
        self.counts.clear()
        self.positions.clear()
        self.heap.clear()

        for idx in self.ll.iter_indices():
            self.add_at(idx)

    def add_at(self, left_idx: Optional[int]) -> None:
        pair = self.ll.get_pair_at(left_idx)

        if pair is None:
            return

        self.counts[pair] += 1
        self.positions[pair].add(left_idx)
        self.heap.push(pair, self.counts[pair])

    def remove_at(self, left_idx: Optional[int]) -> None:
        pair = self.ll.get_pair_at(left_idx)

        if pair is None:
            return

        if self.counts.get(pair, 0) <= 0:
            return

        self.counts[pair] -= 1
        self.positions[pair].discard(left_idx)

        if self.counts[pair] <= 0:
            del self.counts[pair]
            self.positions.pop(pair, None)
        else:
            self.heap.push(pair, self.counts[pair])

    def get_positions(self, pair: Pair) -> List[int]:
        return list(self.positions.get(pair, set()))

    def pop_best_pair(self) -> Optional[Tuple[Pair, int]]:
        return self.heap.pop_best_valid(self.counts)
