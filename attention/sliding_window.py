"""
Sliding Window Attention (Sparse Attention).

Only attends to a fixed window of recent tokens (e.g., W=4096) 
instead of the full context. This changes the compute complexity 
from O(T^2) to O(T * W), allowing for massive context lengths.
"""

import torch
import torch.nn as nn

class SlidingWindowAttention(nn.Module):
    """
    Educational skeleton for Sliding Window Attention.
    """
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # TODO: Implement sliding window masking
        # 1. Compute standard attention scores
        # 2. Apply a banded mask where distance > window_size is masked out (set to -inf)
        # 3. (Advanced) Implement a custom kernel that doesn't compute the masked-out scores at all
        raise NotImplementedError("Sliding Window Attention not yet implemented.")
