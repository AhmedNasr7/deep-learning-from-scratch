"""
FlashAttention (Dao et al., 2022).

Computes exact attention but with hardware-aware memory optimization (tiling).
Avoids materializing the O(T^2) attention matrix in main memory (HBM).
"""

import torch
import torch.nn as nn

class FlashAttention(nn.Module):
    """
    Educational skeleton for FlashAttention.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # TODO: Implement SRAM-aware tiling logic
        # 1. Split Q, K, V into blocks
        # 2. Iteratively compute softmax statistics (m_i, l_i) purely in SRAM
        # 3. Accumulate output blocks without storing the full (T, T) matrix
        raise NotImplementedError("FlashAttention tiling logic not yet implemented.")
