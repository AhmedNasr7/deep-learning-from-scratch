"""
Grouped-Query Attention (GQA) & Multi-Query Attention (MQA).

Instead of n_heads for Q, K, and V:
- MQA uses 1 shared K and 1 shared V head.
- GQA groups Q heads (e.g., 8 Q heads share 1 K and 1 V head).
Reduces KV cache memory footprint drastically during LLM text generation.
"""

import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_query_heads: int, n_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        
        assert n_query_heads % n_kv_heads == 0, "Query heads must be a multiple of KV heads"
        
        # TODO: Initialize projections for Q, K, V
        
    def forward(self, x: torch.Tensor, kv_cache: dict = None) -> torch.Tensor:
        # TODO: Implement GQA logic
        # 1. Project Q, K, V
        # 2. Repeat/broadcast K and V heads to match Q heads
        # 3. Apply standard attention (or FlashAttention)
        raise NotImplementedError("Grouped-Query Attention not yet implemented.")
