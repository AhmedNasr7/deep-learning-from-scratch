import torch
import torch.nn as nn
from .scaled_dot_product import ScaledDotProductAttention

class FusedMultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) using a single fused QKV projection.
    
    Instead of maintaining 3 separate linear layers for Q, K, and V, it uses
    a single massive layer (d_model -> 3 * d_model) to maximize GPU parallelization.
    This is the standard for Encoder-only (ViT/BERT) and Decoder-only (GPT) models.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Fused QKV projection: projects to 3 * d_model in one go
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # your code here
        # 1. Project x all at once using self.qkv_proj
        # 3. Apply self.attention(q, k, v, mask)
        # 4. Recombine heads, apply out_proj, and apply dropout
        

        qkv = self.qkv_proj(x) # fused projection: more efficient
        B, seq, d_3 = qkv.shape
        qkv = qkv.reshape(B, seq, self.n_heads, -1).permute(0, 2, 1, 3) # (B, n_heads, seq, 3 * d_model)
        q, k, v = qkv.split(self.d_k, dim=-1)  # (B, n_heads, seq, d_model) each 

        attn = self.attention(q, k, v, mask)
        attn = attn.permute(0, 2, 1, 3).reshape(B, seq, self.d_model)
        output = self.dropout(self.out_proj(attn))

        return output
        





        


