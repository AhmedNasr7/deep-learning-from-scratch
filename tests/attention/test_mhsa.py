import torch
from attention.multi_head import MultiHeadAttention
from attention.mhsa import FusedMultiHeadSelfAttention

def test_mhsa():
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 4
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. Initialize our new Fused MHSA
    mhsa = FusedMultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
    
    try:
        # Run forward pass
        out = mhsa(x)
        print("✅ MHSA Forward pass successful!")
        print(f"   Output shape: {out.shape}")
        
        # Check shape matches input shape
        assert out.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"
        print("✅ Output shape is perfectly correct!")
        
    except Exception as e:
        print("❌ MHSA Forward pass failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_mhsa()
