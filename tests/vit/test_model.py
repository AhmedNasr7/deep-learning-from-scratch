import torch
from vit.model import build_vit

def test_vit_forward():
    batch_size = 2
    img_size = 32
    num_classes = 10
    in_channels = 3
    
    # 1. Create a dummy batch of images
    # Shape: (Batch, Channels, Height, Width)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # 2. Build the ViT Tiny model (perfect for CIFAR-10 32x32)
    model = build_vit('vit_tiny', num_classes=num_classes, img_size=img_size)
    
    # Print a quick summary of the model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized 'vit_tiny' with {num_params:,} parameters.")
    
    try:
        # 3. Run the forward pass
        logits = model(x)
        print("✅ ViT Forward pass successful!")
        print(f"   Output shape: {logits.shape}")
        
        # 4. Check shape matches (Batch, num_classes)
        assert logits.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {logits.shape}"
        print("✅ Output shape is perfectly correct!")
        
    except Exception as e:
        print("❌ ViT Forward pass failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_vit_forward()
