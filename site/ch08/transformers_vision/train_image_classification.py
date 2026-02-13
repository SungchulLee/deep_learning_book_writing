"""
Train Vision Transformer on Image Classification
"""
import torch
from vision_transformer import VisionTransformer

def train_vit():
    model = VisionTransformer(img_size=224, patch_size=16, num_classes=10)
    
    # Dummy data
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    
    print(f"ViT Model created!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nReady for image classification training!")

if __name__ == '__main__':
    train_vit()
