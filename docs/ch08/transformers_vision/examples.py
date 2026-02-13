"""
Examples and demos for Vision Transformer
Quick start code snippets for common use cases
"""

import torch
from vit_model import create_vit_tiny, create_vit_base, VisionTransformer
from cnn_vs_vit import SimpleCNN, HybridCNNViT
from visualizations import AttentionVisualizer, visualize_patch_embedding


def example_1_basic_inference():
    """
    Example 1: Basic inference with ViT
    """
    print("\n" + "="*60)
    print("Example 1: Basic Inference with Vision Transformer")
    print("="*60 + "\n")
    
    # Create model
    model = create_vit_tiny(n_classes=10)
    model.eval()
    
    # Create random input (batch_size=1, channels=3, height=224, width=224)
    image = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
    
    # Get predictions
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Top probability: {probabilities.max().item():.4f}")
    
    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")


def example_2_compare_models():
    """
    Example 2: Compare CNN, ViT, and Hybrid models
    """
    print("\n" + "="*60)
    print("Example 2: Comparing Different Architectures")
    print("="*60 + "\n")
    
    # Create models
    cnn = SimpleCNN(n_classes=10)
    vit = create_vit_tiny(n_classes=10)
    hybrid = HybridCNNViT(n_classes=10)
    
    # Input
    x = torch.randn(1, 3, 224, 224)
    
    # Compare
    models = {"CNN": cnn, "ViT": vit, "Hybrid": hybrid}
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:10} | Parameters: {n_params:>10,} | Output shape: {output.shape}")


def example_3_patch_visualization():
    """
    Example 3: Visualize patch embedding process
    """
    print("\n" + "="*60)
    print("Example 3: Visualizing Patch Embeddings")
    print("="*60 + "\n")
    
    from vit_model import PatchEmbedding
    
    # Create patch embedding layer
    patch_embed = PatchEmbedding(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    )
    
    # Random image
    image = torch.randn(1, 3, 224, 224)
    
    # Convert to patches
    patches = patch_embed(image)
    
    print(f"Original image shape: {image.shape}")
    print(f"  - Dimensions: (batch, channels, height, width)")
    print(f"  - Size: {image.numel()} values")
    
    print(f"\nPatches shape: {patches.shape}")
    print(f"  - Dimensions: (batch, n_patches, embed_dim)")
    print(f"  - Number of patches: {patch_embed.n_patches}")
    print(f"  - Each patch is: 16×16 pixels = 256 pixels")
    print(f"  - Projected to: {patches.shape[-1]} dimensions")
    
    print("\nThis shows how ViT bridges continuous images to discrete tokens!")


def example_4_attention_mechanism():
    """
    Example 4: Understanding self-attention
    """
    print("\n" + "="*60)
    print("Example 4: Self-Attention Mechanism")
    print("="*60 + "\n")
    
    from vit_model import MultiHeadAttention
    
    # Create attention module
    attention = MultiHeadAttention(embed_dim=384, n_heads=6)
    
    # Random sequence (batch=1, seq_len=197, embed_dim=384)
    # 197 = 196 patches + 1 CLS token
    x = torch.randn(1, 197, 384)
    
    # Apply attention
    output = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nNumber of attention heads: {attention.n_heads}")
    print(f"Dimension per head: {attention.head_dim}")
    
    print("\nKey insight:")
    print("• Each token attends to ALL other tokens simultaneously")
    print("• This is different from CNN which only sees local neighbors")
    print("• Enables global context from the first layer")


def example_5_transfer_learning():
    """
    Example 5: Transfer learning with ViT
    """
    print("\n" + "="*60)
    print("Example 5: Transfer Learning Setup")
    print("="*60 + "\n")
    
    # Load pretrained model (simulated)
    model = create_vit_base(n_classes=1000)  # ImageNet classes
    
    print("Step 1: Load pretrained model")
    print(f"  - Original classes: 1000 (ImageNet)")
    
    # Replace classification head for new task
    n_new_classes = 10
    model.head = torch.nn.Linear(model.head.in_features, n_new_classes)
    
    print(f"\nStep 2: Replace classification head")
    print(f"  - New classes: {n_new_classes}")
    
    # Freeze backbone (optional)
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False
    
    print(f"\nStep 3: Freeze backbone layers")
    print(f"  - Only train classification head")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\nResult:")
    print(f"  - Total parameters: {total:,}")
    print(f"  - Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  - Frozen parameters: {total-trainable:,} ({100*(total-trainable)/total:.1f}%)")


def example_6_model_variants():
    """
    Example 6: Different ViT model sizes
    """
    print("\n" + "="*60)
    print("Example 6: ViT Model Variants")
    print("="*60 + "\n")
    
    from vit_model import create_vit_tiny, create_vit_small, create_vit_base, create_vit_large
    
    models = {
        "ViT-Tiny": create_vit_tiny,
        "ViT-Small": create_vit_small,
        "ViT-Base": create_vit_base,
        "ViT-Large": create_vit_large,
    }
    
    print(f"{'Model':<15} {'Parameters':<15} {'Embed Dim':<12} {'Depth':<10} {'Heads'}")
    print("-" * 70)
    
    for name, create_fn in models.items():
        model = create_fn(n_classes=1000)
        n_params = sum(p.numel() for p in model.parameters())
        
        embed_dim = model.patch_embed.proj.out_channels
        depth = len(model.blocks)
        n_heads = model.blocks[0].attn.n_heads
        
        print(f"{name:<15} {n_params:>12,}   {embed_dim:<12} {depth:<10} {n_heads}")


def example_7_hybrid_architecture():
    """
    Example 7: Hybrid CNN-Transformer model
    """
    print("\n" + "="*60)
    print("Example 7: Hybrid CNN-Transformer Architecture")
    print("="*60 + "\n")
    
    model = HybridCNNViT(n_classes=10)
    
    print("Architecture Pipeline:")
    print("\n1. CNN Stem (ResNet-style)")
    print("   • Input: 224×224×3 image")
    print("   • Convolutions for local feature extraction")
    print("   • Output: 28×28×384 feature maps")
    
    print("\n2. Reshape for Transformer")
    print("   • Flatten spatial dimensions")
    print("   • 28×28 = 784 tokens")
    print("   • Each token: 384 dimensions")
    
    print("\n3. Transformer Encoder")
    print("   • Self-attention across all 784 tokens")
    print("   • Global reasoning on CNN features")
    
    print("\n4. Classification Head")
    print("   • Global average pooling")
    print("   • Linear layer to classes")
    
    print("\nAdvantages:")
    print("• Combines local CNN features with global Transformer reasoning")
    print("• More data-efficient than pure ViT")
    print("• Better inductive biases for vision tasks")


def example_8_key_differences():
    """
    Example 8: Key differences between CNN and ViT
    """
    print("\n" + "="*60)
    print("Example 8: CNN vs ViT - Key Differences")
    print("="*60 + "\n")
    
    print("1. INPUT PROCESSING")
    print("   CNN: Sliding window convolutions")
    print("   ViT: Divide into patches, linear projection")
    
    print("\n2. RECEPTIVE FIELD")
    print("   CNN: Grows gradually with depth")
    print("   ViT: Global from first layer")
    
    print("\n3. INDUCTIVE BIAS")
    print("   CNN: Strong (locality, translation equivariance)")
    print("   ViT: Weak (learns from data)")
    
    print("\n4. DATA REQUIREMENTS")
    print("   CNN: Works well with small datasets")
    print("   ViT: Needs large datasets (or pretraining)")
    
    print("\n5. COMPUTATIONAL COMPLEXITY")
    print("   CNN: O(k²·C·H·W) where k=kernel size")
    print("   ViT: O(N²·D) where N=number of patches")
    
    print("\n6. INTERPRETATION")
    print("   CNN: Activation maps, filter visualization")
    print("   ViT: Attention maps, token importance")


def run_all_examples():
    """Run all examples"""
    example_1_basic_inference()
    example_2_compare_models()
    example_3_patch_visualization()
    example_4_attention_mechanism()
    example_5_transfer_learning()
    example_6_model_variants()
    example_7_hybrid_architecture()
    example_8_key_differences()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_examples()
