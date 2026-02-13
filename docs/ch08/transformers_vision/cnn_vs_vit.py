"""
CNN vs Vision Transformer Comparison
Demonstrates the key differences and bridges between architectures
"""

import torch
import torch.nn as nn
from typing import Dict, List
import time


class SimpleCNN(nn.Module):
    """
    Traditional CNN architecture for comparison.
    Uses convolutional layers with spatial hierarchy.
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class HybridCNNViT(nn.Module):
    """
    Hybrid architecture combining CNN and Transformer.
    Uses CNN for initial feature extraction, then transformer for global modeling.
    This represents the bridge between the two paradigms.
    """
    def __init__(self, n_classes: int = 10, embed_dim: int = 384, depth: int = 6):
        super().__init__()
        
        # CNN stem for feature extraction
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Flatten CNN features into sequence
        # After stem: 224 -> 56 -> 28 -> 28, so 28x28 = 784 patches
        self.flatten = nn.Flatten(2)  # Keep channel, flatten spatial
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 784, embed_dim))
        
        # Transformer blocks
        from vit_model import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, n_heads=6, mlp_ratio=4)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.cnn_stem(x)  # (B, embed_dim, H, W)
        
        # Reshape for transformer
        B, C, H, W = x.shape
        x = self.flatten(x)  # (B, C, H*W)
        x = x.transpose(1, 2)  # (B, H*W, C)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer processing
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling and classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        
        return x


def compare_architectures():
    """
    Compare CNN and ViT architectures across different dimensions.
    """
    from vit_model import create_vit_small
    
    print("\n" + "="*80)
    print("CNN vs Vision Transformer: Architectural Comparison")
    print("="*80 + "\n")
    
    # Create models
    cnn = SimpleCNN(n_classes=10)
    vit = create_vit_small(n_classes=10)
    hybrid = HybridCNNViT(n_classes=10)
    
    # Count parameters
    cnn_params = sum(p.numel() for p in cnn.parameters())
    vit_params = sum(p.numel() for p in vit.parameters())
    hybrid_params = sum(p.numel() for p in hybrid.parameters())
    
    print("1. MODEL SIZE")
    print(f"   CNN:        {cnn_params:>12,} parameters")
    print(f"   ViT:        {vit_params:>12,} parameters")
    print(f"   Hybrid:     {hybrid_params:>12,} parameters")
    
    # Compare receptive fields
    print("\n2. RECEPTIVE FIELD")
    print("   CNN:        Local, hierarchical (grows with depth)")
    print("   ViT:        Global from first layer (self-attention)")
    print("   Hybrid:     Local (CNN) → Global (Transformer)")
    
    # Inductive biases
    print("\n3. INDUCTIVE BIASES")
    print("   CNN:        Strong (translation equivariance, locality)")
    print("   ViT:        Weak (learns from data, needs more samples)")
    print("   Hybrid:     Medium (combines both approaches)")
    
    # Inference speed test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = cnn.to(device)
    vit = vit.to(device)
    hybrid = hybrid.to(device)
    
    x = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = cnn(x)
        _ = vit(x)
        _ = hybrid(x)
    
    # Time CNN
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = cnn(x)
    torch.cuda.synchronize() if device == "cuda" else None
    cnn_time = (time.time() - start) / 100
    
    # Time ViT
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = vit(x)
    torch.cuda.synchronize() if device == "cuda" else None
    vit_time = (time.time() - start) / 100
    
    # Time Hybrid
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = hybrid(x)
    torch.cuda.synchronize() if device == "cuda" else None
    hybrid_time = (time.time() - start) / 100
    
    print("\n4. INFERENCE SPEED (single image)")
    print(f"   CNN:        {cnn_time*1000:.2f} ms")
    print(f"   ViT:        {vit_time*1000:.2f} ms")
    print(f"   Hybrid:     {hybrid_time*1000:.2f} ms")
    
    # Key differences
    print("\n5. KEY ARCHITECTURAL DIFFERENCES")
    print("\n   CNN Approach:")
    print("   • Uses convolution for local feature extraction")
    print("   • Builds spatial hierarchy through pooling")
    print("   • Strong inductive bias (locality, translation equivariance)")
    print("   • Efficient on small datasets")
    
    print("\n   ViT Approach:")
    print("   • Treats image as sequence of patches")
    print("   • Self-attention for global feature extraction")
    print("   • Minimal inductive bias (learns from data)")
    print("   • Requires large datasets to excel")
    
    print("\n   Hybrid Approach:")
    print("   • CNN for low-level features")
    print("   • Transformer for high-level global reasoning")
    print("   • Balances efficiency and performance")
    print("   • Best of both worlds")
    
    print("\n6. WHEN TO USE EACH")
    print("\n   Use CNN when:")
    print("   • Dataset is small (<10k images)")
    print("   • Need fast inference")
    print("   • Spatial locality is important")
    
    print("\n   Use ViT when:")
    print("   • Large dataset available (>1M images)")
    print("   • Need to capture long-range dependencies")
    print("   • Have compute resources for training")
    
    print("\n   Use Hybrid when:")
    print("   • Want benefits of both architectures")
    print("   • Medium-sized dataset")
    print("   • Balance between efficiency and performance")
    
    print("\n" + "="*80 + "\n")


def analyze_attention_vs_convolution():
    """
    Deep dive into self-attention vs convolution mechanisms.
    """
    print("\n" + "="*80)
    print("Self-Attention vs Convolution: Mechanism Comparison")
    print("="*80 + "\n")
    
    print("CONVOLUTION:")
    print("• Fixed receptive field determined by kernel size")
    print("• Local connectivity - only sees nearby pixels")
    print("• Weight sharing across spatial positions")
    print("• Translation equivariant")
    print("• Computationally efficient: O(k²·C·H·W)")
    print("  where k=kernel size, C=channels, H=height, W=width")
    
    print("\nSELF-ATTENTION:")
    print("• Dynamic receptive field - can attend to entire image")
    print("• Global connectivity - sees all patches simultaneously")
    print("• Position-dependent weights (through positional encoding)")
    print("• Permutation invariant (without positional encoding)")
    print("• Computationally expensive: O(N²·D)")
    print("  where N=number of patches, D=embedding dimension")
    
    print("\nBRIDGING CONCEPTS:")
    print("• Both are learnable feature extractors")
    print("• Convolution = local attention with fixed pattern")
    print("• Attention = dynamic convolution with learned patterns")
    print("• Patch embedding in ViT ≈ large stride convolution")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    compare_architectures()
    analyze_attention_vs_convolution()
