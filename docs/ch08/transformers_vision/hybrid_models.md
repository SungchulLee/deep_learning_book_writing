# Hybrid CNN-Transformer Architectures

## Introduction

Hybrid architectures combine the strengths of CNNs and Transformers, leveraging CNN's inductive biases for local feature extraction and Transformer's global modeling capabilities. These models often achieve the best of both worlds: data efficiency from CNNs and powerful representations from Transformers.

## Motivation for Hybrid Approaches

### CNN Strengths
- Strong inductive biases (locality, translation equivariance)
- Data-efficient learning
- Efficient local feature extraction
- Well-understood training dynamics

### Transformer Strengths
- Global receptive field
- Flexible attention patterns
- Excellent scalability
- Superior performance with large data

### The Hybrid Solution
Combine CNN stems for initial feature extraction with Transformer blocks for global reasoning.

## Architecture Patterns

### Pattern 1: CNN Stem + Transformer Body

```python
import torch
import torch.nn as nn

class HybridCNNViT(nn.Module):
    """
    Hybrid architecture: CNN feature extraction + Transformer reasoning.
    
    Pipeline:
    1. CNN stem extracts low-level features (edges, textures)
    2. Features reshaped into sequence
    3. Transformer models global relationships
    4. Classification from aggregated representation
    """
    def __init__(self, n_classes: int = 10, embed_dim: int = 384, depth: int = 6):
        super().__init__()
        
        # CNN stem for initial feature extraction
        self.cnn_stem = nn.Sequential(
            # 224 → 112
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 112 → 56
            
            # 56 → 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 56 → 28
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Calculate number of patches after CNN stem
        # 224 → 112 → 56 → 28, so 28×28 = 784 patches
        self.n_patches = 28 * 28
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, n_heads=6, mlp_ratio=4)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.cnn_stem(x)  # (B, embed_dim, H, W)
        
        # Reshape for transformer: (B, H*W, embed_dim)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
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
```

### Pattern 2: Interleaved CNN and Attention (CoAtNet style)

```python
class ConvAttentionBlock(nn.Module):
    """
    Block that combines convolution and attention.
    Uses depthwise convolution for local features and attention for global.
    """
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.):
        super().__init__()
        
        # Local branch: Depthwise separable convolution
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )
        
        # Global branch: Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        # Local branch
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        local_out = self.local_conv(x_2d)
        local_out = local_out.flatten(2).transpose(1, 2)
        
        # Global branch
        x_norm = self.norm1(x)
        global_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        # Combine local and global
        x = x + local_out + global_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
```

### Pattern 3: Multi-Scale Hybrid (PVT style)

```python
class MultiScaleHybrid(nn.Module):
    """
    Multi-scale hybrid with CNN downsampling and Transformer at each scale.
    Similar to Pyramid Vision Transformer (PVT).
    """
    def __init__(self, n_classes: int = 1000, dims: tuple = (64, 128, 256, 512),
                 depths: tuple = (2, 2, 6, 2), n_heads: tuple = (1, 2, 4, 8)):
        super().__init__()
        
        self.stages = nn.ModuleList()
        in_channels = 3
        
        for i, (dim, depth, heads) in enumerate(zip(dims, depths, n_heads)):
            stage = nn.Sequential(
                # CNN downsampling
                nn.Conv2d(in_channels, dim, kernel_size=3, 
                         stride=2 if i > 0 else 4, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                
                # Transformer blocks at this scale
                *[SpatialTransformerBlock(dim, heads) for _ in range(depth)]
            )
            self.stages.append(stage)
            in_channels = dim
            
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.mean(dim=1)
        
        return self.head(x)


class SpatialTransformerBlock(nn.Module):
    """Transformer block that maintains spatial dimensions."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, n_heads)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = ConvMLP(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

## Notable Hybrid Architectures

### LeViT (2021)

LeViT uses a convolutional stem and replaces softmax attention with a more efficient variant:

```python
class LeViTAttention(nn.Module):
    """
    LeViT-style attention with learned attention bias.
    More efficient than standard attention for small images.
    """
    def __init__(self, dim: int, n_heads: int, resolution: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Learned attention bias (replaces positional encoding)
        self.attention_bias = nn.Parameter(
            torch.zeros(n_heads, resolution ** 2, resolution ** 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention with learned bias
        attn = (q @ k.transpose(-2, -1)) + self.attention_bias
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
```

### MobileViT (2021)

Designed for mobile deployment, combines MobileNet blocks with transformers:

```python
class MobileViTBlock(nn.Module):
    """
    MobileViT block: Local CNN + Global Transformer.
    Efficient design for mobile devices.
    """
    def __init__(self, in_channels: int, dim: int, depth: int, 
                 kernel_size: int = 3, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        
        # Local representation
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 
                     padding=kernel_size // 2),
            nn.Conv2d(in_channels, dim, 1),
        )
        
        # Global representation (Transformer)
        self.global_rep = nn.Sequential(
            *[TransformerBlock(dim, n_heads=dim // 64) for _ in range(depth)]
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, in_channels, 1),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size,
                     padding=kernel_size // 2),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Local features
        local = self.local_rep(x)
        
        # Unfold into patches for global processing
        patches = unfold_to_patches(local, self.patch_size)
        
        # Global transformer
        global_out = self.global_rep(patches)
        
        # Fold back to spatial
        global_out = fold_from_patches(global_out, H, W, self.patch_size)
        
        # Fusion with residual
        fused = self.fusion(torch.cat([local, global_out], dim=1))
        
        return x + fused
```

### EfficientFormer (2022)

Optimized for fast inference while maintaining accuracy:

```python
class EfficientFormerBlock(nn.Module):
    """
    EfficientFormer block with optional attention.
    Uses pooling-based token mixing for efficiency.
    """
    def __init__(self, dim: int, use_attention: bool = False):
        super().__init__()
        
        if use_attention:
            self.token_mixer = EfficientAttention(dim)
        else:
            # Pool-based token mixing (more efficient)
            self.token_mixer = nn.Sequential(
                nn.AvgPool2d(3, 1, 1, count_include_pad=False),
            )
        
        self.mlp = ConvMLP(dim, expansion=4)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

## Design Principles

### 1. Start Local, End Global

Most successful hybrids use CNN in early stages and Transformer in later stages:

```
Early Stages: CNN (local features, high resolution)
    ↓
Middle Stages: Hybrid (transition)
    ↓
Late Stages: Transformer (global reasoning, low resolution)
```

### 2. Efficient Attention at High Resolution

For high-resolution feature maps, use efficient attention variants:

- Window attention (Swin)
- Linear attention (Performer)
- Pooled attention (PVT)
- Local attention only

### 3. Preserve Inductive Biases

Keep useful CNN biases:
- Translation equivariance through convolutions
- Local connectivity in early layers
- Hierarchical feature maps

## Comparison

| Model | Params | ImageNet Top-1 | Throughput |
|-------|--------|----------------|------------|
| ResNet-50 | 25M | 76.1% | Fast |
| ViT-S/16 | 22M | 79.8% | Medium |
| LeViT-256 | 19M | 81.6% | Fast |
| MobileViT-S | 5.6M | 78.4% | Fast (mobile) |
| EfficientFormer-L1 | 12M | 79.2% | Very Fast |

## Training Tips for Hybrids

```python
# Hybrid models often benefit from:
training_config = {
    # Optimizer
    'optimizer': 'AdamW',
    'lr': 1e-3,  # Higher than pure ViT
    'weight_decay': 0.05,
    
    # Warmup (less aggressive than pure ViT)
    'warmup_epochs': 5,
    
    # Augmentation (moderate)
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'rand_augment': 'rand-m7-n2',  # Less aggressive
    
    # Regularization
    'drop_path': 0.05,
    'label_smoothing': 0.1,
}
```

## Conclusion

Hybrid CNN-Transformer architectures offer a practical middle ground:

- Better data efficiency than pure ViT
- Better scalability than pure CNN
- Flexible design space
- State-of-the-art on many benchmarks

The key is thoughtful combination: use CNNs where their biases help, transformers where global modeling matters.

## References

1. Wu, H., et al. "CvT: Introducing Convolutions to Vision Transformers." ICCV 2021.
2. Dai, Z., et al. "CoAtNet: Marrying Convolution and Attention for All Data Sizes." NeurIPS 2021.
3. Graham, B., et al. "LeViT: a Vision Transformer in ConvNet's Clothing." ICCV 2021.
4. Mehta, S., et al. "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer." ICLR 2022.
