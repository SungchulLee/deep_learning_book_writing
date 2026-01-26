# Patch Embeddings

## Introduction

Patch embedding is the foundational operation that bridges continuous images to discrete tokens, making images compatible with the transformer architecture. This operation converts a 2D image into a 1D sequence of vectors, analogous to how text is tokenized into word embeddings.

## The Tokenization Analogy

Just as natural language models tokenize text into words or subwords, Vision Transformers tokenize images into patches:

| NLP | Vision Transformer |
|-----|-------------------|
| Sentence | Image |
| Word/Token | Patch |
| Word embedding | Patch embedding |
| Vocabulary size | Patch content space |

## Mathematical Formulation

### Patch Extraction

Given an input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ with height $H$, width $W$, and $C$ channels, we divide it into non-overlapping patches of size $P \times P$.

The number of patches is:

$$N = \frac{H \times W}{P^2} = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)$$

For a standard 224×224 image with 16×16 patches:

$$N = \frac{224 \times 224}{16 \times 16} = 14 \times 14 = 196 \text{ patches}$$

### Flattening

Each patch is flattened into a vector:

$$\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}$$

For RGB images ($C=3$) with $P=16$:

$$\text{Patch dimension} = 16 \times 16 \times 3 = 768$$

### Linear Projection

Flattened patches are projected to the embedding dimension $D$ using a learnable matrix:

$$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$$

$$\mathbf{z}^i = \mathbf{x}_p^i \cdot \mathbf{E}$$

## Implementation as Convolution

A key insight is that patch embedding can be efficiently implemented as a single convolution with kernel size and stride equal to the patch size:

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Converts an image into patches and projects them into embeddings.
    This is the bridge between CNN-style input and transformer processing.
    
    Equivalent operations:
    1. Split image into P×P patches
    2. Flatten each patch
    3. Linear projection to embed_dim
    
    All accomplished with a single Conv2d!
    """
    def __init__(self, 
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_channels: int = 3, 
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Single convolution performs patch extraction and projection
        # kernel_size=patch_size extracts one patch
        # stride=patch_size ensures non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)
        Returns:
            Patch embeddings (B, N, D)
        """
        # Convolution output: (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, N)
        x = x.flatten(2)
        
        # Transpose to sequence format: (B, N, embed_dim)
        x = x.transpose(1, 2)
        
        return x
```

## Why Convolution Works

The equivalence between patch embedding and convolution can be understood as follows:

**Manual Patch Embedding:**
```python
# Step 1: Extract patches manually
patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
# Shape: (B, C, n_h, n_w, P, P)

# Step 2: Flatten patches
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, N, -1)
# Shape: (B, N, P*P*C)

# Step 3: Linear projection
embeddings = patches @ projection_matrix
# Shape: (B, N, D)
```

**Convolution Approach:**
```python
# All three steps in one operation!
embeddings = conv2d(x).flatten(2).transpose(1, 2)
# Shape: (B, N, D)
```

The convolution kernel weights encode the projection matrix, and the stride ensures non-overlapping patches.

## Visualization

```python
def visualize_patch_embedding(image: torch.Tensor, patch_size: int = 16):
    """
    Visualize how an image is divided into patches.
    """
    import matplotlib.pyplot as plt
    
    img = image[0].permute(1, 2, 0).cpu().numpy()
    H, W = img.shape[:2]
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    
    # Image with patch grid
    axes[1].imshow(img)
    for i in range(n_patches_h + 1):
        axes[1].axhline(y=i * patch_size, color='red', linewidth=2)
    for j in range(n_patches_w + 1):
        axes[1].axvline(x=j * patch_size, color='red', linewidth=2)
    
    axes[1].set_title(f"Patches: {n_patches_h}×{n_patches_w} = {n_patches_h * n_patches_w}")
    
    plt.tight_layout()
    plt.show()
```

## Patch Size Analysis

The choice of patch size involves important trade-offs:

### Smaller Patches (e.g., 8×8)

**Advantages:**
- Higher resolution representation
- More fine-grained features
- Better for detailed images

**Disadvantages:**
- More tokens → quadratic attention cost
- Higher computational requirements

For 224×224 with 8×8 patches: $N = 784$ tokens

### Larger Patches (e.g., 32×32)

**Advantages:**
- Fewer tokens → faster computation
- Lower memory requirements

**Disadvantages:**
- Loss of fine-grained details
- May miss small objects

For 224×224 with 32×32 patches: $N = 49$ tokens

### Standard Choice: 16×16

The standard 16×16 patch size balances resolution and efficiency:
- 196 tokens for 224×224 images
- Each patch captures meaningful local structure
- Manageable computational cost

## Computational Complexity

The patch embedding operation has complexity:

$$O(N \cdot P^2 \cdot C \cdot D) = O\left(\frac{HW}{P^2} \cdot P^2 \cdot C \cdot D\right) = O(HW \cdot C \cdot D)$$

This is linear in image size, making it efficient for large images.

## Overlapping Patches

While standard ViT uses non-overlapping patches, overlapping patches can be used:

```python
class OverlappingPatchEmbed(nn.Module):
    """Patch embedding with overlap for richer features."""
    def __init__(self, img_size=224, patch_size=16, stride=12, 
                 in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,  # stride < patch_size creates overlap
            padding=patch_size // 2
        )
        
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

Overlapping patches provide:
- Smoother spatial transitions
- More robust features
- Used in some ViT variants (e.g., PVT)

## Connection to CNNs

Patch embedding can be viewed as a large-stride convolution, providing a conceptual bridge to CNNs:

| Aspect | CNN Conv Layer | Patch Embedding |
|--------|---------------|-----------------|
| Kernel size | 3×3 (typical) | 16×16 |
| Stride | 1-2 | 16 |
| Output | Spatial feature map | Sequence of tokens |
| Receptive field | Local | Single patch |

This connection suggests hybrid architectures that combine CNN feature extraction with transformer processing.

## Best Practices

1. **Image Size**: Use images divisible by patch size (e.g., 224, 256, 384)

2. **Initialization**: Initialize projection weights with truncated normal:
```python
nn.init.trunc_normal_(self.proj.weight, std=0.02)
```

3. **Normalization**: Consider adding LayerNorm after projection:
```python
self.norm = nn.LayerNorm(embed_dim)
```

4. **Interpolation**: For different image sizes at inference, interpolate positional embeddings

## Applications

Patch embeddings enable transformers to process various visual data:

- **Image Classification**: Standard ViT with CLS token
- **Object Detection**: Patch tokens represent spatial locations
- **Segmentation**: Dense prediction from patch tokens
- **Video**: Extend to 3D patches (temporal + spatial)

## Summary

Patch embedding is the crucial first step that makes images amenable to transformer processing. By treating an image as a sequence of patches, we unlock the power of self-attention for computer vision while maintaining the spatial structure of visual data.
