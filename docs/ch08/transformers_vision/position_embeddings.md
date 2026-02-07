# Position Embeddings for Images

## Introduction

Self-attention in transformers is permutation-invariant—it produces the same output regardless of the order of input tokens. While this property is sometimes desirable, for images we need to preserve spatial information. Position embeddings encode where each patch is located in the original image, enabling the model to understand spatial relationships.

## Why Position Information Matters

Consider two scenarios:
1. A cat in the top-left corner with a dog in the bottom-right
2. A dog in the top-left corner with a cat in the bottom-right

Without position embeddings, self-attention would treat these identically since it only considers pairwise relationships between tokens, not their positions. Position embeddings break this symmetry.

## Mathematical Framework

### Learnable Position Embeddings

The standard ViT uses learnable position embeddings:

$$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

where $N$ is the number of patches and $D$ is the embedding dimension. The "+1" accounts for the CLS token.

The embeddings are added to patch embeddings:

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{z}_0^1; \ldots; \mathbf{z}_0^N] + \mathbf{E}_{pos}$$

### Sinusoidal Position Embeddings

Alternatively, fixed sinusoidal embeddings (as in the original Transformer) can be used:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/D}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$

For 2D images, this extends to:

$$PE_{(x, y)} = [PE_x; PE_y]$$

## Implementation

### Learnable 1D Position Embeddings

```python
import torch
import torch.nn as nn

class LearnablePositionEmbedding(nn.Module):
    """
    Standard learnable position embeddings as used in ViT.
    
    Each position gets a unique learnable vector that is 
    added to the patch embedding at that position.
    """
    def __init__(self, n_patches: int, embed_dim: int, include_cls: bool = True):
        super().__init__()
        n_positions = n_patches + 1 if include_cls else n_patches
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_positions, embed_dim))
        
        # Initialize with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to input."""
        return x + self.pos_embed
```

### 2D Sinusoidal Position Embeddings

```python
import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbedding2D(nn.Module):
    """
    2D sinusoidal position embeddings.
    
    Extends the 1D sinusoidal approach to 2D by concatenating
    separate embeddings for x and y coordinates.
    """
    def __init__(self, embed_dim: int, height: int, width: int, 
                 temperature: float = 10000.0):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        
        self.embed_dim = embed_dim
        pe = self._make_2d_sincos(height, width, embed_dim, temperature)
        self.register_buffer('pe', pe)
        
    def _make_2d_sincos(self, h: int, w: int, d: int, temp: float) -> torch.Tensor:
        """Generate 2D sinusoidal position embeddings."""
        # Create coordinate grids
        y_pos = torch.arange(h).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w).unsqueeze(0).repeat(h, 1)
        
        # Dimension indices
        dim_t = torch.arange(d // 4)
        omega = 1.0 / (temp ** (dim_t / (d // 4)))
        
        # Compute embeddings
        # Each coordinate contributes d/4 sin and d/4 cos values
        y_embed = y_pos.flatten().unsqueeze(1) * omega.unsqueeze(0)
        x_embed = x_pos.flatten().unsqueeze(1) * omega.unsqueeze(0)
        
        pe = torch.cat([
            torch.sin(y_embed),
            torch.cos(y_embed),
            torch.sin(x_embed),
            torch.cos(x_embed)
        ], dim=1)
        
        return pe.unsqueeze(0)  # (1, h*w, d)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to input."""
        return x + self.pe[:, :x.size(1)]
```

### Relative Position Embeddings

```python
class RelativePositionBias(nn.Module):
    """
    Relative position bias as used in Swin Transformer.
    
    Instead of absolute positions, encodes relative distances
    between patch positions.
    """
    def __init__(self, window_size: int, n_heads: int):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        
        # Relative position bias table
        # (2*window-1) possible relative positions in each dimension
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, n_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute pairwise relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self) -> torch.Tensor:
        """Return relative position bias for attention."""
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size ** 2,
            self.window_size ** 2,
            -1
        )
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, N, N)
```

## Position Embedding Interpolation

When using a model trained on one image size for a different size, position embeddings must be interpolated:

```python
def interpolate_pos_embed(model, new_size: int, old_size: int = 224):
    """
    Interpolate position embeddings for different image sizes.
    
    Args:
        model: ViT model with pos_embed parameter
        new_size: New image size
        old_size: Original image size (default 224)
    """
    pos_embed = model.pos_embed
    n_patches_new = (new_size // model.patch_embed.patch_size) ** 2
    n_patches_old = (old_size // model.patch_embed.patch_size) ** 2
    
    if n_patches_new == n_patches_old:
        return pos_embed
    
    # Separate CLS token and patch embeddings
    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    
    # Reshape to 2D grid
    dim = patch_pos.shape[-1]
    h_old = w_old = int(n_patches_old ** 0.5)
    h_new = w_new = int(n_patches_new ** 0.5)
    
    patch_pos = patch_pos.reshape(1, h_old, w_old, dim).permute(0, 3, 1, 2)
    
    # Interpolate
    patch_pos = nn.functional.interpolate(
        patch_pos, 
        size=(h_new, w_new),
        mode='bicubic',
        align_corners=False
    )
    
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
    
    return torch.cat([cls_pos, patch_pos], dim=1)
```

## Visualization and Analysis

### Visualizing Position Embeddings

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_position_embeddings(pos_embed: torch.Tensor, grid_size: int = 14):
    """
    Visualize learned position embeddings.
    
    Shows:
    1. Position embedding vectors as heatmap
    2. Similarity matrix between positions
    """
    # Remove CLS token
    pos_embed = pos_embed[0, 1:].detach().cpu()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of embeddings
    im1 = axes[0].imshow(pos_embed.T, aspect='auto', cmap='coolwarm')
    axes[0].set_xlabel('Patch Position')
    axes[0].set_ylabel('Embedding Dimension')
    axes[0].set_title('Position Embedding Vectors')
    plt.colorbar(im1, ax=axes[0])
    
    # Similarity matrix
    pos_norm = pos_embed / pos_embed.norm(dim=1, keepdim=True)
    similarity = (pos_norm @ pos_norm.T).numpy()
    
    im2 = axes[1].imshow(similarity, cmap='viridis')
    axes[1].set_xlabel('Patch Position')
    axes[1].set_ylabel('Patch Position')
    axes[1].set_title('Position Similarity Matrix')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def visualize_spatial_similarity(pos_embed: torch.Tensor, grid_size: int = 14):
    """
    Show how each position relates to others spatially.
    
    For each reference position, shows similarity to all other positions.
    """
    pos_embed = pos_embed[0, 1:].detach().cpu()
    pos_norm = pos_embed / pos_embed.norm(dim=1, keepdim=True)
    similarity = (pos_norm @ pos_norm.T).numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reference positions to visualize
    refs = [
        (0, 0),                          # Top-left
        (0, grid_size // 2),             # Top-center
        (grid_size // 2, grid_size // 2),# Center
        (grid_size // 2, 0),             # Left-center
        (grid_size - 1, grid_size - 1),  # Bottom-right
        (grid_size // 4, grid_size // 4) # Quarter position
    ]
    
    for ax, (row, col) in zip(axes.flatten(), refs):
        idx = row * grid_size + col
        sim_map = similarity[idx].reshape(grid_size, grid_size)
        
        im = ax.imshow(sim_map, cmap='viridis')
        ax.scatter([col], [row], c='red', s=100, marker='x')
        ax.set_title(f'Reference: ({row}, {col})')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Position Embedding Similarities')
    plt.tight_layout()
    plt.show()
```

## Comparison of Approaches

| Method | Learnable | Extrapolation | Advantages | Disadvantages |
|--------|-----------|---------------|------------|---------------|
| Learnable 1D | Yes | Poor | Simple, effective | Fixed sequence length |
| Sinusoidal 1D | No | Good | Extrapolates well | May be suboptimal |
| Learnable 2D | Yes | Moderate | Preserves 2D structure | More parameters |
| Sinusoidal 2D | No | Good | Natural for images | Fixed patterns |
| Relative | Yes | Excellent | Shift-invariant | More complex |
| RoPE | Partially | Excellent | Good extrapolation | Implementation complexity |

## Empirical Findings

Research has revealed several insights about position embeddings in ViT:

1. **Learned vs. Fixed**: Learned embeddings slightly outperform fixed sinusoidal ones
2. **1D vs. 2D**: 2D-aware embeddings provide marginal improvements
3. **Spatial Structure**: Learned embeddings naturally develop 2D spatial awareness
4. **Interpolation**: Bicubic interpolation works well for different resolutions

## Best Practices

1. **Standard Choice**: Use learnable 1D position embeddings with proper initialization
2. **Multi-Resolution**: Implement position embedding interpolation for flexibility
3. **Initialization**: Use truncated normal with std=0.02
4. **Dropout**: Apply dropout after adding position embeddings

```python
class PositionEmbedding(nn.Module):
    """Complete position embedding module with best practices."""
    def __init__(self, n_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed[:, :x.size(1)]
        return self.dropout(x)
```

## Summary

Position embeddings are essential for enabling Vision Transformers to understand spatial relationships. While various approaches exist, learnable 1D position embeddings remain the standard choice due to their simplicity and effectiveness. Understanding position embeddings is crucial for adapting ViT to different image sizes and for interpreting what the model has learned about image structure.
