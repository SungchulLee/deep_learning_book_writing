# Swin Transformer

## Introduction

The Swin Transformer (Shifted Window Transformer), introduced by Liu et al. (2021), addresses ViT's computational limitations by introducing hierarchical feature maps and local window attention. This design enables linear complexity with respect to image size while maintaining the power of self-attention.

## Motivation

Standard ViT has two main limitations for dense prediction tasks:

1. **Quadratic Complexity**: Self-attention over all patches has $O(N^2)$ complexity
2. **Single-Scale Features**: Produces tokens at a single resolution

These limitations make ViT challenging to use for tasks like object detection and segmentation that require:
- Multi-scale feature maps
- High-resolution outputs
- Efficient processing of large images

Swin Transformer addresses both issues through hierarchical architecture and windowed attention.

## Key Innovations

### 1. Hierarchical Feature Maps

Swin Transformer progressively reduces spatial resolution while increasing channel dimension, similar to CNNs:

```
Stage 1: H/4 × W/4 × C
    ↓ (Patch Merging)
Stage 2: H/8 × W/8 × 2C
    ↓ (Patch Merging)
Stage 3: H/16 × W/16 × 4C
    ↓ (Patch Merging)
Stage 4: H/32 × W/32 × 8C
```

### 2. Window-based Multi-head Self-Attention (W-MSA)

Instead of global attention, compute attention within local windows:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention.
    
    Computes attention within local windows, reducing complexity
    from O(N²) to O(N × M²) where M is window size.
    """
    def __init__(self, dim: int, window_size: int, n_heads: int, 
                 qkv_bias: bool = True, attn_drop: float = 0., 
                 proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, n_heads)
        )
        
        # Create relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.flatten(1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows * B, window_size * window_size, C)
            mask: Attention mask for shifted windows
        """
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask for shifted window attention
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
```

### 3. Shifted Window Partitioning

To enable cross-window connections, alternate between regular and shifted window partitioning:

```python
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition image into non-overlapping windows.
    
    Args:
        x: (B, H, W, C)
        window_size: Window size
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, 
                   H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: Window size
        H, W: Original height and width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, 
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

class ShiftedWindowAttention(nn.Module):
    """Shifted window attention with efficient cyclic shift."""
    
    def __init__(self, dim: int, window_size: int, shift_size: int, 
                 n_heads: int, input_resolution: tuple):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        
        self.attn = WindowAttention(dim, window_size, n_heads)
        
        # Create attention mask for shifted windows
        if shift_size > 0:
            H, W = input_resolution
            mask = self._create_mask(H, W)
            self.register_buffer("attn_mask", mask)
        else:
            self.attn_mask = None
            
    def _create_mask(self, H: int, W: int) -> torch.Tensor:
        """Create attention mask for shifted window attention."""
        img_mask = torch.zeros((1, H, W, 1))
        
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), 
                                   dims=(1, 2))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), 
                          dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        return x
```

## Complete Swin Transformer Block

```python
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with W-MSA and SW-MSA.
    
    Alternates between regular window attention (W-MSA) 
    and shifted window attention (SW-MSA).
    """
    def __init__(self, dim: int, input_resolution: tuple, n_heads: int,
                 window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop: float = 0., attn_drop: float = 0., 
                 drop_path: float = 0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttention(
            dim, window_size, shift_size, n_heads, input_resolution
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop=drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Window attention
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

## Patch Merging

Reduces spatial resolution and increases channels:

```python
class PatchMerging(nn.Module):
    """
    Patch merging layer for hierarchical feature maps.
    
    Reduces spatial resolution by 2x and increases channels by 2x.
    Similar to strided convolution in CNNs.
    """
    def __init__(self, input_resolution: tuple, dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)
        
        # Take every other row and column
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2 * W/2, 2C)
        
        return x
```

## Complete Swin Transformer

```python
class SwinTransformer(nn.Module):
    """
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
    
    Key features:
    1. Hierarchical feature maps (like CNNs)
    2. Linear complexity O(N) via windowed attention
    3. Cross-window connections via shifted windows
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 96,
                 depths: tuple = (2, 2, 6, 2),
                 n_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7,
                 mlp_ratio: float = 4.,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.1):
        super().__init__()
        
        self.n_layers = len(depths)
        self.embed_dim = embed_dim
        self.n_features = int(embed_dim * 2 ** (self.n_layers - 1))
        
        # Patch embedding (smaller patches than ViT)
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim
        )
        patches_resolution = self.patch_embed.patches_resolution
        
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.n_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                n_heads=n_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if i_layer < self.n_layers - 1 else None
            )
            self.layers.append(layer)
            
        self.norm = nn.LayerNorm(self.n_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.n_features, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.flatten(1)
        x = self.head(x)
        
        return x
```

## Computational Complexity

### Standard ViT (Global Attention)
$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

### Swin Transformer (Window Attention)
$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

where $M$ is the window size (typically 7). For a 224×224 image with 56×56 patches:
- ViT: $O(56^2 \times 56^2) = O(9.8M)$
- Swin: $O(56^2 \times 7^2) = O(153K)$

**64× reduction in attention complexity!**

## Model Variants

| Model | Params | C | Depths | Heads | ImageNet Top-1 |
|-------|--------|---|--------|-------|----------------|
| Swin-T | 29M | 96 | (2,2,6,2) | (3,6,12,24) | 81.3% |
| Swin-S | 50M | 96 | (2,2,18,2) | (3,6,12,24) | 83.0% |
| Swin-B | 88M | 128 | (2,2,18,2) | (4,8,16,32) | 83.5% |
| Swin-L | 197M | 192 | (2,2,18,2) | (6,12,24,48) | 87.3%* |

*Pretrained on ImageNet-22K

## Applications

Swin Transformer excels at dense prediction tasks:

### Object Detection
```python
# Swin backbone for detection
backbone = SwinTransformer(
    img_size=800,  # Larger images
    patch_size=4,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    n_heads=(4, 8, 16, 32)
)
# Output: Multi-scale features at 1/4, 1/8, 1/16, 1/32
```

### Semantic Segmentation
```python
# Swin-UNet style architecture
class SwinUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = SwinTransformer(...)
        self.decoder = UNetDecoder(...)
```

## Swin V2 Improvements

Swin Transformer V2 introduces:

1. **Residual post-normalization**: Better training stability
2. **Scaled cosine attention**: Better handling of different resolutions
3. **Log-spaced continuous relative position bias**: Better generalization

## Financial Applications

Swin Transformer's efficiency makes it suitable for:

- **Document Analysis**: Processing high-resolution financial documents
- **Satellite Imagery**: Economic activity from aerial images
- **Multi-scale Charts**: Analyzing financial charts at multiple resolutions
- **Video Analysis**: Financial news and conference video analysis

## Summary

Swin Transformer successfully combines:
- **Transformer's modeling power**: Self-attention for rich representations
- **CNN's efficiency**: Hierarchical features with linear complexity
- **Flexibility**: Suitable for both classification and dense prediction

This makes it a versatile backbone for modern computer vision systems.

## References

1. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
2. Liu, Z., et al. "Swin Transformer V2: Scaling Up Capacity and Resolution." CVPR 2022.
