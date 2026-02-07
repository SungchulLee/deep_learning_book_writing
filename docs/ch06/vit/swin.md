# Swin Transformer

## Introduction

The **Swin Transformer** (Shifted Windows Transformer), introduced by Liu et al. (2021), addresses the two primary computational limitations of standard ViTs: **quadratic complexity** in self-attention and the **lack of multi-scale feature hierarchy**. Swin achieves this through two key innovations: computing self-attention within local **windows** rather than globally, and **shifting** these windows between layers to enable cross-window information flow. The result is a hierarchical vision transformer with linear computational complexity in image size—making it practical as a general-purpose backbone for dense prediction tasks like object detection and segmentation, not just classification.

---

## Motivation: Limitations of Standard ViT

### Computational Cost

Standard ViT computes global self-attention over all $N$ patches, with cost $O(N^2)$. For a 224×224 image with 16×16 patches, $N = 196$ and the cost is manageable. But for dense prediction tasks at higher resolutions:

| Resolution | Patch Size | $N$ | Attention Cost ($N^2$) |
|---|---|---|---|
| 224 × 224 | 16 | 196 | 38K |
| 512 × 512 | 16 | 1,024 | 1.05M |
| 1024 × 1024 | 16 | 4,096 | 16.8M |
| 2048 × 2048 | 16 | 16,384 | 268M |

The quadratic growth makes standard ViT impractical for high-resolution inputs.

### Lack of Hierarchy

CNNs naturally produce **multi-scale feature maps** through progressive downsampling (stride-2 convolutions, pooling). These hierarchical features are essential for dense prediction tasks:

```
CNN feature hierarchy:         ViT (flat):
Stage 1: H/4  × W/4  × C₁    All layers: H/16 × W/16 × d
Stage 2: H/8  × W/8  × C₂    (single resolution throughout)
Stage 3: H/16 × W/16 × C₃
Stage 4: H/32 × W/32 × C₄
```

Swin Transformer addresses both limitations simultaneously.

---

## Architecture Overview

### Hierarchical Design

Swin processes images through four stages, progressively reducing spatial resolution while increasing channel dimension—mirroring the CNN paradigm:

```
Input Image (H × W × 3)
        │
        ▼
┌──────────────────┐
│  Patch Partition  │  4×4 patches → H/4 × W/4 × 48
│  + Linear Embed   │  → H/4 × W/4 × C
└──────────────────┘
        │
        ▼
┌──────────────────┐
│    Stage 1        │  Swin Blocks × 2
│  H/4 × W/4 × C   │  Window Attention + Shifted Window Attention
└──────────────────┘
        │ Patch Merging (2× downsample)
        ▼
┌──────────────────┐
│    Stage 2        │  Swin Blocks × 2
│  H/8 × W/8 × 2C  │
└──────────────────┘
        │ Patch Merging
        ▼
┌──────────────────┐
│    Stage 3        │  Swin Blocks × 6
│  H/16 × W/16 × 4C│
└──────────────────┘
        │ Patch Merging
        ▼
┌──────────────────┐
│    Stage 4        │  Swin Blocks × 2
│  H/32 × W/32 × 8C│
└──────────────────┘
```

### Patch Merging

Between stages, **patch merging** reduces spatial resolution by 2× while doubling channels—analogous to stride-2 convolution in CNNs:

```python
class PatchMerging(nn.Module):
    """
    Merge 2×2 adjacent patches into one, reducing resolution by 2×
    and doubling the channel dimension.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, H*W, C).
        H, W : int
            Spatial dimensions.

        Returns
        -------
        torch.Tensor
            Shape (B, H/2 * W/2, 2C).
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Extract 2×2 groups
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2 * W/2, 2C)
        return x
```

---

## Window-Based Self-Attention

### Local Windows

Instead of computing attention over all $N = HW/P^2$ patches (cost $O(N^2)$), Swin partitions the feature map into non-overlapping **windows** of size $M \times M$ (typically $M = 7$) and computes attention independently within each window:

$$\text{Cost}_{\text{global}} = O(N^2 d) = O\left(\frac{H^2 W^2}{P^4} d\right)$$

$$\text{Cost}_{\text{window}} = O\left(\frac{N}{M^2} \cdot M^4 d\right) = O(N M^2 d)$$

Since $M$ is fixed (typically 7), the window attention cost is **linear in $N$**:

| Resolution | $N$ | Global $O(N^2)$ | Window $O(NM^2)$, $M=7$ |
|---|---|---|---|
| 224 × 224 | 3,136 | 9.8M | 153K |
| 512 × 512 | 16,384 | 268M | 803K |
| 1024 × 1024 | 65,536 | 4.3B | 3.2M |

Window attention is 40–1300× cheaper than global attention at typical resolutions.

```python
def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, H, W, C).
    window_size : int
        Window size M.

    Returns
    -------
    torch.Tensor
        Shape (B * num_windows, M, M, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to feature map."""
    B_windows = windows.shape[0]
    num_windows = (H // window_size) * (W // window_size)
    B = B_windows // num_windows

    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
```

### The Cross-Window Problem

Window attention alone creates an information bottleneck: patches in different windows cannot attend to each other. A patch near the boundary of one window cannot see its spatial neighbor in the adjacent window.

---

## Shifted Window Mechanism

### How Shifting Works

Swin solves the cross-window problem by **shifting** the window partition by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels in alternating layers:

```
Layer ℓ (Regular Windows):          Layer ℓ+1 (Shifted Windows):

┌────┬────┬────┬────┐              ╔═══╦════╦════╦═══╗
│    │    │    │    │              ║   ║    ║    ║   ║
│ W₁ │ W₂ │ W₃ │ W₄ │              ║   ║    ║    ║   ║
│    │    │    │    │              ║ A ║ B  ║ C  ║ D ║
├────┼────┼────┼────┤    Shift    ╠═══╬════╬════╬═══╣
│    │    │    │    │   -----→    ║   ║    ║    ║   ║
│ W₅ │ W₆ │ W₇ │ W₈ │              ║ E ║ F  ║ G  ║ H ║
│    │    │    │    │              ║   ║    ║    ║   ║
├────┼────┼────┼────┤              ╠═══╬════╬════╬═══╣
│    │    │    │    │              ║ I ║ J  ║ K  ║ L ║
│ W₉ │W₁₀│W₁₁│W₁₂│              ╚═══╩════╩════╩═══╝
└────┴────┴────┴────┘
```

After shifting, patches that were at the boundary of adjacent regular windows now share a shifted window, enabling cross-window connections. Over two consecutive layers, every patch can indirectly communicate with all patches within a $2M \times 2M$ region.

### Efficient Implementation via Cyclic Shift

Rather than creating variable-sized windows at the boundaries (which would be computationally inconvenient), Swin uses **cyclic shifting** to maintain uniform window sizes, combined with an attention mask to prevent inappropriate cross-window attention:

```python
class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (M, M)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        M = window_size[0]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * M - 1) * (2 * M - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(M), torch.arange(M), indexing='ij'
        ))
        coords_flat = coords.reshape(2, -1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += M - 1
        relative_coords[:, :, 1] += M - 1
        relative_coords[:, :, 0] *= 2 * M - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (num_windows*B, M*M, C).
        mask : torch.Tensor, optional
            Attention mask for shifted windows, shape (num_windows, M*M, M*M).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        # Apply mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x
```

### Swin Transformer Block

Each Swin block alternates between regular and shifted window attention:

```python
class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with (shifted) window attention."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, H, W, attn_mask=None):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Reverse partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual + FFN
        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## Swin Model Configurations

| Model | $C$ | Layers per Stage | Heads per Stage | Params | FLOPs | ImageNet Top-1 |
|---|---|---|---|---|---|---|
| Swin-T | 96 | [2, 2, 6, 2] | [3, 6, 12, 24] | 29M | 4.5G | 81.3% |
| Swin-S | 96 | [2, 2, 18, 2] | [3, 6, 12, 24] | 50M | 8.7G | 83.0% |
| Swin-B | 128 | [2, 2, 18, 2] | [4, 8, 16, 32] | 88M | 15.4G | 83.5% |
| Swin-L | 192 | [2, 2, 18, 2] | [6, 12, 24, 48] | 197M | 34.5G | 86.3% |

Swin-T achieves 81.3% top-1 on ImageNet with only 4.5 GFLOPs—competitive with DeiT-S (79.8%) and EfficientNet-B3 (81.6%) at similar compute.

---

## Comparison with ViT and CNN

### Feature Hierarchy

```
Swin (hierarchical):         ViT (flat):             CNN (hierarchical):
Stage 1: H/4 × W/4 × C     All: H/16 × W/16 × d   Stage 1: H/4 × W/4 × 64
Stage 2: H/8 × W/8 × 2C                             Stage 2: H/8 × W/8 × 128
Stage 3: H/16 × W/16 × 4C                            Stage 3: H/16 × W/16 × 256
Stage 4: H/32 × W/32 × 8C                            Stage 4: H/32 × W/32 × 512
```

### Dense Prediction Performance

Swin's hierarchical features make it superior for dense tasks:

| Task | Backbone | Performance |
|---|---|---|
| Object Detection (COCO) | ResNet-50 | 41.0 AP |
| Object Detection (COCO) | Swin-T | 43.7 AP |
| Object Detection (COCO) | Swin-S | 45.0 AP |
| Semantic Segmentation (ADE20K) | ResNet-101 | 44.9 mIoU |
| Semantic Segmentation (ADE20K) | Swin-T | 44.5 mIoU |
| Semantic Segmentation (ADE20K) | Swin-S | 47.6 mIoU |

---

## Swin V2

**Swin V2** (Liu et al., 2022) extends the original with several improvements for scaling to larger models and higher resolutions:

1. **Post-norm** replaces pre-norm (stabilizes large model training)
2. **Scaled cosine attention** replaces dot-product attention:

$$\text{Attention}(i, j) = \frac{\cos(\mathbf{q}_i, \mathbf{k}_j)}{\tau} + B_{ij}$$

where $\tau$ is a learnable temperature per head, preventing attention entropy collapse at scale.

3. **Log-spaced continuous position bias** replaces the discrete bias table, enabling smoother resolution transfer:

$$B_{ij} = \text{MLP}(\log(1 + |\Delta_{ij}|) \cdot \text{sign}(\Delta_{ij}))$$

---

## Quantitative Finance Applications

### Multi-Scale Market Microstructure

Swin's hierarchical windowed attention maps naturally to **multi-scale market analysis**:

| Swin Stage | Financial Analog | Window Content |
|---|---|---|
| Stage 1 (finest) | Tick-level microstructure | Individual trades within a time window |
| Stage 2 | Minute-bar patterns | Aggregated trade statistics |
| Stage 3 | Hourly dynamics | Intraday trends and volatility |
| Stage 4 (coarsest) | Daily regime | Session-level features |

Each stage processes local patterns at its resolution, and patch merging aggregates information across time scales—analogous to how market participants operate at different frequencies.

### Windowed Attention for Order Book Data

Limit order book snapshots have natural locality: nearby price levels are more correlated than distant ones. Window attention captures this structure efficiently:

```python
# Order book as sequence: [bid_10, ..., bid_1, ask_1, ..., ask_10]
# Window size M=5: captures local price level dynamics
# Shifted windows: enable cross-spread interactions (bid ↔ ask)
```

### Computational Advantages for Real-Time Finance

Swin's linear complexity is particularly relevant for real-time financial applications:

- **High-frequency trading**: Processing order book updates at microsecond latency requires efficient architectures
- **Large asset universes**: Cross-sectional models over thousands of assets benefit from windowed attention
- **Streaming inference**: Hierarchical processing enables early-exit predictions from shallow stages when latency is critical

---

## Summary

The Swin Transformer bridges the gap between the flexibility of transformers and the practical requirements of hierarchical feature extraction and computational efficiency:

1. **Window attention** reduces complexity from $O(N^2)$ to $O(NM^2)$—linear in image size
2. **Shifted windows** enable cross-window information flow without global attention
3. **Hierarchical design** with patch merging produces multi-scale features for dense prediction
4. **Relative position bias** provides translation-aware attention without absolute position embeddings
5. Swin serves as a **drop-in replacement** for CNN backbones in detection and segmentation frameworks

For quantitative finance, Swin's multi-scale hierarchical processing and linear complexity make it well-suited for real-time applications over large asset universes and multi-frequency data.

---

## References

1. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.
2. Liu, Z., et al. (2022). Swin Transformer V2: Scaling Up Capacity and Resolution. *CVPR 2022*.
3. Dong, X., et al. (2022). CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Window Self-Attention. *CVPR 2022*.
4. Yang, J., et al. (2021). Focal Self-attention for Local-Global Interactions in Vision Transformers. *NeurIPS 2021*.
