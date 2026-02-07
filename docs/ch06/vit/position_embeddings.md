# Position Embeddings for Images

## Introduction

Transformers are **permutation-equivariant** by design: self-attention computes pairwise similarities between all tokens, treating the sequence as a set with no inherent ordering. While this is elegant, it means a transformer cannot distinguish between different spatial arrangements of identical patches without explicit positional information. **Position embeddings** inject spatial structure into the token representations, enabling the model to learn position-dependent features.

For images, position embeddings must encode 2D spatial relationships—a fundamentally different challenge from the 1D sequential positions in NLP transformers. This section examines the design space of position embeddings for vision transformers, from simple learned embeddings to sophisticated relative and rotary schemes.

---

## Why Position Information Matters

### The Permutation Invariance Problem

Without position embeddings, a ViT processes the patch sequence as an unordered set. Given patches $\{\mathbf{z}^{(1)}, \ldots, \mathbf{z}^{(N)}\}$, self-attention computes:

$$\text{Attention}(\mathbf{z}^{(i)}, \mathbf{z}^{(j)}) = \frac{(\mathbf{z}^{(i)} W_Q)(\mathbf{z}^{(j)} W_K)^\top}{\sqrt{d_k}}$$

This is invariant to any permutation $\pi$ of the patch indices. A horizontally flipped image (which reverses columns of patches) would produce the same attention patterns—clearly undesirable for most vision tasks.

### What Position Embeddings Encode

After training, position embeddings learn to encode:

1. **Absolute position**: Where a patch is located in the image grid
2. **Relative distance**: How far apart two patches are (captured implicitly by learned embeddings, explicitly by relative methods)
3. **2D structure**: Row and column relationships in the image grid

Empirically, the learned position embeddings of a trained ViT show clear 2D spatial structure: each position embedding is most similar to its spatial neighbors, confirming that the model discovers the grid topology from data.

---

## Types of Position Embeddings

### 1D Learned Position Embeddings (Standard ViT)

The original ViT uses a simple set of learnable vectors, one per position:

$$\mathbf{z}_0^{(i)} = \mathbf{x}_p^{(i)} \mathbf{E} + \mathbf{e}_{\text{pos}}^{(i)}, \quad \mathbf{e}_{\text{pos}}^{(i)} \in \mathbb{R}^d, \quad i = 0, 1, \ldots, N$$

where $i = 0$ corresponds to the CLS token. The position embeddings $\{\mathbf{e}_{\text{pos}}^{(i)}\}_{i=0}^{N}$ are parameters learned during training.

Despite being "1D" (indexed by a single integer), these embeddings implicitly learn 2D spatial structure through training on images with consistent patch ordering (raster scan: left-to-right, top-to-bottom).

```python
import torch
import torch.nn as nn

class Learned1DPositionEmbedding(nn.Module):
    """Standard ViT 1D learned position embeddings."""
    def __init__(self, num_patches, embed_dim, include_cls=True):
        super().__init__()
        num_positions = num_patches + 1 if include_cls else num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """Add position embeddings to token sequence."""
        return x + self.pos_embed
```

**Advantages**: Simple, effective, no assumptions about input structure.

**Disadvantages**: Fixed to a specific sequence length (resolution), no explicit 2D structure.

### 2D Learned Position Embeddings

A natural extension assigns separate embeddings for row and column positions, reducing parameters while making the 2D structure explicit:

$$\mathbf{e}_{\text{pos}}^{(i,j)} = \mathbf{e}_{\text{row}}^{(i)} + \mathbf{e}_{\text{col}}^{(j)}$$

where $i \in \{1, \ldots, H/P\}$ indexes the row and $j \in \{1, \ldots, W/P\}$ indexes the column.

```python
class Learned2DPositionEmbedding(nn.Module):
    """Factored 2D learned position embeddings."""
    def __init__(self, grid_h, grid_w, embed_dim):
        super().__init__()
        self.row_embed = nn.Parameter(torch.zeros(1, grid_h, embed_dim // 2))
        self.col_embed = nn.Parameter(torch.zeros(1, grid_w, embed_dim // 2))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, x):
        """
        x: (B, N, embed_dim) where N = grid_h * grid_w
        """
        B = x.shape[0]
        # Expand row embeddings: (1, grid_h, d/2) → (1, grid_h, grid_w, d/2)
        row = self.row_embed.unsqueeze(2).expand(-1, -1, self.grid_w, -1)
        # Expand col embeddings: (1, grid_w, d/2) → (1, grid_h, grid_w, d/2)
        col = self.col_embed.unsqueeze(1).expand(-1, self.grid_h, -1, -1)
        # Concatenate and reshape
        pos = torch.cat([row, col], dim=-1)  # (1, grid_h, grid_w, d)
        pos = pos.reshape(1, -1, pos.shape[-1])  # (1, N, d)
        return x + pos


# Parameter comparison for 14×14 grid with d=768
n_patches = 14 * 14  # 196
d = 768
params_1d = (n_patches + 1) * d  # 151,296 (with CLS)
params_2d = (14 + 14) * (d // 2)  # 10,752
print(f"1D parameters: {params_1d:,}")   # 151,296
print(f"2D parameters: {params_2d:,}")   # 10,752
print(f"Reduction: {params_1d / params_2d:.1f}x")  # 14.1x
```

Dosovitskiy et al. (2021) found that 2D-aware position embeddings provide no significant improvement over 1D learned embeddings for ViT, suggesting that the model can infer 2D structure from the data.

### Sinusoidal Position Embeddings

Extending the sinusoidal scheme from the original Transformer to 2D:

$$\mathbf{e}_{\text{pos}}^{(i,j)} = [\sin(i/\omega_1), \cos(i/\omega_1), \ldots, \sin(j/\omega_1), \cos(j/\omega_1), \ldots]$$

where $\omega_k = 10000^{2k/d}$ are frequency terms.

```python
import math

def sinusoidal_2d_position_embedding(grid_h, grid_w, embed_dim):
    """
    Generate fixed 2D sinusoidal position embeddings.

    Parameters
    ----------
    grid_h, grid_w : int
        Grid dimensions.
    embed_dim : int
        Embedding dimension (must be divisible by 4).

    Returns
    -------
    torch.Tensor
        Position embeddings, shape (grid_h * grid_w, embed_dim).
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    half_dim = embed_dim // 4

    omega = torch.arange(half_dim, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / half_dim))

    row_pos = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    col_pos = torch.arange(grid_w, dtype=torch.float32).unsqueeze(1)  # (W, 1)

    row_embed = torch.cat([
        torch.sin(row_pos * omega),
        torch.cos(row_pos * omega),
    ], dim=1)  # (H, embed_dim/2)

    col_embed = torch.cat([
        torch.sin(col_pos * omega),
        torch.cos(col_pos * omega),
    ], dim=1)  # (W, embed_dim/2)

    # Broadcast and concatenate
    pos_embed = torch.cat([
        row_embed.unsqueeze(1).expand(-1, grid_w, -1),
        col_embed.unsqueeze(0).expand(grid_h, -1, -1),
    ], dim=2)  # (H, W, embed_dim)

    return pos_embed.reshape(-1, embed_dim)  # (H*W, embed_dim)
```

**Advantages**: No learnable parameters, provides a smooth prior on spatial distances, generalizes to unseen positions.

**Disadvantages**: Fixed frequency spectrum may not match the data's spatial structure.

### Relative Position Embeddings

Rather than encoding absolute positions, relative position embeddings encode the **spatial offset** between pairs of tokens in the attention computation:

$$\text{Attention}(i, j) = \frac{(\mathbf{z}^{(i)} W_Q)(\mathbf{z}^{(j)} W_K)^\top + \mathbf{z}^{(i)} W_Q \mathbf{r}_{i-j}^\top}{\sqrt{d_k}}$$

where $\mathbf{r}_{i-j}$ is a learnable embedding indexed by the relative position $(i - j)$.

For 2D images, the relative position has two components (row offset, column offset). Swin Transformer uses a **relative position bias** added directly to the attention logits:

$$\text{Attention}(i, j) = \frac{Q_i K_j^\top}{\sqrt{d_k}} + B_{(r_i - r_j, c_i - c_j)}$$

where $B$ is a learnable bias table indexed by relative row and column offsets.

```python
class RelativePositionBias(nn.Module):
    """2D relative position bias for attention (as in Swin Transformer)."""
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size  # (Wh, Ww)
        Wh, Ww = window_size

        # Bias table: (2*Wh-1) * (2*Ww-1) possible relative positions
        num_relative_positions = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
        coords_flat = coords.reshape(2, -1)  # (2, Wh*Ww)

        # Pairwise relative coordinates
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += Wh - 1  # Shift to start from 0
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1  # Convert to 1D index
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, n):
        """Return relative position bias of shape (num_heads, n, n)."""
        bias = self.relative_position_bias_table[self.relative_position_index[:n, :n]]
        return bias.permute(2, 0, 1).contiguous()  # (num_heads, n, n)
```

### Rotary Position Embeddings (RoPE)

Originally developed for NLP, RoPE has been adapted for 2D images. It encodes position by rotating query and key vectors in pairs of dimensions:

$$\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \odot \cos(m\theta) + \text{rotate}(\mathbf{q}) \odot \sin(m\theta)$$

For 2D, separate rotations encode row and column positions:

$$\text{RoPE}_{2D}(\mathbf{q}, i, j) = \text{RoPE}_{\text{row}}(\text{RoPE}_{\text{col}}(\mathbf{q}, j), i)$$

RoPE naturally encodes relative distances because $\text{RoPE}(\mathbf{q}, m)^\top \text{RoPE}(\mathbf{k}, n)$ depends only on $m - n$.

---

## Resolution Generalization via Interpolation

A key practical challenge is that position embeddings trained at one resolution cannot directly transfer to another. For a model trained on $224 \times 224$ images with $P = 16$ (producing a $14 \times 14$ grid), applying it to $384 \times 384$ images produces a $24 \times 24$ grid with 576 patches—but only 196 position embeddings exist.

### Bicubic Interpolation

The standard solution interpolates the learned position embeddings to the new grid size:

```python
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed, old_grid_size, new_grid_size):
    """
    Interpolate position embeddings for resolution transfer.

    Parameters
    ----------
    pos_embed : torch.Tensor
        Original position embeddings (1, N_old + 1, d), including CLS.
    old_grid_size : tuple
        Original (H, W) grid dimensions.
    new_grid_size : tuple
        Target (H, W) grid dimensions.

    Returns
    -------
    torch.Tensor
        Interpolated position embeddings (1, N_new + 1, d).
    """
    # Separate CLS token and patch position embeddings
    cls_pos = pos_embed[:, :1, :]  # (1, 1, d)
    patch_pos = pos_embed[:, 1:, :]  # (1, N_old, d)

    d = patch_pos.shape[-1]
    old_h, old_w = old_grid_size
    new_h, new_w = new_grid_size

    # Reshape to 2D grid: (1, old_h, old_w, d) → (1, d, old_h, old_w)
    patch_pos = patch_pos.reshape(1, old_h, old_w, d).permute(0, 3, 1, 2)

    # Bicubic interpolation to new grid
    patch_pos = F.interpolate(
        patch_pos.float(),
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=False,
    )

    # Reshape back: (1, d, new_h, new_w) → (1, N_new, d)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, d)

    # Recombine with CLS token
    return torch.cat([cls_pos, patch_pos], dim=1)


# Example: transfer from 224 → 384
pos_224 = torch.randn(1, 197, 768)  # 14*14 + 1 CLS
pos_384 = interpolate_pos_embed(pos_224, (14, 14), (24, 24))
print(f"Original: {pos_224.shape}")      # (1, 197, 768)
print(f"Interpolated: {pos_384.shape}")  # (1, 577, 768)
```

Fine-tuning at the new resolution after interpolation typically recovers most of the performance lost from the resolution mismatch.

---

## Quantitative Finance Applications

### Variable-Length Asset Sequences

In cross-sectional financial models, the number of assets $N$ may vary (e.g., different index compositions over time). Position embeddings must accommodate variable sequence lengths:

- **Learned absolute**: Requires padding to maximum size, wasting capacity for smaller universes
- **Sinusoidal**: Naturally handles any sequence length
- **Relative**: Most suitable for financial data, since relationships between assets are often better characterized by relative features (sector distance, correlation) than absolute position

### Temporal Position in Multi-Horizon Models

For multi-horizon return prediction, temporal position embeddings encode the forecast horizon:

$$\mathbf{e}_{\text{pos}}^{(t)} \text{ encodes both calendar time and forecast offset}$$

Sinusoidal embeddings with frequencies aligned to known market cycles (weekly = 5 days, monthly = 21 days, quarterly = 63 days) can inject useful temporal priors:

```python
def market_aware_position_embedding(seq_len, embed_dim, market_freqs=[5, 21, 63, 252]):
    """Position embeddings with market-cycle-aware frequencies."""
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    dim_per_freq = embed_dim // (2 * len(market_freqs))

    embeddings = []
    for freq in market_freqs:
        omega = 2 * math.pi / freq
        dims = torch.arange(dim_per_freq, dtype=torch.float32)
        freqs = omega / (10 ** (dims / dim_per_freq))
        embeddings.append(torch.sin(pos * freqs))
        embeddings.append(torch.cos(pos * freqs))

    return torch.cat(embeddings, dim=1)[:, :embed_dim]
```

---

## Empirical Comparison

| Method | Parameters | Resolution Transfer | Relative Distance | Performance (ImageNet) |
|---|---|---|---|---|
| 1D Learned | $O(N \cdot d)$ | Requires interpolation | Implicit | Strong baseline |
| 2D Learned | $O((H'+W') \cdot d)$ | Separate row/col interpolation | Implicit | Comparable to 1D |
| Sinusoidal | 0 | Natural extrapolation | Implicit | Slightly below learned |
| Relative Bias | $O((2H'-1)(2W'-1) \cdot h)$ | Generalizes within window | Explicit | Best for local attention |
| RoPE | 0 | Good extrapolation | Explicit | Competitive |

The original ViT paper found that 1D learned embeddings perform as well as more complex 2D schemes, suggesting that the transformer learns to infer spatial structure from the data. However, for architectures with local attention windows (Swin), relative position biases provide meaningful improvements.

---

## Summary

Position embeddings are essential for enabling transformers to process spatially structured data. Key takeaways:

1. **1D learned embeddings** are the simplest and surprisingly effective—the standard choice for ViT
2. **2D factored embeddings** reduce parameters but offer marginal performance gains
3. **Sinusoidal embeddings** provide resolution generalization without learned parameters
4. **Relative position biases** capture spatial relationships explicitly and are preferred for local attention
5. **Resolution transfer** requires position embedding interpolation, typically bicubic
6. In quantitative finance, position embeddings encode temporal structure, forecast horizons, or cross-sectional ordering

---

## References

1. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
2. Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations. *NAACL 2018*.
3. Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, 568, 127063.
4. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.
5. Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS 2017*.
