# Patch Embedding

## Introduction

**Patch embedding** is the mechanism by which Vision Transformers convert a 2D image into a 1D sequence of token embeddings—the format required by the transformer encoder. This step is analogous to word embedding in NLP: just as each word is mapped to a dense vector, each image patch is mapped to a $d$-dimensional embedding vector. The design of the patch embedding layer determines the resolution, computational cost, and information content available to subsequent transformer layers.

---

## Mathematical Formulation

### From Image to Patches

Given an input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ and a patch size $P \times P$, the image is partitioned into a grid of non-overlapping patches:

$$N = \frac{H}{P} \times \frac{W}{P} = \frac{HW}{P^2}$$

Each patch $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P \times P \times C}$ is flattened into a vector $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2 C}$ and then linearly projected into the embedding space:

$$\mathbf{z}_0^{(i)} = \mathbf{x}_p^{(i)} \mathbf{E} + \mathbf{b}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 C) \times d}, \quad i = 1, \ldots, N$$

where $d$ is the transformer's hidden dimension and $\mathbf{b} \in \mathbb{R}^d$ is the bias term.

### Equivalence to Strided Convolution

The patch embedding operation is mathematically equivalent to a 2D convolution with kernel size $P$ and stride $P$:

$$\text{Conv2d}(\text{in\_channels}=C,\; \text{out\_channels}=d,\; \text{kernel\_size}=P,\; \text{stride}=P)$$

This equivalence is not just notational—it is the standard implementation in practice because it is more efficient than explicit reshaping and matrix multiplication:

```python
# Method 1: Explicit flatten + linear (conceptually clear)
patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=P, p2=P)
embeddings = linear(patches)  # (B, N, d)

# Method 2: Conv2d (equivalent, more efficient)
embeddings = conv2d(x)  # (B, d, H/P, W/P)
embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, N, d)
```

Both produce identical outputs, but the Conv2d implementation leverages optimized CUDA kernels for the combined extract-and-project operation.

### Dimensional Analysis

For a standard configuration (ImageNet with ViT-Base):

| Parameter | Value | Description |
|---|---|---|
| Image size $H \times W$ | $224 \times 224$ | Standard ImageNet resolution |
| Channels $C$ | 3 | RGB |
| Patch size $P$ | 16 | Standard for ViT-B/16 |
| Number of patches $N$ | $14 \times 14 = 196$ | Sequence length |
| Patch dimension $P^2 C$ | $16^2 \times 3 = 768$ | Raw patch vector size |
| Embedding dimension $d$ | 768 | Transformer hidden dim |
| Embedding parameters | $768 \times 768 + 768 = 590,592$ | $\mathbf{E}$ and $\mathbf{b}$ |

Note that for ViT-B/16, the raw patch dimension ($P^2 C = 768$) happens to equal the embedding dimension ($d = 768$), so $\mathbf{E}$ is a square matrix. This is coincidental—for ViT-B/32, $P^2 C = 3072 \neq 768$.

---

## Impact of Patch Size

Patch size $P$ is one of the most important hyperparameters in ViT, controlling the tradeoff between resolution and computational cost.

### Resolution-Cost Tradeoff

Since the sequence length $N = HW/P^2$ enters the self-attention computation as $O(N^2)$:

| Patch Size | Patches ($N$) | Attention Cost | Information per Patch |
|---|---|---|---|
| $P = 32$ | 49 | $49^2 = 2{,}401$ | Coarse (32×32 pixels) |
| $P = 16$ | 196 | $196^2 = 38{,}416$ | Standard |
| $P = 8$ | 784 | $784^2 = 614{,}656$ | Fine-grained |
| $P = 4$ | 3,136 | $3{,}136^2 = 9{,}834{,}496$ | Very fine (impractical without efficient attention) |

Halving the patch size **quadruples** the sequence length and increases attention cost by **16×**. This quadratic scaling is the primary computational bottleneck of standard ViTs.

### Effect on Learned Features

Smaller patches allow the model to capture finer spatial details but require the transformer to integrate information over longer sequences:

- **Large patches** ($P = 32$): Each patch contains a large spatial region, requiring fewer attention layers to model global relationships but losing fine-grained detail
- **Small patches** ($P = 16$): The standard balance—sufficient detail for most recognition tasks while maintaining manageable sequence lengths
- **Very small patches** ($P \leq 8$): Approach pixel-level processing but at prohibitive cost for standard attention

---

## PyTorch Implementation

### Standard Patch Embedding

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding.

    Splits the image into non-overlapping patches and projects
    each to a d-dimensional embedding vector.

    Parameters
    ----------
    img_size : int
        Input image size (assumes square images).
    patch_size : int
        Size of each patch (assumes square patches).
    in_channels : int
        Number of input channels (e.g., 3 for RGB).
    embed_dim : int
        Dimension of the output embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # Conv2d equivalent to flatten + linear projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Patch embeddings, shape (B, N, embed_dim).
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Expected ({self.img_size}, {self.img_size}), got ({H}, {W})"

        # (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, embed_dim, N) → (B, N, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# Demonstrate
patch_embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
img = torch.randn(4, 3, 224, 224)
tokens = patch_embed(img)
print(f"Input:  {img.shape}")       # (4, 3, 224, 224)
print(f"Output: {tokens.shape}")    # (4, 196, 768)
print(f"Num patches: {patch_embed.num_patches}")  # 196
```

### Overlapping Patch Embedding

Some architectures use overlapping patches to preserve boundary information that non-overlapping patches discard:

```python
class OverlappingPatchEmbedding(nn.Module):
    """
    Overlapping patch embedding using stride < kernel_size.

    This preserves spatial information at patch boundaries and
    provides smoother transitions between adjacent tokens.
    """
    def __init__(self, img_size=224, patch_size=16, stride=12, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches_h = (img_size - patch_size) // stride + 1
        self.num_patches_w = (img_size - patch_size) // stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size - stride) // 2,  # Maintain spatial size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# Compare patch counts
overlap_embed = OverlappingPatchEmbedding(img_size=224, patch_size=16, stride=12, embed_dim=768)
tokens_overlap = overlap_embed(img)
print(f"Non-overlapping patches: {patch_embed.num_patches}")    # 196
print(f"Overlapping patches:     {overlap_embed.num_patches}")  # 324 (18×18)
```

---

## Visualizing Patch Embeddings

### What the Projection Learns

The learned projection matrix $\mathbf{E}$ can be reshaped and visualized as a set of $d$ filters, each of size $P \times P \times C$. After training, these filters resemble Gabor-like basis functions—similar to the filters learned by the first layer of a CNN:

```python
import matplotlib.pyplot as plt

def visualize_patch_filters(model, num_filters=64, ncols=8):
    """Visualize the learned patch embedding filters."""
    # Extract Conv2d weights: (embed_dim, in_channels, P, P)
    weights = model.patch_embed.proj.weight.data.clone()

    # Normalize for visualization
    weights = weights - weights.min()
    weights = weights / weights.max()

    nrows = num_filters // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Show RGB filter as image
            filt = weights[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(filt)
        ax.axis('off')
    plt.suptitle('Learned Patch Embedding Filters', fontsize=14)
    plt.tight_layout()
    plt.show()
```

The learned filters typically show:

- **Edge detectors** at various orientations (similar to CNN first-layer filters)
- **Color-sensitive** patterns (different filters respond to different color channels)
- **Frequency-selective** patterns (some capture low-frequency, others high-frequency content)

This demonstrates that even without convolutional inductive biases, the patch embedding layer learns to extract similar low-level features—the structure is discovered from data rather than imposed by architecture.

---

## Quantitative Finance Applications

### Time Series Patching

The patch embedding concept extends directly to financial time series. Instead of spatial patches, we extract **temporal patches** from multivariate time series:

Given a multivariate time series $\mathbf{x} \in \mathbb{R}^{T \times F}$ with $T$ time steps and $F$ features:

$$N = \frac{T}{P}, \quad \mathbf{x}_p^{(i)} \in \mathbb{R}^{P \times F}$$

Each temporal patch captures $P$ consecutive time steps across all features. This was formalized in **PatchTST** (Nie et al., 2023):

```python
class TimeSeriesPatchEmbedding(nn.Module):
    """
    Patch embedding for multivariate time series.

    Splits time series into non-overlapping temporal windows
    and projects each to the embedding space.

    Parameters
    ----------
    seq_len : int
        Length of input time series.
    patch_len : int
        Length of each temporal patch.
    n_features : int
        Number of input features (e.g., OHLCV = 5).
    embed_dim : int
        Dimension of output embeddings.
    """
    def __init__(self, seq_len=252, patch_len=21, n_features=5, embed_dim=128):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.proj = nn.Linear(patch_len * n_features, embed_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, F) - batch of multivariate time series.

        Returns
        -------
        torch.Tensor
            Shape (B, N, embed_dim) - sequence of patch embeddings.
        """
        B, T, F = x.shape
        # Reshape into patches: (B, T, F) → (B, N, P*F)
        x = x.reshape(B, self.num_patches, self.patch_len * F)
        return self.proj(x)


# Example: 1 year of daily OHLCV data, 21-day (monthly) patches
ts_embed = TimeSeriesPatchEmbedding(seq_len=252, patch_len=21, n_features=5, embed_dim=128)
ts = torch.randn(32, 252, 5)  # 32 stocks, 252 days, 5 features
tokens = ts_embed(ts)
print(f"Time series: {ts.shape}")    # (32, 252, 5)
print(f"Patches:     {tokens.shape}")  # (32, 12, 128) — 12 monthly tokens
```

### Patch Size as Temporal Granularity

The choice of patch size in financial time series corresponds to **temporal granularity**, analogous to lookback windows in traditional signal processing:

| Patch Length | Interpretation | Use Case |
|---|---|---|
| 1 day | Individual observations | High-frequency features, no aggregation |
| 5 days | Weekly | Short-term momentum signals |
| 21 days | Monthly | Standard factor rebalancing frequency |
| 63 days | Quarterly | Fundamental factor cycles |
| 252 days | Annual | Long-term trend and seasonality |

### Volatility Surface Patching

For volatility surfaces discretized on a $K \times T$ grid (strikes × maturities), 2D patch embedding directly applies. Each patch captures a local region of the vol surface, and the transformer learns relationships between different strike-maturity regions:

```python
class VolSurfacePatchEmbedding(nn.Module):
    """Patch embedding for discretized volatility surfaces."""
    def __init__(self, n_strikes=20, n_maturities=10, patch_k=5, patch_t=5, embed_dim=128):
        super().__init__()
        self.num_patches = (n_strikes // patch_k) * (n_maturities // patch_t)
        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=(patch_k, patch_t),
            stride=(patch_k, patch_t),
        )

    def forward(self, x):
        # x: (B, 1, n_strikes, n_maturities)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x
```

---

## Common Pitfalls and Best Practices

### Pitfall 1: Input Size Mismatch

Non-overlapping patch embedding requires the image dimensions to be exactly divisible by the patch size:

```python
# ❌ Wrong: 225 is not divisible by 16
img = torch.randn(1, 3, 225, 225)
# This will silently truncate or error

# ✅ Correct: Resize to compatible dimensions
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])
```

### Pitfall 2: Forgetting Normalization

Unlike CNN features which are typically normalized by batch norm after each convolution, patch embeddings enter the transformer with only the projection's natural scale. Layer normalization within the transformer handles this, but poor initialization of the projection layer can cause early training instability:

```python
# ✅ Good practice: Initialize projection with controlled variance
nn.init.xavier_uniform_(self.proj.weight)
nn.init.zeros_(self.proj.bias)
```

### Pitfall 3: Resolution Generalization

A model trained with patch size $P$ on images of size $H \times W$ cannot directly process images of different sizes because the position embeddings have a fixed length. Interpolating position embeddings enables some generalization (see [Position Embeddings](position_embeddings.md)).

---

## Summary

Patch embedding converts images (or other grid-structured data) into sequences of tokens suitable for transformer processing. The key design choices are:

1. **Patch size** controls the resolution-cost tradeoff ($N \propto 1/P^2$, attention cost $\propto 1/P^4$)
2. **Embedding dimension** determines the capacity of each token representation
3. **Overlapping vs. non-overlapping** patches trade smoothness for efficiency
4. The operation is mathematically equivalent to a **strided convolution**, which is the standard implementation

In quantitative finance, patch embedding naturally extends to temporal patching of time series and spatial patching of volatility surfaces, with patch size corresponding to the granularity of temporal or spatial aggregation.

---

## References

1. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
2. Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*.
3. Yuan, L., et al. (2021). Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet. *ICCV 2021*.
4. Xiao, T., et al. (2021). Early Convolutions Help Transformers See Better. *NeurIPS 2021*.
