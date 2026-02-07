# ViT Overview

## Introduction

**Vision Transformers** (ViTs) represent a paradigm shift in computer vision: rather than relying on convolutions with their built-in inductive biases of locality and translation equivariance, ViTs treat an image as a **sequence of patches** and process it with the same transformer architecture that revolutionized natural language processing. The core insight, introduced by Dosovitskiy et al. (2021), is that with sufficient data and compute, transformers can match or surpass CNNs on image tasks without any convolution-specific priors.

This challenges a decades-long assumption that spatial inductive biases are essential for vision. ViTs instead learn these biases from data—an approach that trades sample efficiency for flexibility, a tradeoff particularly relevant in quantitative finance where the nature of spatial/temporal structure may be unknown or time-varying.

---

## From Sequences to Images

### The Key Idea

Transformers operate on sequences of tokens. To apply them to images, ViT converts a 2D image into a 1D sequence through a simple procedure:

1. **Split** the image into fixed-size non-overlapping patches
2. **Flatten** each patch into a vector
3. **Project** each flattened patch into a $d$-dimensional embedding space
4. **Prepend** a learnable classification token (`[CLS]`)
5. **Add** learnable position embeddings
6. **Process** the resulting sequence with a standard transformer encoder

```
Image (H × W × C)
    │
    ▼
┌─────────────────────────────────┐
│  Split into N = HW/P² patches   │
│  Each patch: P × P × C          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Flatten: P²C-dimensional vector │
│  per patch                       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Linear projection → d dims      │
│  + CLS token + Position Emb.     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Transformer Encoder (L layers)  │
│  Multi-Head Self-Attention + FFN │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  CLS token output → MLP Head     │
│  → Classification                │
└─────────────────────────────────┘
```

### Mathematical Formulation

Given an image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ and patch size $P$, the number of patches is:

$$N = \frac{H \times W}{P^2}$$

Each patch $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2 \cdot C}$ is linearly projected:

$$\mathbf{z}_0^{(i)} = \mathbf{x}_p^{(i)} \mathbf{E} + \mathbf{e}_{\text{pos}}^{(i)}, \quad i = 1, \ldots, N$$

where $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times d}$ is the patch embedding matrix and $\mathbf{e}_{\text{pos}}^{(i)} \in \mathbb{R}^d$ is the position embedding. The full input sequence, including the CLS token, is:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}} \;;\; \mathbf{z}_0^{(1)} \;;\; \cdots \;;\; \mathbf{z}_0^{(N)}] \in \mathbb{R}^{(N+1) \times d}$$

This sequence is then processed by $L$ transformer encoder layers:

$$\mathbf{z}_\ell' = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \ell = 1, \ldots, L$$

$$\mathbf{z}_\ell = \text{FFN}(\text{LN}(\mathbf{z}_\ell')) + \mathbf{z}_\ell', \quad \ell = 1, \ldots, L$$

where MSA is multi-head self-attention, FFN is a feed-forward network, and LN is layer normalization. The final classification uses the CLS token output:

$$\hat{y} = \text{MLP}_{\text{head}}(\text{LN}(\mathbf{z}_L^{(0)}))$$

---

## Comparison with CNNs

### Inductive Biases

The fundamental difference between ViTs and CNNs lies in their inductive biases:

| Property | CNN | ViT |
|---|---|---|
| **Locality** | Built-in (kernels are local) | Must be learned from data |
| **Translation equivariance** | Exact (weight sharing) | Approximate (learned from data) |
| **Scale hierarchy** | Explicit (pooling layers) | Implicit (attention can attend to any scale) |
| **Global context** | Only at deep layers (large receptive fields) | From the first layer (self-attention is global) |
| **Parameter efficiency** | High (shared kernels) | Lower (requires more data) |
| **Flexibility** | Limited by fixed kernel size | Can learn arbitrary spatial relationships |

### When ViTs Excel

ViTs tend to outperform CNNs when:

- **Large datasets** are available (>100M images for pretraining, or strong augmentation strategies)
- **Global relationships** matter (the task requires understanding long-range spatial dependencies)
- **Transfer learning** is the primary paradigm (pretrained ViTs transfer well across tasks)
- **Flexibility** is valued over sample efficiency

ViTs tend to underperform CNNs when:

- **Small datasets** are available (insufficient data to learn spatial biases)
- **Local features** dominate (tasks where locality is the primary structure)
- **Computational budget** is limited (self-attention is $O(N^2)$ in sequence length)

---

## ViT Model Variants

The original ViT paper defined several model sizes following BERT naming conventions:

| Model | Layers ($L$) | Hidden Dim ($d$) | MLP Dim | Heads | Parameters |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

Patch sizes of 16 and 32 are common, with notation like **ViT-B/16** indicating ViT-Base with 16×16 patches. Smaller patches produce longer sequences and increase computational cost quadratically but improve resolution.

### Computational Cost

For an image with $N$ patches and model dimension $d$:

- **Self-attention**: $O(N^2 d)$ — quadratic in number of patches
- **FFN**: $O(N d^2)$ — linear in number of patches
- **Total per layer**: $O(N^2 d + N d^2)$

For a 224×224 image with 16×16 patches, $N = 196$. Doubling resolution to 448×448 gives $N = 784$, increasing the attention cost by $16\times$. This quadratic scaling motivates efficient variants like Swin Transformer.

---

## PyTorch Implementation

### Minimal ViT

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel_size=stride=patch_size is equivalent to
        # flattening patches and applying a linear projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, N, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (as in original ViT)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)

        # Add position embeddings
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # Classification from CLS token
        cls_output = x[:, 0]  # (B, embed_dim)
        return self.head(cls_output)


# Verify shapes
model = ViT(img_size=224, patch_size=16, num_classes=10, embed_dim=768, depth=12, num_heads=12)
dummy = torch.randn(2, 3, 224, 224)
output = model(dummy)
print(f"Input:  {dummy.shape}")   # (2, 3, 224, 224)
print(f"Output: {output.shape}")  # (2, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Quantitative Finance Applications

### Cross-Asset Attention

ViTs' global self-attention mechanism is naturally suited to modeling **cross-asset dependencies**. By treating a universe of $N$ assets as a sequence (analogous to image patches), a transformer can learn pairwise attention weights that capture dynamic correlation structures:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The attention matrix reveals which assets the model considers most relevant for predicting each target—providing interpretable cross-sectional factor relationships.

### Volatility Surface as Image

A discretized volatility surface $\sigma(K, T)$ over a grid of strikes $K$ and maturities $T$ can be treated as a single-channel image. ViT's ability to capture both local structure (smile curvature) and global relationships (term structure correlations) makes it well-suited for:

- Volatility surface interpolation and extrapolation
- Anomaly detection in option markets
- Dynamic hedging parameter estimation

### Advantages over CNNs in Finance

In financial applications, ViTs offer several advantages:

1. **Non-local dependencies**: Asset returns exhibit correlations that are not spatially local; self-attention captures these directly
2. **Dynamic attention**: The attention pattern adapts to market regime, unlike fixed convolutional kernels
3. **Flexible input structure**: Variable-length sequences (different numbers of assets, maturities) are handled naturally
4. **Interpretability**: Attention weights provide insight into which features or assets drive predictions

---

## Historical Context and Significance

The ViT paper challenged the deeply held assumption that convolutional inductive biases are necessary for vision. Key milestones:

- **2017**: Transformers introduced for NLP (Vaswani et al.)
- **2019**: Stand-Alone Self-Attention (Ramachandran et al.) replaces some convolutions with local attention
- **2020**: ViT (Dosovitskiy et al.) eliminates convolutions entirely, matching SOTA with large-scale pretraining
- **2021**: DeiT (Touvron et al.) demonstrates training-efficient ViTs on ImageNet alone
- **2021**: Swin Transformer (Liu et al.) introduces hierarchical structure with linear complexity

This progression reveals a general principle: **stronger inductive biases help when data is scarce, but become limiting when data is abundant**. The same principle applies in quantitative finance—parametric factor models (strong bias) vs. flexible deep learning (weak bias).

---

## Section Roadmap

This section covers the complete ViT pipeline:

1. **[Patch Embedding](patch_embedding.md)**: Converting images to sequences of token embeddings
2. **[Position Embeddings for Images](position_embeddings.md)**: Encoding spatial structure in the transformer
3. **[CLS Token](cls_token.md)**: The classification mechanism and alternatives
4. **[ViT Architecture](vit_architecture.md)**: Complete architecture details and training procedures
5. **[DeiT](deit.md)**: Data-efficient training with knowledge distillation
6. **[Swin Transformer](swin.md)**: Hierarchical vision transformers with shifted windows
7. **[Hybrid Architectures](hybrid.md)**: Combining convolutional and attention-based processing

---

## References

1. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
2. Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS 2017*.
3. Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention. *ICML 2021*.
4. Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.
5. Khan, S., et al. (2022). Transformers in Vision: A Survey. *ACM Computing Surveys*, 54(10s).
