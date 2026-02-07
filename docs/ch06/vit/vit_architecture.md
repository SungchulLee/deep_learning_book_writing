# ViT Architecture

## Introduction

The **Vision Transformer** (ViT) architecture, introduced by Dosovitskiy et al. (2021), applies the standard transformer encoder—without modification—to sequences of image patches. This section provides a comprehensive treatment of the complete ViT architecture, covering its components, training procedure, scaling behavior, and practical considerations.

---

## Complete Architecture

### End-to-End Pipeline

The ViT architecture consists of four stages:

```
┌───────────────────────────────────────────────────────────────────┐
│                     Vision Transformer (ViT)                      │
│                                                                   │
│  ┌──────────┐   ┌───────────┐   ┌─────────────┐   ┌──────────┐  │
│  │  Patch    │   │  + CLS    │   │ Transformer  │   │   MLP    │  │
│  │ Embedding │──▶│  + Pos    │──▶│   Encoder    │──▶│   Head   │  │
│  │           │   │  Embed    │   │  (L layers)  │   │          │  │
│  └──────────┘   └───────────┘   └─────────────┘   └──────────┘  │
│                                                                   │
│  (B,C,H,W)     (B,N+1,d)        (B,N+1,d)        (B,classes)    │
└───────────────────────────────────────────────────────────────────┘
```

### Formal Definition

Given an image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$:

**Step 1 — Patch Embedding:**

$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}} \;;\; \mathbf{x}_p^{(1)} \mathbf{E} \;;\; \cdots \;;\; \mathbf{x}_p^{(N)} \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

where $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times d}$, $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times d}$.

**Step 2 — Transformer Encoder** (repeated $L$ times):

$$\mathbf{z}_\ell' = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$

$$\mathbf{z}_\ell = \text{FFN}(\text{LN}(\mathbf{z}_\ell')) + \mathbf{z}_\ell'$$

Note the **pre-norm** configuration: layer normalization is applied before each sub-layer, not after. This differs from the original Transformer (post-norm) and provides more stable training for deep networks.

**Step 3 — Classification:**

$$\hat{y} = \text{MLP}_{\text{head}}(\text{LN}(\mathbf{z}_L^{(0)}))$$

### Transformer Encoder Layer Detail

Each encoder layer contains two sub-layers with residual connections:

**Multi-Head Self-Attention (MSA):**

$$\text{MSA}(\mathbf{z}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(\mathbf{z} W_i^Q, \mathbf{z} W_i^K, \mathbf{z} W_i^V)$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

where $d_k = d / h$ is the dimension per head.

**Feed-Forward Network (FFN):**

$$\text{FFN}(\mathbf{z}) = \text{GELU}(\mathbf{z} W_1 + b_1) W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, and $d_{\text{ff}} = 4d$ (the standard MLP ratio).

---

## Model Configurations

### Standard Variants

| Config | Layers $L$ | Dim $d$ | FFN Dim | Heads $h$ | Params | Patch Size |
|---|---|---|---|---|---|---|
| ViT-Ti | 12 | 192 | 768 | 3 | 5.7M | 16 |
| ViT-S | 12 | 384 | 1536 | 6 | 22M | 16 |
| ViT-B | 12 | 768 | 3072 | 12 | 86M | 16 |
| ViT-L | 24 | 1024 | 4096 | 16 | 307M | 16 |
| ViT-H | 32 | 1280 | 5120 | 16 | 632M | 14 |

The naming convention **ViT-B/16** indicates ViT-Base with patch size 16. Using patch size 32 reduces computation by $4\times$ but loses fine-grained spatial information.

### Parameter Counting

For ViT-B/16 on 224×224 images ($N = 196$, $d = 768$):

| Component | Formula | Parameters |
|---|---|---|
| Patch embedding | $P^2 C \times d + d$ | 590,592 |
| CLS token | $d$ | 768 |
| Position embedding | $(N + 1) \times d$ | 151,296 |
| Per encoder layer: MSA | $4 d^2 + 4d$ | 2,362,368 |
| Per encoder layer: FFN | $2 \times d \times 4d + d + 4d$ | 4,722,432 |
| Per encoder layer: LN (×2) | $4d$ | 3,072 |
| All 12 encoder layers | $12 \times 7{,}087{,}872$ | 85,054,464 |
| Final LN | $2d$ | 1,536 |
| Classification head | $d \times C + C$ | 769,000 |
| **Total** | | **~86.6M** |

The transformer encoder dominates at ~98% of total parameters, with patch embedding contributing < 1%.

---

## Training Procedure

### The Data Efficiency Problem

The original ViT paper revealed a critical insight: **ViTs require significantly more data than CNNs** to train effectively from scratch. Without convolutional inductive biases, the model must learn spatial structure entirely from data.

| Training Data | ViT-B/16 | ResNet-152 |
|---|---|---|
| ImageNet-1k (1.3M images) | 77.9% | 78.3% |
| ImageNet-21k (14M images) | 81.3% | 79.1% |
| JFT-300M (300M images) | **88.6%** | 87.5% |

ViT underperforms ResNet on ImageNet-1k but surpasses it decisively when pretrained on larger datasets. This crossover point—where transformers overtake CNNs—has implications for data-constrained domains like finance.

### Standard Training Recipe

The ViT training pipeline involves two phases:

**Phase 1 — Pretraining** (large dataset, lower resolution):

```python
# Pretraining configuration (ImageNet-21k)
config = {
    'optimizer': 'Adam',
    'base_lr': 1e-3,
    'weight_decay': 0.03,           # Higher than typical CNN training
    'warmup_epochs': 10,
    'total_epochs': 300,
    'batch_size': 4096,             # Large batch with LR scaling
    'image_size': 224,
    'label_smoothing': 0.1,
    'dropout': 0.0,                 # No dropout during pretraining
    'stochastic_depth': 0.1,       # Drop entire transformer layers
}
```

**Phase 2 — Fine-tuning** (target dataset, optionally higher resolution):

```python
# Fine-tuning configuration
config = {
    'optimizer': 'SGD',             # SGD often better for fine-tuning
    'lr': 0.01,
    'weight_decay': 0.0,
    'total_epochs': 20,
    'batch_size': 512,
    'image_size': 384,              # Higher resolution
    'label_smoothing': 0.1,
}
```

When fine-tuning at a higher resolution, position embeddings are interpolated (see [Position Embeddings](position_embeddings.md)).

### Data Augmentation

Because ViTs lack convolutional inductive biases, strong data augmentation is critical:

| Augmentation | Purpose | ViT Impact |
|---|---|---|
| RandAugment | Diverse transformations | Critical for ImageNet-1k training |
| Mixup ($\alpha = 0.8$) | Input interpolation regularization | Significant improvement |
| CutMix ($\alpha = 1.0$) | Patch-level mixing | Complements Mixup |
| Random erasing | Occlusion robustness | Moderate improvement |
| Color jitter | Color invariance | Standard |

```python
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    rand_augment_transform('rand-m9-mstd0.5', {}),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Regularization

ViTs benefit from aggressive regularization, especially when training on limited data:

- **Stochastic depth**: Randomly skip entire transformer layers during training (drop rate 0.1 for ViT-B, 0.4 for ViT-L)
- **Weight decay**: Higher than typical CNN training (0.03–0.3)
- **Label smoothing**: Standard $\epsilon = 0.1$
- **Dropout**: Applied in attention and FFN (0.0–0.1, often 0.0 with stochastic depth)

```python
class TransformerLayerWithStochasticDepth(nn.Module):
    """Transformer layer with stochastic depth regularization."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.drop_path_rate = drop_path_rate

    def drop_path(self, x):
        """Randomly drop the entire residual branch during training."""
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1 - self.drop_path_rate
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
        return x * mask / keep_prob  # Scale to maintain expected value

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
```

---

## Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop entire residual paths (stochastic depth)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_prob
        return x * keep / (1 - self.drop_prob)


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block (pre-norm)."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0,
                 attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer implementation.

    Parameters
    ----------
    img_size : int
        Input image size (square).
    patch_size : int
        Patch size.
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of classification classes.
    embed_dim : int
        Transformer embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        FFN hidden dimension ratio.
    drop_rate : float
        Dropout rate.
    attn_drop_rate : float
        Attention dropout rate.
    drop_path_rate : float
        Stochastic depth rate.
    """
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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth: linearly increasing drop rate per layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding: (B, C, H, W) → (B, N, d)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, d)

        # Add position embeddings and apply dropout
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Classify from CLS token
        return self.head(x[:, 0])


# Instantiate model variants
def vit_tiny(num_classes=1000):
    return VisionTransformer(embed_dim=192, depth=12, num_heads=3, num_classes=num_classes)

def vit_small(num_classes=1000):
    return VisionTransformer(embed_dim=384, depth=12, num_heads=6, num_classes=num_classes)

def vit_base(num_classes=1000):
    return VisionTransformer(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)

def vit_large(num_classes=1000):
    return VisionTransformer(embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)


# Test
model = vit_base(num_classes=10)
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f"Output shape: {out.shape}")  # (2, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Scaling Behavior

### Scaling Laws

ViTs exhibit predictable scaling behavior—performance improves log-linearly with compute, data, and model size:

$$\text{Error} \propto C^{-\alpha}$$

where $C$ is the compute budget and $\alpha \approx 0.07$ for vision tasks (Zhai et al., 2022). Key findings:

1. **Larger models are more compute-efficient**: ViT-L at 100 epochs outperforms ViT-B at 300 epochs despite similar total compute
2. **Data and model size should scale together**: Doubling model size without more data leads to diminishing returns
3. **Patch size should decrease as compute increases**: Finer patches improve performance but increase cost quadratically

### Comparison with CNN Scaling

| Scaling Dimension | CNN Behavior | ViT Behavior |
|---|---|---|
| Depth | Diminishing returns past ~100 layers | Consistent gains up to 32+ layers |
| Width | Moderate gains | Strong gains |
| Resolution | Strong gains (standard) | Strong gains (via smaller patches) |
| Data | Saturates earlier | Continues improving with more data |

---

## Quantitative Finance Considerations

### Data Constraints

Financial datasets are typically orders of magnitude smaller than ImageNet:

| Domain | Typical Dataset Size | ViT Viability |
|---|---|---|
| ImageNet-1k | 1.3M images | Marginal (needs DeiT-style training) |
| Daily equity returns (US) | ~5,000 stocks × 5,000 days ≈ 25M samples | Viable with small ViT |
| Intraday tick data | ~10B events/year | Viable but preprocessing-heavy |
| Options surfaces | ~1,000 underlyings × 252 days × 1 surface | Very limited (needs transfer learning) |

For most financial applications, **transfer learning** (pretraining on a related large dataset, fine-tuning on the target) or **small ViT variants** (ViT-Ti, ViT-S) are necessary.

### Architecture Recommendations for Finance

| Data Regime | Recommended Architecture | Rationale |
|---|---|---|
| < 10K samples | CNN (ResNet-18/34) | Strong inductive biases essential |
| 10K – 1M samples | Small ViT or DeiT with augmentation | Balance between bias and flexibility |
| 1M – 100M samples | Standard ViT-B | Sufficient data to learn spatial structure |
| > 100M samples | Large ViT with scaling | Transformer scaling laws apply |

### Self-Supervised Pretraining for Finance

When labeled financial data is scarce, self-supervised pretraining on unlabeled financial data can improve ViT performance:

- **Masked patch prediction**: Randomly mask patches of a time series or volatility surface and train to reconstruct them (analogous to masked image modeling)
- **Contrastive learning**: Train embeddings to be similar for augmented views of the same financial state (SimCLR/BYOL-style)
- **Next-period prediction**: Predict the next temporal patch given previous patches

---

## Summary

The Vision Transformer architecture demonstrates that standard transformers, with minimal modifications, can match or surpass CNNs on vision tasks when given sufficient data. Key architectural features include:

1. **Pre-norm** configuration for stable training of deep networks
2. **Stochastic depth** with linearly increasing drop rates
3. **Large batch training** with warmup and cosine schedule
4. **Strong data augmentation** to compensate for lack of inductive biases
5. Predictable **scaling laws** that guide compute allocation

For quantitative finance, the key implication is that the choice between CNN and ViT architectures should be driven primarily by the **data regime**: CNNs for data-scarce problems, ViTs for data-rich settings where flexibility to learn novel patterns justifies the higher data requirements.

---

## References

1. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
2. Zhai, X., et al. (2022). Scaling Vision Transformers. *CVPR 2022*.
3. Steiner, A., et al. (2022). How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers. *TMLR 2022*.
4. He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*.
5. Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention. *ICML 2021*.
