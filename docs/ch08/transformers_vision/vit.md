# Vision Transformer (ViT)

## Introduction

The Vision Transformer (ViT) represents a paradigm shift in computer vision, demonstrating that pure transformer architectures—originally designed for natural language processing—can achieve state-of-the-art results on image classification tasks. Introduced by Dosovitskiy et al. in "An Image is Worth 16x16 Words" (2021), ViT treats images as sequences of patches, bridging the gap between CNNs and transformers.

## Motivation: Why Apply Transformers to Vision?

Traditional CNNs process images through hierarchical convolutions with local receptive fields. While effective, this approach has limitations:

**CNN Constraints:**
- Receptive field grows slowly with depth
- Long-range dependencies require many layers
- Inductive biases may limit flexibility on large datasets

**Transformer Advantages:**
- Global receptive field from the first layer
- Flexible attention patterns learned from data
- Proven scalability in NLP domain

The key insight of ViT is that images can be "tokenized" into patches, making them compatible with the transformer architecture that has revolutionized NLP.

## Architecture Overview

The ViT architecture consists of four main components:

```
Input Image → Patch Embedding → Transformer Encoder → Classification Head → Output
     ↓              ↓                    ↓                    ↓
  (H,W,C)    (N patches, D)      (N+1 tokens, D)         (classes)
```

### Mathematical Formulation

Given an input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, the ViT pipeline proceeds as follows:

**Step 1: Patch Extraction**
Divide the image into $N = \frac{HW}{P^2}$ non-overlapping patches of size $P \times P$:

$$\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}, \quad i = 1, \ldots, N$$

For a standard $224 \times 224$ image with $P = 16$: $N = \frac{224 \times 224}{16 \times 16} = 196$ patches, each of dimension $P^2 \cdot C = 16^2 \cdot 3 = 768$.

**Step 2: Linear Projection**
Project each flattened patch to dimension $D$:

$$\mathbf{z}_0^i = \mathbf{x}_p^i \mathbf{E} + \mathbf{e}_{pos}^i, \quad \mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$$

In practice, this projection is implemented as a 2D convolution with kernel size $P$ and stride $P$:

```python
# Equivalent to flattening patches + linear projection
self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
```

**Step 3: Class Token Prepending**
Prepend a learnable classification token:

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{z}_0^1; \mathbf{z}_0^2; \ldots; \mathbf{z}_0^N] + \mathbf{E}_{pos}$$

The `[CLS]` token aggregates information from all patches through self-attention and is used for the final classification decision.

**Step 4: Transformer Encoding**
Apply $L$ transformer encoder layers:

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$
$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

Note the **pre-norm** design (LayerNorm before each sublayer), consistent with modern Transformer practice.

**Step 5: Classification**
Use the class token for prediction:

$$\mathbf{y} = \text{MLP}_{head}(\text{LN}(\mathbf{z}_L^0))$$

### Positional Embeddings

ViT uses **learned 1D positional embeddings**—a simple lookup table that maps position index to a vector:

$$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

Despite images being 2D, learned 1D embeddings work well because the model discovers the 2D structure from data. Visualization of learned positional embeddings shows that nearby patches (in 2D image space) develop similar positional encodings, effectively recovering the grid layout.

**2D-aware alternatives** (explored in the original paper) include learned 2D embeddings that separate row and column position. These provide marginal improvement (~0.5%) and are not standard.

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification.
    
    Key innovations:
    1. Treats images as sequences of patches
    2. Uses transformer encoder for image classification
    3. Bridges CNN-style input processing with transformer architecture
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Convert image to patch embeddings
        x = self.patch_embed(x)  # (B, N, D)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification using class token
        x = self.norm(x)
        return self.head(x[:, 0])  # Take CLS token
```

## Model Variants

ViT comes in several standard configurations:

| Model | Parameters | Embed Dim | Depth | Heads | Patch Size |
|-------|-----------|-----------|-------|-------|------------|
| ViT-Tiny | 5M | 192 | 12 | 3 | 16 |
| ViT-Small | 22M | 384 | 12 | 6 | 16 |
| ViT-Base | 86M | 768 | 12 | 12 | 16 |
| ViT-Large | 307M | 1024 | 24 | 16 | 16 |
| ViT-Huge | 632M | 1280 | 32 | 16 | 14 |

### Factory Functions

```python
def create_vit_base(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Base: 86M parameters"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768,
        depth=12, n_heads=12, n_classes=n_classes
    )

def create_vit_large(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Large: 307M parameters"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024,
        depth=24, n_heads=16, n_classes=n_classes
    )
```

## Key Insights

### 1. Global Receptive Field
Unlike CNNs where receptive field grows gradually, ViT has global receptive field from the first layer through self-attention. This allows early layers to capture long-range spatial dependencies that would require many CNN layers to achieve.

### 2. Data Requirements
ViT requires large-scale pretraining (e.g., ImageNet-21k or JFT-300M) to outperform CNNs. With limited data, CNNs' inductive biases are advantageous. The critical threshold is approximately 10–100M images: below this, CNNs win; above it, ViT's flexibility pays off.

### 3. Computational Complexity
Self-attention has quadratic complexity $O(N^2)$ in sequence length, compared to CNNs' linear complexity in image size. For a $224 \times 224$ image with $16 \times 16$ patches, $N = 196$, making the attention matrix ($196 \times 196$) manageable. Larger images or smaller patches increase $N$ and the associated quadratic cost.

### 4. Transfer Learning
Pretrained ViT models transfer exceptionally well to downstream tasks, often surpassing CNN performance. When transferring to different image sizes, positional embeddings can be interpolated (2D bicubic interpolation of the learned position grid).

### 5. Hybrid Architectures
A practical compromise uses a CNN backbone (e.g., ResNet) to extract feature maps, then feeds the CNN features as "patches" to the Transformer:

$$\text{Image} \xrightarrow{\text{CNN}} \text{Feature Map} \xrightarrow{\text{Flatten}} \text{Patch Tokens} \xrightarrow{\text{Transformer}} \text{Output}$$

Hybrid models often outperform pure ViT at smaller scales because the CNN provides useful inductive biases (translation equivariance, locality) while the Transformer captures global interactions.

### 6. Self-Supervised Pre-training

**MAE (Masked Autoencoders)**: Mask a large fraction (75%) of patches and reconstruct the missing pixels. This is the vision analog of BERT's masked language modeling and has proven highly effective for learning visual representations.

**DINO**: Self-distillation with no labels—a student network learns to match the output of a momentum-updated teacher network. ViT features trained with DINO show remarkable emergent properties, including discovering object segmentation boundaries without any segmentation supervision.

## ViT Successors

| Model | Key Innovation | Improvement |
|-------|---------------|-------------|
| **DeiT** | Knowledge distillation from CNN teacher | Trains effectively on ImageNet-1K alone |
| **Swin Transformer** | Shifted window attention, hierarchical features | Linear complexity in image size |
| **BEiT** | BERT-style pre-training with visual tokens | Better self-supervised representations |
| **MAE** | High-ratio masking + reconstruction | Sample-efficient pre-training |
| **EVA** | Scaling ViT to 1B+ parameters | State-of-the-art vision representations |

## Training Considerations

**Effective Training Strategies:**

```python
# Label smoothing for better generalization
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1
)

# Cosine annealing with warmup
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

**Data Augmentation:**
- RandAugment or AutoAugment
- Mixup and CutMix
- Random erasing

## Applications in Quantitative Finance

ViT architectures have found applications in financial analysis:

1. **Chart Pattern Recognition**: Analyzing candlestick charts and technical patterns
2. **Document Analysis**: Processing financial reports and statements
3. **Satellite Imagery**: Economic activity estimation from aerial images
4. **Multi-modal Finance**: Combining visual and textual financial data

## References

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Touvron, H., et al. "Training data-efficient image transformers & distillation through attention." ICML 2021. (DeiT)
3. He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. (MAE)
4. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.
5. Caron, M., et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021. (DINO)
