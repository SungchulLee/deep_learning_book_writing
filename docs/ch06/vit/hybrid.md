# Hybrid Architectures

## Introduction

**Hybrid architectures** combine convolutional and transformer components to leverage the strengths of both: CNN's strong inductive biases and local feature extraction efficiency, and the transformer's ability to model long-range dependencies through self-attention. Rather than choosing between CNNs and transformers, hybrid designs use convolutions for early-stage local feature extraction and transformers for later-stage global reasoning—or interleave both throughout the network.

This approach is motivated by a fundamental observation: the early layers of trained ViTs learn filters similar to CNN first layers (edge detectors, color patterns), suggesting that convolutions are a more efficient way to extract these low-level features. Transformers, on the other hand, excel at the higher-level task of integrating features across spatial locations—where their flexibility over fixed convolutional receptive fields provides genuine advantages.

---

## Motivation: Best of Both Worlds

### Empirical Evidence

Several findings motivate hybrid designs:

1. **Early ViT layers waste capacity**: The first few transformer layers of ViT learn local, CNN-like features at much higher computational cost than convolutions would require
2. **Early convolutions improve stability**: Xiao et al. (2021) showed that replacing the patch embedding with a small CNN stem dramatically improves ViT training stability and reduces sensitivity to hyperparameters
3. **Transformers need resolution**: Global self-attention is most valuable at coarser spatial resolutions where each token already encodes rich local information
4. **CNNs plateau at depth**: CNNs benefit less from additional depth beyond ~50–100 layers, while transformers continue improving

### The Hybrid Principle

```
                CNN Strengths              Transformer Strengths
               ┌──────────────┐          ┌───────────────────┐
               │ Local features│          │ Global dependencies│
Low-level ──── │ Translation eq│ ───────  │ Dynamic attention  │ ──── High-level
               │ Parameter eff.│          │ Flexible receptive │
               │ Fast on GPU   │          │ Scalable with data │
               └──────────────┘          └───────────────────┘

          Hybrid: Use CNN here ──────── Use Transformer here
          (Early stages, local)          (Late stages, global)
```

---

## Architecture Patterns

### Pattern 1: CNN Stem + Transformer Body

Replace ViT's linear patch embedding with a small convolutional network that progressively downsamples the input:

```python
import torch
import torch.nn as nn


class ConvStem(nn.Module):
    """
    CNN stem replacing the linear patch embedding.

    Progressively downsamples using strided convolutions,
    providing better low-level features than flat patch projection.

    Following Xiao et al. (2021), "Early Convolutions Help
    Transformers See Better."
    """
    def __init__(self, in_channels=3, embed_dim=768, target_stride=16):
        super().__init__()
        # Progressive downsampling: 2× per conv block
        # Total stride = 16 (matching ViT-B/16 patch size)
        self.stem = nn.Sequential(
            # Stage 1: stride 2 → H/2 × W/2
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 2: stride 2 → H/4 × W/4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Stage 3: stride 2 → H/8 × W/8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Stage 4: stride 2 → H/16 × W/16
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
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
            Patch-like embeddings, shape (B, N, embed_dim).
        """
        x = self.stem(x)  # (B, embed_dim, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class HybridViT(nn.Module):
    """ViT with CNN stem instead of linear patch embedding."""
    def __init__(self, img_size=224, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.conv_stem = ConvStem(in_channels, embed_dim)
        num_patches = (img_size // 16) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv_stem(x)  # (B, N, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])
```

**Benefits of CNN stem**:

- More stable training (tolerates wider learning rate ranges)
- Better performance at smaller model sizes
- Lower computational cost for early feature extraction
- Natural handling of different input resolutions

### Pattern 2: CNN Backbone + Transformer Head (CoAtNet Style)

Use a full CNN backbone for early stages and switch to transformer layers for later stages:

```python
class CoAtNetStyle(nn.Module):
    """
    Hybrid architecture: CNN stages → Transformer stages.

    Stage 0-1: MBConv (mobile inverted bottleneck) blocks
    Stage 2-3: Transformer blocks with relative attention

    Based on CoAtNet (Dai et al., 2021).
    """
    def __init__(self, img_size=224, num_classes=1000):
        super().__init__()

        # CNN stages (efficient local processing)
        self.stage0 = self._make_conv_stage(3, 64, num_blocks=2, stride=2)
        self.stage1 = self._make_conv_stage(64, 128, num_blocks=2, stride=2)

        # Transition: flatten spatial dims for transformer
        # After stage1: H/4 × W/4 × 128

        # Transformer stages (global reasoning)
        self.stage2 = self._make_transformer_stage(128, 256, num_blocks=6, stride=2)
        self.stage3 = self._make_transformer_stage(256, 512, num_blocks=2, stride=2)

        # After all stages: H/32 × W/32 × 512
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(512, num_classes)

    def _make_conv_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        for _ in range(num_blocks - 1):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ])
        return nn.Sequential(*layers)

    def _make_transformer_stage(self, in_dim, out_dim, num_blocks, stride):
        layers = nn.ModuleList()
        # Downsample with conv
        layers.append(nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        ))
        # Transformer blocks
        for _ in range(num_blocks):
            layers.append(nn.TransformerEncoderLayer(
                d_model=out_dim, nhead=out_dim // 64,
                dim_feedforward=4 * out_dim, dropout=0.1,
                activation='gelu', batch_first=True, norm_first=True,
            ))
        return layers

    def forward(self, x):
        # CNN stages: (B, C, H, W) → (B, 128, H/4, W/4)
        x = self.stage0(x)
        x = self.stage1(x)

        # Transition to transformer
        for i, layer in enumerate(self.stage2):
            if i == 0:
                x = layer(x)  # Conv downsample: (B, 256, H/8, W/8)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # (B, N, 256)
            else:
                x = layer(x)  # Transformer block

        # Reshape back for stage3 downsample
        x = x.transpose(1, 2).view(B, -1, H, W)
        for i, layer in enumerate(self.stage3):
            if i == 0:
                x = layer(x)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
            else:
                x = layer(x)

        # Pool and classify
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
```

### Pattern 3: Interleaved CNN-Transformer Blocks

Alternate between convolutional and transformer blocks within each stage, allowing local and global processing at every scale:

```python
class ConvTransformerBlock(nn.Module):
    """
    Interleaved convolution-transformer block.

    1. Depthwise conv for local features
    2. Self-attention for global features
    3. FFN for channel mixing
    """
    def __init__(self, dim, num_heads, kernel_size=7):
        super().__init__()
        # Local: depthwise convolution
        self.local_norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim,
        )
        self.local_proj = nn.Linear(dim, dim)

        # Global: self-attention
        self.global_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Channel: FFN
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        # Local processing (depthwise conv)
        h = self.local_norm(x)
        h = h.transpose(1, 2)  # (B, C, N) for Conv1d
        h = self.dwconv(h).transpose(1, 2)  # (B, N, C)
        x = x + self.local_proj(h)

        # Global processing (self-attention)
        h = self.global_norm(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]

        # Channel mixing (FFN)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

---

## Notable Hybrid Architectures

### Summary Table

| Architecture | Year | Design | Key Innovation |
|---|---|---|---|
| **ViT + ResNet stem** | 2021 | CNN stem → Transformer | Stabilizes ViT training |
| **CoAtNet** | 2021 | MBConv stages → Transformer stages | Systematic search over hybrid designs |
| **LeViT** | 2021 | Conv → Attention → Conv | Optimized for fast inference |
| **CvT** | 2021 | Conv projections in attention | Convolutional attention projections |
| **MaxViT** | 2022 | MBConv + Block + Grid attention | Multi-axis attention at every stage |
| **EfficientFormer** | 2022 | Conv stages + Transformer stage | Mobile-optimized hybrid |
| **ConvNeXt** | 2022 | Pure CNN with transformer design principles | "Modernized" ResNet matching Swin |

### ConvNeXt: Transformers Informing CNN Design

An important counterpoint to hybrid architectures is **ConvNeXt** (Liu et al., 2022), which demonstrates that a pure CNN—modernized with design principles borrowed from transformers—can match Swin Transformer performance:

| Design Decision | ResNet | ConvNeXt | Inspiration |
|---|---|---|---|
| Training recipe | 90 epochs, weak aug | 300 epochs, strong aug | DeiT |
| Stage compute ratio | [3,4,6,3] | [3,3,9,3] | Swin-T |
| Stem | 7×7 conv, stride 2 | 4×4 conv, stride 4 | Patchify |
| Activation | ReLU | GELU | Transformers |
| Normalization | BatchNorm | LayerNorm | Transformers |
| Depthwise conv | No | Yes (7×7) | Large kernel attention |
| Inverted bottleneck | No | Yes | MobileNet/FFN |

```python
class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv + inverted bottleneck."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        return residual + x
```

ConvNeXt achieves 87.8% top-1 on ImageNet (with ImageNet-22k pretraining), matching Swin-L while being simpler to implement and faster at inference.

---

## Design Guidelines

### When to Use Each Pattern

| Scenario | Recommended Architecture | Rationale |
|---|---|---|
| **Limited compute, need speed** | CNN stem + small transformer | CNN stem is efficient; fewer transformer layers |
| **Dense prediction tasks** | Swin or CoAtNet-style | Hierarchical features needed |
| **Large-scale pretraining** | Full ViT or Swin | Transformer scaling laws apply |
| **Mobile/edge deployment** | LeViT or EfficientFormer | Optimized for latency |
| **Simplicity + strong baseline** | ConvNeXt | Pure CNN, transformer-inspired design |
| **Maximum flexibility** | MaxViT | Multi-axis attention captures all patterns |

### Rules of Thumb

1. **Use convolutions for early stages** (first 1–2 stages) where local features dominate
2. **Use transformers for later stages** where global context provides genuine benefits
3. **Match the inductive bias to the data regime**: more convolution when data is scarce, more attention when data is abundant
4. **Profile actual latency**, not just FLOPs—attention operations have different hardware characteristics than convolutions

---

## Quantitative Finance Applications

### Multi-Frequency Financial Modeling

Hybrid architectures naturally map to multi-frequency financial data processing:

```
┌─────────────────────────────────────────────────┐
│         Hybrid Architecture for Finance          │
│                                                   │
│  CNN Stages (Local Patterns):                     │
│  ├─ 1D conv over tick data → microstructure       │
│  ├─ 1D conv over minute bars → intraday patterns  │
│  └─ Feature aggregation to daily resolution       │
│                                                   │
│  Transformer Stages (Global Relationships):       │
│  ├─ Cross-asset attention over daily features     │
│  ├─ Temporal attention across lookback window     │
│  └─ Multi-horizon prediction heads                │
└─────────────────────────────────────────────────┘
```

```python
class FinancialHybrid(nn.Module):
    """
    Hybrid CNN-Transformer for multi-frequency financial data.

    CNN stages extract local temporal patterns from high-frequency data.
    Transformer stages model cross-asset and long-range temporal dependencies.
    """
    def __init__(self, n_features, n_assets, seq_len, embed_dim=128, num_heads=8):
        super().__init__()

        # CNN: Local temporal pattern extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2),  # Downsample 5×
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=5, stride=5, padding=2),  # Downsample 5×
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Transformer: Global cross-asset and temporal reasoning
        self.cross_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                batch_first=True, norm_first=True,
            ),
            num_layers=4,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)  # Return prediction

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, n_assets, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Predicted returns, shape (B, n_assets).
        """
        B, A, T, F = x.shape

        # Process each asset's time series with CNN
        x = x.reshape(B * A, T, F).transpose(1, 2)  # (B*A, F, T)
        x = self.temporal_conv(x)  # (B*A, embed_dim, T')
        x = x.mean(dim=-1)  # Global average pool over time: (B*A, embed_dim)

        # Reshape to cross-asset sequence
        x = x.view(B, A, -1)  # (B, n_assets, embed_dim)

        # Cross-asset transformer attention
        x = self.cross_attention(x)
        x = self.norm(x)

        # Per-asset predictions
        return self.head(x).squeeze(-1)  # (B, n_assets)
```

### Volatility Surface Modeling

For volatility surface modeling, a hybrid architecture can use:

- **2D CNN** to capture local smile/skew patterns (nearby strikes and maturities)
- **Transformer** to model global term structure relationships and cross-strike dependencies that violate locality

### Practical Advantages in Finance

1. **Faster training convergence**: CNN components provide useful inductive biases for the structured parts of financial data (local temporal patterns), reducing the data requirements
2. **Better generalization**: The transformer component can adapt to changing market regimes and novel cross-asset relationships
3. **Interpretability**: Attention weights in the transformer stages reveal which assets and time periods drive predictions, while CNN features capture specific temporal patterns
4. **Computational efficiency**: CNN stages process high-frequency data efficiently before the more expensive transformer stages operate on compressed representations

---

## Summary

Hybrid CNN-Transformer architectures combine the efficiency and inductive biases of convolutions with the flexibility and global modeling capacity of self-attention:

1. **CNN stem + Transformer body** is the simplest hybrid, improving training stability and efficiency
2. **Stage-wise hybrids** (CoAtNet) use CNNs for early stages and transformers for later stages
3. **Interleaved designs** alternate local and global processing at every scale
4. **ConvNeXt** demonstrates that transformer design principles can be incorporated into pure CNNs
5. The optimal CNN-transformer balance depends on the **data regime**, **compute budget**, and **task requirements**

For quantitative finance, hybrid architectures offer a principled way to process multi-frequency data: CNNs for efficient local temporal pattern extraction, transformers for flexible cross-asset and long-range dependency modeling.

---

## References

1. Xiao, T., et al. (2021). Early Convolutions Help Transformers See Better. *NeurIPS 2021*.
2. Dai, Z., et al. (2021). CoAtNet: Marrying Convolution and Attention for All Data Sizes. *NeurIPS 2021*.
3. Graham, B., et al. (2021). LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference. *ICCV 2021*.
4. Wu, H., et al. (2021). CvT: Introducing Convolutions to Vision Transformers. *ICCV 2021*.
5. Tu, Z., et al. (2022). MaxViT: Multi-Axis Vision Transformer. *ECCV 2022*.
6. Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR 2022*.
7. Li, Y., et al. (2022). EfficientFormer: Vision Transformers at MobileNet Speed. *NeurIPS 2022*.
