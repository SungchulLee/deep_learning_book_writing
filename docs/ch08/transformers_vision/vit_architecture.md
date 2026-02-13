# ViT Architecture

## Overview

The Vision Transformer (ViT) architecture directly applies the standard transformer encoder to sequences of image patches, with minimal image-specific modifications.

## Architecture Details

Given an image $x \in \mathbb{R}^{H \times W \times C}$ and patch size $P$, the number of patches is $N = HW/P^2$.

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 d_model=768, n_heads=12, n_layers=12, n_classes=1000):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, 0])
```

## Model Variants

| Model | Layers | Hidden | Heads | Params |
|-------|--------|--------|-------|--------|
| ViT-S/16 | 12 | 384 | 6 | 22M |
| ViT-B/16 | 12 | 768 | 12 | 86M |
| ViT-L/16 | 24 | 1024 | 16 | 307M |
| ViT-H/14 | 32 | 1280 | 16 | 632M |

The notation ViT-X/P indicates variant X with patch size P. Smaller patches give more tokens (higher resolution) but quadratically higher compute.

## Key Design Choices

Pre-norm (LayerNorm before attention/FFN) is standard for ViT, unlike BERT which uses post-norm. GELU activation in the FFN. No convolutional layers except the initial patch embedding projection.

## Data Requirements

ViT requires large-scale pre-training (ImageNet-21K or JFT-300M) to match CNN performance. On ImageNet-1K alone, ViTs underperform ResNets due to lack of the inductive biases (locality, translation equivariance) that CNNs provide.
