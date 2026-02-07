# Diffusion Transformer (DiT)

The **Diffusion Transformer (DiT)** (Peebles & Xie, 2023) replaces the U-Net backbone with a standard Vision Transformer, demonstrating that transformer scaling laws apply to diffusion models.

## Motivation

U-Nets introduce architectural complexity (skip connections, resolution changes) that complicates scaling. DiT shows a simpler, pure-transformer architecture can match or exceed U-Net performance, with predictable scaling laws.

## Architecture

DiT processes noisy latent patches through a stack of transformer blocks conditioned on timestep and class label.

### Patchification

Input latent $z_t \in \mathbb{R}^{h \times w \times c}$ (from a pre-trained VAE) is divided into non-overlapping $p \times p$ patches, producing $N = hw/p^2$ tokens, each linearly projected to dimension $d$.

### DiT Block with adaLN-Zero

Each block uses **adaptive layer normalisation** where scale, shift, and gate parameters are predicted from the conditioning signal:

$$[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2] = \text{MLP}(c)$$

The $\alpha$ values gate residual connections, initialised to zero so the network starts as an identity function. The conditioning vector $c$ sums timestep and class embeddings.

### Output

A final linear layer maps tokens back to patch dimensions, and tokens are unpatchified to produce the noise prediction at original resolution.

## Scaling Properties

DiT follows clear scaling laws—FID improves log-linearly with compute:

| Model | Parameters | Gflops | FID-50K (256×256) |
|-------|-----------|--------|-------------------|
| DiT-S/8 | 33M | 6 | 68.4 |
| DiT-B/4 | 130M | 56 | 43.5 |
| DiT-L/2 | 458M | 197 | 9.62 |
| DiT-XL/2 | 675M | 524 | 2.27 |

Smaller patch size $p$ (more tokens) improves quality but increases compute quadratically.

## Comparison with U-Net

| Property | U-Net | DiT |
|----------|-------|-----|
| Architecture | Encoder-decoder with skips | Flat transformer |
| Resolution handling | Multi-scale | Single resolution (patches) |
| Conditioning | FiLM, cross-attention | adaLN-Zero |
| Scaling | Complex | Predictable scaling laws |
| Compute efficiency | Better for small models | Better at scale |

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np


class DiTBlock(nn.Module):
    """Single DiT block with adaLN-Zero conditioning."""

    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # adaLN-Zero: predict scale, shift, gate for both sublayers
        self.adaLN = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.adaLN(c).unsqueeze(1).chunk(6, dim=-1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params

        # Self-attention sublayer
        h = self.norm1(x) * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h)
        x = x + alpha1 * h

        # FFN sublayer
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.mlp(h)
        x = x + alpha2 * h

        return x


class DiT(nn.Module):
    """Simplified Diffusion Transformer."""

    def __init__(
        self,
        input_channels: int = 4,
        patch_size: int = 2,
        dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_classes: int = 1000,
        img_size: int = 32,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.proj = nn.Conv2d(
            input_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # Conditioning
        cond_dim = dim
        self.time_embed = nn.Sequential(
            nn.Linear(dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.class_embed = nn.Embedding(num_classes + 1, cond_dim)  # +1 for null

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, cond_dim) for _ in range(depth)]
        )

        # Output
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_size ** 2 * input_channels)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size

        # Patchify
        tokens = self.proj(x).flatten(2).transpose(1, 2) + self.pos_embed

        # Build conditioning
        t_emb = sinusoidal_embed(t, tokens.shape[-1])
        c = self.time_embed(t_emb)
        if y is not None:
            c = c + self.class_embed(y)

        # Transformer
        for block in self.blocks:
            tokens = block(tokens, c)

        # Unpatchify
        tokens = self.head(self.norm(tokens))
        h_p, w_p = H // p, W // p
        out = tokens.reshape(B, h_p, w_p, p, p, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return out


def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.unsqueeze(-1) * freqs
    return torch.cat([args.cos(), args.sin()], dim=-1)
```

## Significance

DiT established that diffusion models benefit from the same scaling principles as language models: more parameters and more compute yield better results on a smooth, predictable curve. This has driven the adoption of transformer-based architectures in production systems including Sora, Stable Diffusion 3, and FLUX.

## References

1. Peebles, W., & Xie, S. (2023). "Scalable Diffusion Models with Transformers." *ICCV*.
2. Chen, J., et al. (2024). "Pixart-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis." *ICLR*.
