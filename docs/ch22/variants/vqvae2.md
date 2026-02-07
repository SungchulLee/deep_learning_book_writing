# VQ-VAE-2: Hierarchical Vector Quantized VAE

Multi-scale discrete latent representations for high-fidelity generation.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the hierarchical architecture of VQ-VAE-2
- Understand how multi-scale quantization captures local and global features
- Describe the two-stage generation pipeline (VQ-VAE-2 + autoregressive prior)
- Implement the hierarchical encoder-decoder structure

---

## Motivation: Beyond Single-Scale VQ-VAE

### Limitations of VQ-VAE

Standard VQ-VAE uses a single level of quantization. For complex, high-resolution images, a single codebook must simultaneously capture global structure (shape, layout, pose) and local details (texture, fine edges). This dual demand limits both reconstruction quality and generation coherence.

### The Hierarchical Solution

VQ-VAE-2 (Razavi et al., 2019) introduces **multiple levels of quantization** operating at different spatial resolutions. Each level captures information at a different scale:

- **Top level:** Low spatial resolution, encodes global structure
- **Bottom level:** Higher spatial resolution, encodes local details conditioned on top-level codes

---

## Architecture

### Two-Level Hierarchy

```
Input x (256×256)
       │
       ▼
┌──────────────┐
│  Encoder_bot │──────► z_e_bot (64×64)
└──────┬───────┘              │
       │                      ▼
       │               ┌─────────────┐
       │               │  VQ_bottom  │──► z_q_bot (64×64, discrete)
       │               └─────────────┘
       ▼                      │
┌──────────────┐              │
│  Encoder_top │──────► z_e_top (32×32)
└──────────────┘              │
                              ▼
                       ┌─────────────┐
                       │  VQ_top     │──► z_q_top (32×32, discrete)
                       └─────────────┘
                              │
                       ┌──────┴──────┐
                       ▼             ▼
                ┌─────────────┐  ┌─────────────┐
                │ Decoder_top │  │ Decoder_bot │──► x̂ (256×256)
                └─────────────┘  └─────────────┘
```

### Information Flow

The top encoder processes the bottom encoder's output to extract global features. During decoding, the top-level codes are decoded first, providing a "scaffold" that conditions the bottom-level decoder for local detail generation.

---

## Implementation

### Hierarchical VQ-VAE-2 Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual block for encoder/decoder."""
    
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, x):
        return x + self.block(x)


class VQVAE2(nn.Module):
    """
    VQ-VAE-2 with two-level hierarchy.
    
    Args:
        in_channels: Input image channels
        hidden_dim: Base hidden dimension
        num_embeddings: Codebook size for each level
        embedding_dim: Dimension of codebook vectors
        num_res_blocks: Number of residual blocks per stage
    """
    
    def __init__(self, in_channels=3, hidden_dim=128,
                 num_embeddings=512, embedding_dim=64,
                 num_res_blocks=2):
        super().__init__()
        
        # ============ BOTTOM ENCODER ============
        # Downsample 4x: input → hidden features
        enc_bot_layers = [
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        ]
        for _ in range(num_res_blocks):
            enc_bot_layers.append(ResBlock(hidden_dim))
        self.encoder_bot = nn.Sequential(*enc_bot_layers)
        
        # ============ TOP ENCODER ============
        # Further downsample 2x from bottom encoder output
        enc_top_layers = [
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        ]
        for _ in range(num_res_blocks):
            enc_top_layers.append(ResBlock(hidden_dim))
        self.encoder_top = nn.Sequential(*enc_top_layers)
        
        # Pre-quantization projections
        self.pre_vq_bot = nn.Conv2d(hidden_dim, embedding_dim, 1)
        self.pre_vq_top = nn.Conv2d(hidden_dim, embedding_dim, 1)
        
        # ============ VECTOR QUANTIZERS ============
        from vqvae import VectorQuantizer  # reuse from VQ-VAE section
        self.vq_bot = VectorQuantizer(num_embeddings, embedding_dim)
        self.vq_top = VectorQuantizer(num_embeddings, embedding_dim)
        
        # ============ TOP DECODER ============
        # Upsample top codes to bottom resolution
        dec_top_layers = [nn.Conv2d(embedding_dim, hidden_dim, 3, padding=1)]
        for _ in range(num_res_blocks):
            dec_top_layers.append(ResBlock(hidden_dim))
        dec_top_layers.append(
            nn.ConvTranspose2d(hidden_dim, embedding_dim, 4, stride=2, padding=1)
        )
        self.decoder_top = nn.Sequential(*dec_top_layers)
        
        # ============ BOTTOM DECODER ============
        # Combines bottom codes + upsampled top codes
        dec_bot_layers = [
            nn.Conv2d(embedding_dim + embedding_dim, hidden_dim, 3, padding=1)
        ]
        for _ in range(num_res_blocks):
            dec_bot_layers.append(ResBlock(hidden_dim))
        dec_bot_layers.extend([
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),
        ])
        self.decoder_bot = nn.Sequential(*dec_bot_layers)
    
    def forward(self, x):
        """
        Forward pass through hierarchical VQ-VAE-2.
        
        Returns:
            recon: Reconstructed image
            vq_loss: Combined VQ loss from both levels
            (indices_bot, indices_top): Codebook indices
        """
        # Encode bottom
        h_bot = self.encoder_bot(x)
        z_e_bot = self.pre_vq_bot(h_bot)
        
        # Encode top (from bottom features)
        h_top = self.encoder_top(h_bot)
        z_e_top = self.pre_vq_top(h_top)
        
        # Quantize top
        z_q_top, vq_loss_top, indices_top = self.vq_top(z_e_top)
        
        # Decode top → bottom resolution
        top_decoded = self.decoder_top(z_q_top)
        
        # Quantize bottom
        z_q_bot, vq_loss_bot, indices_bot = self.vq_bot(z_e_bot)
        
        # Decode bottom (conditioned on top)
        combined = torch.cat([z_q_bot, top_decoded], dim=1)
        recon = self.decoder_bot(combined)
        
        vq_loss = vq_loss_top + vq_loss_bot
        
        return recon, vq_loss, (indices_bot, indices_top)
```

---

## Two-Stage Generation

### Stage 1: Train VQ-VAE-2

Train the hierarchical encoder-decoder to learn the codebooks and produce high-quality reconstructions.

### Stage 2: Train Autoregressive Priors

After training VQ-VAE-2, train separate autoregressive models (e.g., PixelSNAIL) over the discrete codes:

1. **Top prior** $p(z_{\text{top}})$: Models the top-level codes
2. **Bottom prior** $p(z_{\text{bot}} | z_{\text{top}})$: Models bottom codes conditioned on top codes

### Generation Pipeline

```python
def generate_vqvae2(model, top_prior, bottom_prior, device, num_samples=1):
    """
    Two-stage generation with VQ-VAE-2.
    
    1. Sample top codes from top prior
    2. Sample bottom codes from bottom prior (conditioned on top)
    3. Decode both levels to generate image
    """
    model.eval()
    
    with torch.no_grad():
        # Stage 1: Sample top-level codes
        top_codes = top_prior.sample(num_samples)  # [B, H_top, W_top]
        z_q_top = model.vq_top.embedding(top_codes)  # Look up embeddings
        
        # Stage 2: Sample bottom-level codes conditioned on top
        top_decoded = model.decoder_top(z_q_top.permute(0, 3, 1, 2))
        bottom_codes = bottom_prior.sample(num_samples, condition=top_decoded)
        z_q_bot = model.vq_bot.embedding(bottom_codes)
        
        # Decode
        combined = torch.cat([z_q_bot.permute(0, 3, 1, 2), top_decoded], dim=1)
        images = model.decoder_bot(combined)
    
    return images
```

---

## What Each Level Captures

### Top Level: Global Structure

The top-level codes, operating at low spatial resolution, capture global attributes such as object identity and category, overall shape and pose, color palette and lighting, and spatial layout.

### Bottom Level: Local Details

The bottom-level codes, operating at higher resolution, capture fine-grained textures, edge details and boundaries, local patterns and features, and high-frequency information.

This decomposition is what enables VQ-VAE-2 to produce significantly sharper and more coherent images than single-level VQ-VAE.

---

## VQ-VAE-2 vs VQ-VAE

| Aspect | VQ-VAE | VQ-VAE-2 |
|--------|--------|----------|
| **Hierarchy** | Single level | Multiple levels |
| **Global coherence** | Limited | Strong (top level) |
| **Local detail** | Single codebook | Dedicated bottom level |
| **Generation** | Single prior | Hierarchical priors |
| **Image quality** | Good | Near photorealistic |
| **Complexity** | Moderate | Higher |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Multi-scale quantization** | Top = global structure, bottom = local details |
| **Hierarchical decoding** | Top decoded first, conditions bottom decoder |
| **Two-stage generation** | Train VQ-VAE-2, then train autoregressive priors |
| **Quality improvement** | Significantly sharper than single-level VQ-VAE |

---

## Exercises

### Exercise 1: Two-Level Training

Implement and train VQ-VAE-2 on CIFAR-10. Compare reconstruction quality against single-level VQ-VAE.

### Exercise 2: Level Analysis

Reconstruct images using only top-level codes (zero out bottom) and only bottom-level codes (random top). What does each level contribute?

### Exercise 3: Codebook Utilization

Compare codebook utilization between top and bottom levels. Which has more "dead" codes?

---

## What's Next

The next section covers [Hierarchical VAEs](hierarchical.md) with continuous latent variables at multiple scales.
