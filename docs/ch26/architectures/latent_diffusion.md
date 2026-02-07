# Latent Diffusion Models

**Latent diffusion** (Rombach et al., 2022) performs the diffusion process in a compressed latent space rather than pixel space, dramatically reducing computational cost while maintaining—or improving—generation quality. This is the architecture underlying Stable Diffusion.

## Motivation

Running diffusion in pixel space at high resolution is expensive: a 512×512×3 image has ~786K dimensions. Most of these dimensions encode imperceptible high-frequency details. Latent diffusion compresses to a 64×64×4 latent space (a 48× reduction), concentrating the diffusion process on semantically meaningful structure.

## Architecture Overview

```
Text prompt → CLIP Text Encoder → text embeddings [77, 768]
                                         ↓
Noise z_T → U-Net / DiT (cross-attention) → denoised latent ẑ_0
                                         ↓
                              VAE Decoder → output image [512, 512, 3]
```

Three components, trained in stages:

### 1. VAE (Variational Autoencoder)

The VAE compresses images to and from the latent space:

**Encoder** $\mathcal{E}$: Maps image $x \in \mathbb{R}^{H \times W \times 3}$ to latent $z \in \mathbb{R}^{h \times w \times c}$ where $h = H/f$, $w = W/f$ with downsampling factor $f \in \{4, 8\}$.

**Decoder** $\mathcal{D}$: Reconstructs $\hat{x} = \mathcal{D}(z)$ from the latent.

The VAE is trained with a combination of reconstruction loss, KL regularisation, and perceptual (LPIPS) loss. A small KL weight ensures the latent space is smooth without collapsing it to a standard Gaussian. After training, the VAE is frozen.

| Configuration | Latent shape (512×512 input) | Compression |
|---------------|------------------------------|-------------|
| $f=4, c=3$ | 128×128×3 | 16× |
| $f=8, c=4$ | 64×64×4 | 48× |
| $f=8, c=16$ | 64×64×16 | 12× |

### 2. Text Encoder (CLIP)

CLIP's text encoder maps natural language prompts to a sequence of embeddings:

1. **Tokenise**: Text → token IDs (max 77 tokens)
2. **Embed**: Token IDs → dense vectors via transformer
3. **Output**: Sequence of embeddings $c \in \mathbb{R}^{77 \times 768}$

These embeddings condition the denoising network via cross-attention.

### 3. Denoising Network

A U-Net (or DiT) operates in the latent space, predicting noise $\epsilon_\theta(z_t, t, c)$ conditioned on timestep $t$ and text embeddings $c$. Text conditioning enters through **cross-attention** layers:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

where $Q = W_Q h$ (from spatial features), $K = W_K c$, $V = W_V c$ (from text embeddings).

## Training

Training proceeds in two stages:

**Stage 1: Train VAE.** Optimise the encoder-decoder on images with reconstruction + perceptual + KL losses. This typically requires ~100K–500K steps on a large image dataset.

**Stage 2: Train diffusion model.** Freeze the VAE. Encode all training images to latents. Train the denoising U-Net in latent space with the standard noise-prediction loss, conditioned on text via cross-attention.

The latent-space training loss is:

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t, c}\!\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]$$

where $z_0 = \mathcal{E}(x_0)$ and $z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$.

## Inference Pipeline

```python
"""Latent Diffusion Inference Pipeline (Conceptual)."""

import torch


def generate(
    text_encoder,
    unet,
    vae_decoder,
    scheduler,
    prompt: str,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    device: str = "cuda",
):
    """Generate an image from a text prompt.

    Args:
        text_encoder: CLIP text encoder.
        unet: Denoising U-Net in latent space.
        vae_decoder: VAE decoder (latent → image).
        scheduler: DDIM or other sampling scheduler.
        prompt: Text description.
        num_steps: Number of denoising steps.
        guidance_scale: CFG scale (higher = more faithful to prompt).
    """
    # Encode text
    text_emb = text_encoder(prompt)  # [1, 77, 768]
    null_emb = text_encoder("")      # Unconditional embedding

    # Sample initial noise in latent space
    z = torch.randn(1, 4, 64, 64, device=device)

    # Denoising loop with classifier-free guidance
    timesteps = scheduler.get_timesteps(num_steps)

    for t in timesteps:
        # Predict noise: conditional and unconditional
        eps_cond = unet(z, t, text_emb)
        eps_uncond = unet(z, t, null_emb)

        # Classifier-free guidance
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Scheduler step (DDIM, DPM-Solver, etc.)
        z = scheduler.step(z, eps, t)

    # Decode latent to image
    image = vae_decoder(z)  # [1, 3, 512, 512]
    return image
```

## Advantages over Pixel-Space Diffusion

**Computational efficiency.** Training and inference are 4–48× cheaper. A single GPU can train models that would require clusters in pixel space.

**Semantic compression.** The VAE's latent space captures perceptually meaningful features, so diffusion operates on semantic rather than pixel-level structure.

**Resolution flexibility.** The same latent-space model can be trained at one resolution and adapted to others by adjusting the VAE.

**Modular design.** Components (VAE, text encoder, denoiser) can be improved independently.

## Stable Diffusion Variants

| Version | Architecture | Key features |
|---------|-------------|--------------|
| SD 1.x | U-Net, CLIP ViT-L/14 | Original latent diffusion |
| SD 2.x | U-Net, OpenCLIP ViT-H | Larger text encoder |
| SDXL | Dual U-Net, dual text encoders | Higher quality, 1024×1024 |
| SD 3 | DiT (MMDiT), T5 + CLIP | Transformer backbone |

## References

1. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*.
2. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML*.
3. Esser, P., et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML*.
