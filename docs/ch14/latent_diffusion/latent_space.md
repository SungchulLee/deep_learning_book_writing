# Latent Space Diffusion

## Introduction

**Latent diffusion** performs the diffusion process in a compressed latent space rather than pixel space, dramatically reducing computational costs while maintaining quality.

!!! success "Key Innovation"
    Train diffusion in 4×-8× downsampled latent space, then decode to high-resolution images.

## Motivation

### Pixel Space Problems

- 512×512×3 = 786,432 dimensions
- Each U-Net forward pass processes entire image
- Attention scales as O(n²) with resolution

### Latent Space Solution

- 64×64×4 = 16,384 dimensions (48× smaller!)
- Same quality with fraction of compute
- Enables high-resolution generation

## Architecture

```
Image (512×512×3) → VAE Encoder → Latent (64×64×4)
                                       ↓
                               Diffusion in Latent
                                       ↓
Latent (64×64×4) → VAE Decoder → Image (512×512×3)
```

## Components

### 1. VAE Encoder-Decoder

Compresses images to/from latent space:
- Encoder: Image → Latent
- Decoder: Latent → Image
- Trained separately, frozen during diffusion

### 2. Diffusion U-Net

Operates entirely in latent space:
- Input: Noisy latent z_t
- Output: Predicted noise ε
- Conditioned on timestep and text

### 3. Text Encoder

CLIP for text conditioning (see text-to-image section).

## Training

```python
"""
Latent Diffusion Training
=========================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDiffusion:
    """Latent diffusion model."""
    
    def __init__(self, vae, unet, text_encoder, scheduler, device):
        self.vae = vae.to(device)
        self.unet = unet.to(device)
        self.text_encoder = text_encoder.to(device)
        self.scheduler = scheduler
        self.device = device
        
        # Freeze VAE
        self.vae.requires_grad_(False)
    
    def encode(self, x):
        """Encode image to latent."""
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * 0.18215  # Scaling factor
        return latent
    
    def decode(self, z):
        """Decode latent to image."""
        z = z / 0.18215
        with torch.no_grad():
            image = self.vae.decode(z).sample
        return image
    
    def training_step(self, images, prompts):
        """Single training step."""
        # Encode images to latent
        latents = self.encode(images)
        
        # Encode text
        text_emb = self.text_encoder(prompts)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        t = torch.randint(0, 1000, (len(images),), device=self.device)
        
        # Add noise
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, t, text_emb)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
```

## Advantages

| Aspect | Pixel Diffusion | Latent Diffusion |
|--------|-----------------|------------------|
| Memory | ~24GB | ~8GB |
| Speed | 1× | 4-8× faster |
| Resolution | Limited | High-res capable |
| Quality | Good | Comparable |

## Summary

Latent diffusion enables efficient high-resolution generation by:
1. **Compressing** images via pretrained VAE
2. **Diffusing** in low-dimensional latent space
3. **Decoding** final latents to images

## Navigation

- **Previous**: [Text-to-Image](../conditional/text_to_image.md)
- **Next**: [Stable Diffusion Architecture](stable_diffusion.md)
