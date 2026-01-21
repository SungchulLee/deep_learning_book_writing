# VAE Encoder-Decoder

## Introduction

The **VAE** (Variational Autoencoder) in Stable Diffusion compresses images to a lower-dimensional latent space where diffusion is performed.

## Architecture

### Encoder
- Input: 512×512×3 image
- Output: 64×64×4 latent
- Structure: ResNet blocks + downsampling

### Decoder
- Input: 64×64×4 latent
- Output: 512×512×3 image
- Structure: ResNet blocks + upsampling

## Key Properties

| Property | Value |
|----------|-------|
| Compression | 8× spatial |
| Latent channels | 4 |
| Loss | Reconstruction + KL + Perceptual |

## Training

Trained separately from diffusion:
1. Reconstruction loss (L1/L2)
2. KL divergence for regularization
3. Perceptual loss (LPIPS)
4. GAN loss for sharpness

## Summary

The VAE provides efficient compression while preserving visual quality, enabling diffusion in a manageable latent space.

## Navigation

- **Previous**: [Stable Diffusion Architecture](stable_diffusion.md)
- **Next**: [U-Net Denoiser](unet_denoiser.md)
