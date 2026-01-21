# Diffusion vs Normalizing Flows

## Introduction

Both diffusion models and normalizing flows are likelihood-based generative models, but they differ significantly in architecture and training.

## Comparison Table

| Aspect | Diffusion Models | Normalizing Flows |
|--------|------------------|-------------------|
| **Transformation** | Gradual denoising | Single invertible map |
| **Architecture** | Free-form U-Net | Constrained (invertible) |
| **Likelihood** | ELBO or exact (ODE) | Exact |
| **Training** | Denoising objective | Max likelihood |
| **Sampling** | Iterative (10-1000 steps) | Single forward pass |
| **Sample quality** | State-of-the-art | Good but not SOTA |
| **Latent space** | Gaussian prior | Gaussian prior |

## Architectural Freedom

### Diffusion
- Any architecture that predicts noise/score
- No invertibility constraints
- Can use powerful networks (U-Net, Transformer)

### Flows
- Must be invertible with tractable Jacobian
- Limits expressiveness
- Coupling layers, autoregressive, residual flows

## Sample Quality vs Speed

| Model | FID (ImageNet) | Sampling Time |
|-------|----------------|---------------|
| Diffusion (1000 steps) | 2-3 | ~1 min |
| Diffusion (50 steps) | 4-5 | ~3 sec |
| Flow (Glow) | 45 | ~0.1 sec |
| Flow (Flow++) | 3.3 | ~1 sec |

## When to Use Each

### Diffusion
- State-of-the-art quality needed
- Sampling time not critical
- Complex data (images, audio)

### Normalizing Flows
- Fast inference required
- Exact likelihood needed
- Density estimation tasks
- Variational inference (VAE posteriors)

## Connection via Probability Flow ODE

Diffusion models with probability flow ODE define a continuous normalizing flow:
- Same marginals as SDE
- Deterministic transformation
- Enables likelihood computation

## Summary

- **Diffusion**: Better quality, slower sampling, flexible architecture
- **Flows**: Fast sampling, exact likelihood, architectural constraints

## Navigation

- **Previous**: [CLIP Text Encoder](../latent_diffusion/clip_encoder.md)
- **Next**: [Diffusion vs GANs](diffusion_vs_gans.md)
