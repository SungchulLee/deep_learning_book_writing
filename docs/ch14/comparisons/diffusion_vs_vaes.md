# Diffusion vs VAEs

## Introduction

Diffusion models evolved from VAEs conceptually but differ significantly in practice.

## Comparison Table

| Aspect | Diffusion Models | VAEs |
|--------|------------------|------|
| **Latent space** | Fixed prior (Gaussian) | Learned encoder |
| **Latent dimension** | Same as data | Compressed |
| **Generation** | Iterative denoising | Single decode |
| **Sample quality** | State-of-the-art | Often blurry |
| **Training** | Simple MSE | ELBO (recon + KL) |
| **Interpolation** | Via DDIM inversion | Direct in latent |
| **Posterior collapse** | Not applicable | Common issue |

## Mathematical Connection

### VAE
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

### Diffusion (ELBO view)
$$\mathcal{L}_{\text{diffusion}} = -\mathbb{E}[\log p(x_0|x_1)] + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$$

Diffusion = VAE with:
- Many latent variables ($x_1, ..., x_T$)
- Fixed encoder (forward diffusion)
- Hierarchical structure

## Sample Quality

VAEs suffer from:
- **Blurry outputs**: MSE reconstruction encourages mean predictions
- **Posterior collapse**: Ignoring latent in powerful decoders

Diffusion avoids these:
- **Iterative refinement**: Gradually adds detail
- **No encoder**: Latent is just noise

## When to Use Each

### Diffusion
- Best sample quality needed
- Sufficient compute for generation
- Complex high-dimensional data

### VAEs
- Fast generation required
- Representation learning goal
- Simple/low-dimensional data
- Latent space manipulation

## Latent Diffusion: Best of Both

Combines VAE compression with diffusion quality:
1. VAE: Compress image to latent
2. Diffusion: Model latent distribution
3. VAE: Decode latent to image

## Summary

- **Diffusion**: Superior quality, slower generation
- **VAEs**: Fast, learned representations, lower quality
- **Latent Diffusion**: Combines strengths of both

## Navigation

- **Previous**: [Diffusion vs GANs](diffusion_vs_gans.md)
- **Next**: [Chapter 15 - Autoregressive Models](../../ch15/index.md)
