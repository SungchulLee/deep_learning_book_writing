# Diffusion vs GANs

## Introduction

Diffusion models and GANs represent different paradigms for generative modeling, with distinct trade-offs.

## Comparison Table

| Aspect | Diffusion Models | GANs |
|--------|------------------|------|
| **Training** | Stable (MSE loss) | Unstable (adversarial) |
| **Mode coverage** | Full distribution | May miss modes |
| **Sample quality** | Excellent | Excellent |
| **Diversity** | High | Can be limited |
| **Sampling speed** | Slow (iterative) | Fast (single pass) |
| **Likelihood** | Available (ODE) | Not available |
| **Conditioning** | Easy (CFG) | Requires careful design |

## Training Stability

### Diffusion
- Simple regression loss
- No min-max optimization
- Stable across hyperparameters
- No mode collapse

### GANs
- Adversarial training (min-max)
- Requires careful balancing
- Sensitive to architecture/hyperparameters
- Mode collapse risk

## Sample Quality Timeline

| Year | GAN Best FID | Diffusion Best FID |
|------|--------------|-------------------|
| 2019 | BigGAN: 7.0 | - |
| 2020 | StyleGAN2: 2.8 | DDPM: 3.2 |
| 2021 | StyleGAN2-ADA: 2.4 | ADM: 2.1 |
| 2022 | - | DiT: 2.3 |

## Use Cases

### Choose Diffusion When
- Training stability matters
- Need full distribution coverage
- Conditioning flexibility required
- Likelihood evaluation needed

### Choose GANs When
- Real-time generation needed
- Limited compute for inference
- Well-understood domain
- Specific GAN architecture exists (StyleGAN for faces)

## Hybrid Approaches

Some recent work combines both:
- **Denoising Diffusion GANs**: Use discriminator for faster sampling
- **GAN prior diffusion**: GAN provides base, diffusion refines

## Summary

- **Diffusion**: More stable, better coverage, slower
- **GANs**: Fast generation, may miss modes, harder to train

## Navigation

- **Previous**: [Diffusion vs Flows](diffusion_vs_flows.md)
- **Next**: [Diffusion vs VAEs](diffusion_vs_vaes.md)
