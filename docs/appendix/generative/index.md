# A7 Generative Models

## Overview

This appendix provides complete PyTorch implementations of deep generative model architectures spanning five major paradigms: autoencoders, variational autoencoders (VAEs), generative adversarial networks (GANs), diffusion models, and normalizing flows, plus Neural ODEs as a continuous-depth generative framework. Each model learns to represent and sample from complex data distributions, enabling synthetic data generation, density estimation, and latent space manipulation. In quantitative finance, generative models are applied to scenario generation, data augmentation for rare events, and distribution modeling for risk management.

## Architectures

### Autoencoders and Variational Autoencoders

| Model | Year | Key Innovation |
|-------|------|----------------|
| [Autoencoder](autoencoder.py) | 1986 | Deterministic encoder–decoder for representation learning |
| [VAE](vae.py) | 2013 | Variational inference with reparameterization trick |
| [Beta-VAE](beta_vae.py) | 2017 | Disentangled representations via weighted KL penalty |
| [VQ-VAE](vqvae.py) | 2017 | Discrete latent codes via vector quantization |

### Generative Adversarial Networks

| Model | Year | Key Innovation |
|-------|------|----------------|
| [GAN](gan.py) | 2014 | Adversarial training: generator vs. discriminator |
| [DCGAN](dcgan.py) | 2015 | Stable convolutional GAN architecture guidelines |
| [StyleGAN](stylegan.py) | 2019 | Style-based generator with progressive growing |
| [Pix2Pix](pix2pix.py) | 2017 | Paired image-to-image translation with conditional GAN |
| [CycleGAN](cyclegan.py) | 2017 | Unpaired image translation via cycle consistency |

### Diffusion Models

| Model | Year | Key Innovation |
|-------|------|----------------|
| [DDPM](ddpm.py) | 2020 | Denoising diffusion with learned reverse process |
| [Stable Diffusion](stable_diffusion.py) | 2022 | Latent-space diffusion with text conditioning |

### Normalizing Flows

| Model | Year | Key Innovation |
|-------|------|----------------|
| [RealNVP](realnvp.py) | 2016 | Affine coupling layers for tractable density estimation |
| [Glow](glow.py) | 2018 | Invertible 1×1 convolutions, actnorm |

### Continuous-Depth Models

| Model | Year | Key Innovation |
|-------|------|----------------|
| [Neural ODE](neural_ode.py) | 2018 | ODE solver replaces discrete layers, continuous normalizing flows |

## Key Concepts

### Generative Model Taxonomy

| Paradigm | Density | Training | Sample Quality | Latent Space |
|----------|---------|----------|----------------|--------------|
| Autoencoder | None (deterministic) | Reconstruction | N/A | Continuous, unstructured |
| VAE | Approximate (ELBO) | Stable | Moderate | Continuous, structured |
| GAN | Implicit | Adversarial (unstable) | High | Continuous |
| Diffusion | Tractable (variational) | Stable | State-of-the-art | Iterative denoising |
| Flow | Exact (change of variables) | Stable | Good | Invertible mapping |
| Neural ODE | Exact (continuous) | Stable | Good | Continuous dynamics |

### Core Mathematical Frameworks

- **Autoencoder**: Minimize reconstruction loss: $\mathcal{L} = \|x - \hat{x}\|^2$
- **VAE**: Maximize the evidence lower bound (ELBO): $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))$
- **GAN**: Minimax game: $\min_G \max_D \, \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$
- **Diffusion**: Learn to reverse a Markov noising process: $p_\theta(x_{t-1}|x_t)$
- **Flow**: Exact log-likelihood via change of variables: $\log p(x) = \log p(f(x)) + \log|\det J_f(x)|$
- **Neural ODE**: Continuous dynamics: $\frac{dz}{dt} = f_\theta(z(t), t)$, solved with adaptive ODE integrators

### Training Considerations

- **Mode collapse** (GANs): Generator produces limited variety; mitigated by spectral normalization, progressive training
- **Posterior collapse** (VAEs): Decoder ignores latent code; addressed by KL annealing, free bits
- **Sampling speed** (Diffusion): Thousands of denoising steps; accelerated by DDIM, DPM-Solver
- **Expressiveness** (Flows): Coupling layers must be sufficiently flexible; multi-scale architectures help
- **Adjoint method** (Neural ODE): Memory-efficient backpropagation through ODE solver

## Quantitative Finance Applications

- **Scenario generation**: VAEs and flows for generating plausible market scenarios for stress testing and VaR estimation
- **Synthetic data augmentation**: Generate rare-event samples (crashes, tail risks) to improve model robustness
- **Distribution modeling**: Normalizing flows for exact density estimation of asset return distributions
- **Time series generation**: Diffusion models for generating realistic synthetic financial time series
- **Domain adaptation**: CycleGAN for adapting models across market regimes or asset classes
- **Anomaly detection**: Reconstruction-based detection using autoencoder, VAE, or flow likelihoods
- **Continuous dynamics**: Neural ODEs for modeling continuous-time financial processes (stochastic volatility, term structure evolution)

## Prerequisites

- [Ch4: Training Deep Networks](../../ch04/index.md) — optimization, regularization
- [A1: Classic CNNs](../cnn/index.md) — convolutional generators and discriminators
- [A10: Utility Modules — Loss Functions](../utils/losses.py) — reconstruction losses, adversarial losses
- [A10: Utility Modules — Normalization Layers](../utils/normalization.py) — batch norm, spectral norm, actnorm
