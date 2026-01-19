# Chapter 14: Diffusion & Score-based Models

This chapter covers the theory and implementation of diffusion models, one of the most powerful approaches to generative modeling.

## Overview

Diffusion models learn to generate data by reversing a gradual noising process. They have achieved state-of-the-art results in image generation, audio synthesis, and many other domains.

## Chapter Contents

### 14.1 Score-based Generative Models

- [Score Function Definition](score_based/score_function.md) - The gradient of log-density
- [Score Matching](score_based/score_matching.md) - Learning scores without normalization
- [Denoising Score Matching](score_based/denoising_score_matching.md) - Practical score estimation

### 14.2 Diffusion Process

- [Forward Diffusion Process](diffusion_process/forward_diffusion.md) - Gradually corrupting data
- [Noise Schedules](diffusion_process/noise_schedules.md) - Linear, cosine, and learned schedules

### 14.3 Reverse Process

- [Reverse SDE](reverse_process/reverse_sde.md) - Generating data from noise

### 14.4 Denoising Diffusion Models

- [DDPM](ddpm/ddpm.md) - Denoising Diffusion Probabilistic Models
- [DDPM Loss Function](ddpm/loss_function.md) - Training objectives and parameterizations

### 14.5 Fast Sampling Methods

- [DDIM](fast_sampling/ddim.md) - Deterministic sampling in fewer steps

### 14.6 Conditional Diffusion

- [Classifier-Free Guidance](conditional/classifier_free.md) - Improving conditional generation

### 14.7 Latent Diffusion

- [Stable Diffusion Architecture](latent_diffusion/stable_diffusion.md) - VAE + U-Net + CLIP
- [U-Net Denoiser](latent_diffusion/unet_denoiser.md) - The core neural network

## Learning Path

### Foundations (Start Here)

1. **Score Function** - Understand what we're learning
2. **Denoising Score Matching** - How to learn it efficiently
3. **Forward Diffusion** - The data corruption process

### Core DDPM

4. **DDPM** - The complete framework
5. **Loss Function** - Training objectives
6. **Reverse SDE** - How generation works

### Advanced Topics

7. **DDIM** - Fast sampling
8. **Classifier-Free Guidance** - Better conditional generation
9. **Stable Diffusion** - Latent space diffusion

## Key Equations

### Forward Process
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

### Training Loss
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### Reverse Mean
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

### Classifier-Free Guidance
$$\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))$$

## Prerequisites

- Probability theory (Gaussian distributions, Bayes' theorem)
- Neural network basics (CNNs, attention)
- PyTorch fundamentals
- Basic understanding of VAEs (for latent diffusion)

## Further Reading

### Foundational Papers

- **DDPM**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **Score-based**: "Generative Modeling by Estimating Gradients" (Song & Ermon, 2019)
- **DDIM**: "Denoising Diffusion Implicit Models" (Song et al., 2020)
- **Latent Diffusion**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)

### Tutorials and Resources

- Lilian Weng's blog on diffusion models
- Hugging Face Diffusers documentation
- Stanford CS236 course materials
