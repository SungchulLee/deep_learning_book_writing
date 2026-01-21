# Chapter 14: Diffusion & Score-based Models

This chapter covers the theory and implementation of diffusion models, one of the most powerful approaches to generative modeling. We develop these models from first principles, connecting them to score matching, Bayesian inference, and stochastic differential equations.

## Overview

Diffusion models learn to generate data by reversing a gradual noising process. They have achieved state-of-the-art results in image generation, audio synthesis, and many other domains.

**Central Insight**: Denoising is equivalent to learning the score function $\nabla_x \log p(x)$, which enables sampling without computing intractable normalizing constants.

## Chapter Contents

### 14.1 Score-based Generative Models

The mathematical foundation connecting scores to generative modeling.

- [Score Function Definition](score_based/score_function.md) - The gradient of log-density
- [Score Matching](score_based/score_matching.md) - Learning scores without normalization
- [Denoising Score Matching](score_based/denoising_score_matching.md) - Practical score estimation
- [Sliced Score Matching](score_based/sliced_score_matching.md) - Random projections for efficiency
- [Noise Conditional Score Networks](score_based/ncsn.md) - Multi-scale score modeling

### 14.2 Diffusion Process

Understanding the forward (noise-adding) process.

- [Forward Diffusion Process](diffusion_process/forward_diffusion.md) - Gradually corrupting data
- [Noise Schedules](diffusion_process/noise_schedules.md) - Linear, cosine, and learned schedules
- [Variance Exploding vs Preserving](diffusion_process/ve_vs_vp.md) - VE-SDE and VP-SDE formulations
- [SDE Formulation](diffusion_process/sde_formulation.md) - Continuous-time perspective

### 14.3 Reverse Process

Learning to denoise and generate.

- [Reverse SDE](reverse_process/reverse_sde.md) - Generating data from noise
- [Score Function Learning](reverse_process/score_learning.md) - Neural network approximation
- [Probability Flow ODE](reverse_process/probability_flow_ode.md) - Deterministic generation path
- [Connection to Langevin Dynamics](reverse_process/langevin_connection.md) - Sampling theory unification

### 14.4 Denoising Diffusion Models

The DDPM framework and training.

- [DDPM](ddpm/ddpm.md) - Denoising Diffusion Probabilistic Models
- [DDPM Loss Function](ddpm/loss_function.md) - Training objectives and parameterizations
- [DDPM Training](ddpm/training.md) - Practical implementation details
- [DDPM Sampling](ddpm/sampling.md) - Step-by-step generation

### 14.5 Fast Sampling Methods

Accelerating generation from diffusion models.

- [DDIM](fast_sampling/ddim.md) - Deterministic sampling in fewer steps
- [Ancestral Sampling](fast_sampling/ancestral.md) - Stochastic generation variants
- [Progressive Distillation](fast_sampling/distillation.md) - Knowledge distillation for speed
- [Consistency Models](fast_sampling/consistency.md) - Direct mapping to clean data

### 14.6 Conditional Diffusion

Guiding generation toward desired outputs.

- [Classifier Guidance](conditional/classifier_guidance.md) - Using external classifiers
- [Classifier-Free Guidance](conditional/classifier_free.md) - Joint unconditional/conditional training
- [Text-to-Image](conditional/text_to_image.md) - CLIP and language conditioning

### 14.7 Latent Diffusion

Efficient high-resolution generation.

- [Latent Space Diffusion](latent_diffusion/latent_space.md) - VAE + diffusion combination
- [Stable Diffusion Architecture](latent_diffusion/stable_diffusion.md) - VAE + U-Net + CLIP
- [VAE Encoder-Decoder](latent_diffusion/vae_component.md) - Compression component
- [U-Net Denoiser](latent_diffusion/unet_denoiser.md) - The core neural network
- [CLIP Text Encoder](latent_diffusion/clip_encoder.md) - Text conditioning

### 14.8 Comparisons

Understanding when to use each generative approach.

- [Diffusion vs Normalizing Flows](comparisons/diffusion_vs_flows.md) - Flexibility vs exact likelihood
- [Diffusion vs GANs](comparisons/diffusion_vs_gans.md) - Stability vs speed
- [Diffusion vs VAEs](comparisons/diffusion_vs_vaes.md) - Quality vs simplicity

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

### Score-Noise Relationship
$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

### Training Loss
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### Reverse Mean
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

### Classifier-Free Guidance
$$\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))$$

## Prerequisites

- **Chapter 12-13**: Bayesian inference, MCMC, Langevin dynamics
- **Chapter 9**: Autoencoders and VAEs (for latent diffusion)
- **Chapter 11**: Normalizing flows (for comparisons)
- Probability theory (Gaussian distributions, Bayes' theorem)
- Neural network basics (CNNs, attention)
- PyTorch fundamentals

## Further Reading

### Foundational Papers

- **DDPM**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **Score-based**: "Generative Modeling by Estimating Gradients" (Song & Ermon, 2019)
- **Score SDE**: "Score-Based Generative Modeling through SDEs" (Song et al., 2021)
- **DDIM**: "Denoising Diffusion Implicit Models" (Song et al., 2020)
- **Latent Diffusion**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- **Consistency Models**: "Consistency Models" (Song et al., 2023)

### Tutorials and Resources

- Lilian Weng's blog on diffusion models
- Hugging Face Diffusers documentation
- Stanford CS236 course materials
