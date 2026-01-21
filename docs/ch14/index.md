# Chapter 14: Diffusion and Score-Based Models

This chapter provides a comprehensive treatment of **diffusion models** and **score-based generative models**, representing the state-of-the-art in generative modeling. We develop these models from first principles, connecting them to Bayesian inference, Langevin dynamics, and stochastic differential equations.

## Overview

Diffusion models learn to generate data by reversing a gradual noising process. They have achieved state-of-the-art results in image generation, audio synthesis, and many other domains.

**Central Insight**: Denoising is equivalent to learning the score function $\nabla_x \log p(x)$, which enables sampling without computing intractable normalizing constants.

## Chapter Contents

### 14.1 Foundations and Intuition

Start here for high-level understanding before diving into mathematics.

- [What is a Diffusion Model?](foundations/overview.md) - Core idea, intuition, and why they matter
- [Diffusion vs Normalizing Flows](foundations/diffusion_vs_flows_intuition.md) - Key differences, tradeoffs, when to use each

### 14.2 Score-Based Generative Models

The mathematical foundation connecting scores to generative modeling.

- [Score Function Definition](score_based/score_function.md) - The gradient of log-density
- [Score Matching](score_based/score_matching.md) - Learning scores without normalization
- [Denoising Score Matching](score_based/denoising_score_matching.md) - Practical score estimation
- [Sliced Score Matching](score_based/sliced_score_matching.md) - Random projections for efficiency
- [Noise Conditional Score Networks](score_based/ncsn.md) - Multi-scale score modeling

### 14.3 Forward Diffusion Process

Understanding the noise-adding process.

- [Forward Process Fundamentals](diffusion_process/forward_diffusion.md) - Markov chain, marginals, reparameterization
- [Noise Schedules](diffusion_process/noise_schedules.md) - Linear, cosine, and learned schedules
- [VE vs VP SDEs](diffusion_process/ve_vs_vp.md) - Variance Exploding and Preserving formulations
- [SDE Formulation](diffusion_process/sde_formulation.md) - Continuous-time perspective

### 14.4 Reverse Process

Learning to denoise and generate.

- [Reverse SDE](reverse_process/reverse_sde.md) - Anderson's theorem, score-based reverse drift
- [Reverse Mean Derivation](reverse_process/reverse_mean.md) - Why the formula has that form
- [Posterior Computation](reverse_process/posterior_computation.md) - Product of Gaussians derivation
- [Score Learning](reverse_process/score_learning.md) - Neural network approximation
- [Probability Flow ODE](reverse_process/probability_flow_ode.md) - Deterministic alternative
- [Langevin Connection](reverse_process/langevin_connection.md) - Unifying sampling perspective

### 14.5 Denoising Diffusion Probabilistic Models

The DDPM framework and training.

- [DDPM Overview](ddpm/ddpm.md) - Complete framework, Ho et al. 2020
- [Loss Function](ddpm/loss_function.md) - VLB derivation, simple loss
- [Training Details](ddpm/training.md) - EMA, gradient clipping, hyperparameters
- [Sampling Algorithms](ddpm/sampling.md) - Iterative denoising, variance choices

### 14.6 Fast Sampling Methods

Accelerating generation from diffusion models.

- [DDIM](fast_sampling/ddim.md) - Non-Markovian, deterministic sampling, step skipping
- [Ancestral Sampling](fast_sampling/ancestral.md) - Original DDPM sampling
- [Progressive Distillation](fast_sampling/distillation.md) - Knowledge distillation for speed
- [Consistency Models](fast_sampling/consistency.md) - Direct mapping to clean data

### 14.7 Conditional Generation

Guiding diffusion models toward desired outputs.

- [Classifier Guidance](conditional/classifier_guidance.md) - External classifier gradients
- [Classifier-Free Guidance](conditional/classifier_free.md) - Joint training approach
- [Text-to-Image](conditional/text_to_image.md) - CLIP conditioning

### 14.8 Latent Diffusion Models

Efficient high-resolution generation.

- [Latent Space Diffusion](latent_diffusion/latent_space.md) - VAE + diffusion combination
- [Stable Diffusion](latent_diffusion/stable_diffusion.md) - Complete architecture
- [Training Stable Diffusion](latent_diffusion/training_stable_diffusion.md) - Procedure and compute
- [VAE Component](latent_diffusion/vae_component.md) - Encoder-decoder details
- [U-Net Denoiser](latent_diffusion/unet_denoiser.md) - Architecture deep dive
- [CLIP Text Encoder](latent_diffusion/clip_encoder.md) - Text embedding

### 14.9 Model Comparisons

Understanding tradeoffs between generative approaches.

- [Diffusion vs Normalizing Flows](comparisons/diffusion_vs_flows.md) - Detailed comparison
- [Diffusion vs GANs](comparisons/diffusion_vs_gans.md) - Stability, quality, speed
- [Diffusion vs VAEs](comparisons/diffusion_vs_vaes.md) - Mathematical connections

## Learning Path

### Level 1: Foundations (Start Here!)
1. **Overview** - What is a diffusion model?
2. **Diffusion vs Flows** - Key differences
3. **Forward Diffusion** - How noise corrupts data

### Level 2: Core Theory
4. **Score Function** - Mathematical foundation
5. **Denoising Score Matching** - How to learn scores
6. **Reverse Process** - How generation works

### Level 3: DDPM
7. **DDPM** - Complete framework
8. **Loss Function** - Training objective
9. **Posterior Computation** - The math behind it

### Level 4: Advanced Topics
10. **DDIM** - Fast sampling
11. **Classifier-Free Guidance** - Better conditioning
12. **Stable Diffusion** - Latent space approach

## Key Equations

### Forward Process
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

### Score-Noise Relationship
$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

### Training Loss
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### Reverse Mean
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

### Posterior Mean
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

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
- **Improved DDPM**: "Improved Denoising Diffusion" (Nichol & Dhariwal, 2021)
- **Classifier-Free**: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- **Latent Diffusion**: "High-Resolution Image Synthesis with LDMs" (Rombach et al., 2022)
- **Consistency**: "Consistency Models" (Song et al., 2023)

### Tutorials and Resources
- Lilian Weng's blog on diffusion models
- Hugging Face Diffusers documentation
- Stanford CS236 course materials
