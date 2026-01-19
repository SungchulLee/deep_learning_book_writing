# Denoising Diffusion Probabilistic Models (DDPM)

**DDPM** (Ho et al., 2020) is the foundational work that demonstrated diffusion models can achieve high-quality image generation competitive with GANs.

## Overview

DDPM combines three key ideas:

1. A **forward process** that gradually corrupts data with Gaussian noise
2. A **reverse process** modeled by a neural network
3. A **simplified training objective** that predicts the noise added at each step

## The Forward Process

Starting from data $x_0 \sim q(x_0)$, define:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I)
$$

where $\alpha_t = 1 - \beta_t$ and $\{\beta_t\}_{t=1}^T$ is a fixed variance schedule.

### Key Property: Direct Sampling

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
$$

Equivalently:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

## The Reverse Process

DDPM learns to reverse the forward process:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)
$$

where:
- $p(x_T) = \mathcal{N}(0, I)$ (pure noise prior)
- $p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$

## The True Posterior

When $x_0$ is known, the true reverse transition has a closed form:

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

### Posterior Mean

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

### Posterior Variance

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

## Noise Prediction Parameterization

Instead of directly predicting $\mu_\theta$, DDPM predicts the **noise** $\epsilon_\theta(x_t, t)$.

### Reconstructing $x_0$

From $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$:

$$
\hat{x}_0(x_t, t) = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)\right)
$$

### Computing the Mean

Substituting into the posterior mean formula:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)
$$

This is the **DDPM mean parameterization**.

## Training Objective

### Variational Lower Bound

The ELBO for diffusion models decomposes as:

$$
\mathcal{L}_{\text{VLB}} = \mathcal{L}_0 + \mathcal{L}_1 + \cdots + \mathcal{L}_{T-1} + \mathcal{L}_T
$$

where each $\mathcal{L}_t$ is a KL divergence between Gaussians.

### Simplified Objective

Ho et al. showed that a simple MSE loss on noise prediction works better:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

This is equivalent to:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$

### Why Simplified Works

The simplified objective:
1. Drops per-timestep weighting from the VLB
2. Emphasizes perceptually important noise levels
3. Provides more stable gradients

## Training Algorithm

```
Algorithm: DDPM Training
─────────────────────────
repeat:
    x_0 ~ q(x_0)                          # Sample data
    t ~ Uniform({1, ..., T})               # Sample timestep
    ε ~ N(0, I)                           # Sample noise
    
    x_t = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε   # Noisy sample
    
    loss = ||ε - ε_θ(x_t, t)||²           # Predict noise
    
    θ ← θ - η * ∇_θ loss                  # Update parameters
until converged
```

## Sampling Algorithm

```
Algorithm: DDPM Sampling
────────────────────────
x_T ~ N(0, I)                             # Start from noise

for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else z = 0
    
    ε̂ = ε_θ(x_t, t)                       # Predict noise
    
    μ_θ = (1/sqrt(α_t)) * (x_t - (1-α_t)/sqrt(1-α̅_t) * ε̂)
    
    x_{t-1} = μ_θ + sqrt(β̃_t) * z         # Sample

return x_0
```

## Implementation

```python
import torch
import torch.nn as nn

class DDPM:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.T = T
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Posterior variance
        self.beta_tildes = (1 - self.alpha_bars[:-1]) / (1 - self.alpha_bars[1:]) * self.betas[1:]
        self.beta_tildes = torch.cat([self.betas[:1], self.beta_tildes])
    
    def training_loss(self, x_0):
        """Compute simplified DDPM loss."""
        batch_size = x_0.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Create noisy samples
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, shape, device):
        """Generate samples via reverse diffusion."""
        # Start from noise
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_batch)
            
            # Compute mean
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            
            mu = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred
            )
            
            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.beta_tildes[t])
                x = mu + sigma * noise
            else:
                x = mu
        
        return x
```

## Network Architecture

DDPM uses a **U-Net** with:

1. **Time embedding**: Sinusoidal position encoding for timestep $t$
2. **ResNet blocks**: Skip connections within encoder/decoder
3. **Self-attention**: At lower resolutions (16×16, 8×8)
4. **GroupNorm**: Instead of BatchNorm

### Time Embedding

$$
\text{emb}(t) = [\sin(t \cdot \omega_1), \cos(t \cdot \omega_1), \ldots, \sin(t \cdot \omega_{d/2}), \cos(t \cdot \omega_{d/2})]
$$

where $\omega_k = 1 / 10000^{2k/d}$ (similar to Transformer positional encoding).

## Hyperparameters

| Parameter | DDPM Value | Notes |
|-----------|------------|-------|
| $T$ | 1000 | Number of diffusion steps |
| $\beta_1$ | $10^{-4}$ | Initial noise variance |
| $\beta_T$ | 0.02 | Final noise variance |
| Schedule | Linear | $\beta_t$ interpolation |
| Variance | Fixed $\tilde{\beta}_t$ | Not learned |

## Results and Impact

DDPM achieved:
- **CIFAR-10**: FID 3.17 (competitive with GANs)
- **LSUN 256×256**: High-quality bedroom/church images
- Stable training without adversarial dynamics

## Limitations

1. **Slow sampling**: 1000 steps per image
2. **Fixed variance**: $\sigma_t$ not learned
3. **No conditioning**: Original DDPM is unconditional

These limitations motivated follow-up works like DDIM, Improved DDPM, and classifier-free guidance.

## Summary

DDPM established the modern framework for diffusion models: a simple noise prediction objective combined with a U-Net architecture. The key insight is that predicting the noise $\epsilon$ at each step is equivalent to learning the reverse diffusion process. Despite slow sampling, DDPM's stable training and high-quality outputs made it the foundation for subsequent advances like Stable Diffusion.
