# DDPM Sampling

## Introduction

DDPM sampling iteratively denoises from pure noise to generate samples. This section covers the sampling algorithm and practical considerations.

## Sampling Algorithm

### Standard DDPM Sampling

```
Algorithm: DDPM Sampling
------------------------
Input: Trained model ε_θ, timesteps T
Output: Sample x_0

1. x_T ~ N(0, I)
2. for t = T, T-1, ..., 1:
3.   ε = ε_θ(x_t, t)              # Predict noise
4.   μ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε)  # Compute mean
5.   if t > 1:
6.     z ~ N(0, I)
7.     x_{t-1} = μ + σ_t z        # Add noise
8.   else:
9.     x_{t-1} = μ                # Final step: no noise
10. return x_0
```

### Variance Options

Two common choices for $\sigma_t$:

1. **Learned**: $\sigma_t^2 = \exp(v \log \beta_t + (1-v) \log \tilde{\beta}_t)$
2. **Fixed**: $\sigma_t = \sqrt{\beta_t}$ or $\sigma_t = \sqrt{\tilde{\beta}_t}$

where $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$

## PyTorch Implementation

```python
"""
DDPM Sampling
=============
"""

import torch
import torch.nn as nn
from typing import Optional, List
from tqdm import tqdm


class DDPMSampler:
    """DDPM sampling with various options."""
    
    def __init__(
        self,
        model: nn.Module,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model
        self.n_timesteps = n_timesteps
        self.device = device
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[:-1]
        ])
        
        # Precompute
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Posterior variance
        self.posterior_var = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Single denoising step."""
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        eps_pred = self.model(x_t, t_tensor)
        
        # Compute x_0 prediction (for clipping)
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        
        x_0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_cumprod
        
        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)
        
        # Compute mean
        coef1 = self.betas[t] * self.alphas_cumprod_prev[t].sqrt() / (1 - self.alphas_cumprod[t])
        coef2 = (1 - self.alphas_cumprod_prev[t]) * self.sqrt_alphas[t] / (1 - self.alphas_cumprod[t])
        
        mean = coef1 * x_0_pred + coef2 * x_t
        
        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = self.posterior_var[t].sqrt()
            return mean + sigma * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        return_trajectory: bool = False,
        show_progress: bool = True
    ) -> torch.Tensor:
        """Generate samples."""
        self.model.eval()
        
        # Start from noise
        x = torch.randn(shape, device=self.device)
        
        trajectory = [x.cpu()] if return_trajectory else None
        
        timesteps = range(self.n_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc='Sampling')
        
        for t in timesteps:
            x = self.p_sample(x, t)
            if return_trajectory and t % 100 == 0:
                trajectory.append(x.cpu())
        
        if return_trajectory:
            return x, trajectory
        return x
    
    @torch.no_grad()
    def sample_with_guidance(
        self,
        shape: tuple,
        guidance_fn: callable,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """Sample with external guidance."""
        x = torch.randn(shape, device=self.device)
        
        for t in tqdm(range(self.n_timesteps - 1, -1, -1)):
            # Enable gradients for guidance
            x = x.requires_grad_(True)
            
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Get guidance gradient
            with torch.enable_grad():
                guidance = guidance_fn(x, t)
                grad = torch.autograd.grad(guidance.sum(), x)[0]
            
            x = x.detach()
            
            # Regular denoising step
            eps_pred = self.model(x, t_tensor)
            
            # Modify noise prediction with guidance
            eps_pred = eps_pred - guidance_scale * self.sqrt_one_minus_alphas_cumprod[t] * grad
            
            # Compute mean and sample
            sqrt_alpha = self.alphas_cumprod[t].sqrt()
            x_0_pred = (x - self.sqrt_one_minus_alphas_cumprod[t] * eps_pred) / sqrt_alpha
            
            coef1 = self.betas[t] * self.alphas_cumprod_prev[t].sqrt() / (1 - self.alphas_cumprod[t])
            coef2 = (1 - self.alphas_cumprod_prev[t]) * self.sqrt_alphas[t] / (1 - self.alphas_cumprod[t])
            
            mean = coef1 * x_0_pred + coef2 * x
            
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + self.posterior_var[t].sqrt() * noise
            else:
                x = mean
        
        return x


def visualize_sampling():
    """Visualize the sampling process."""
    import matplotlib.pyplot as plt
    
    # Simple 2D example
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 64), nn.SiLU(),
                nn.Linear(64, 64), nn.SiLU(),
                nn.Linear(64, 2)
            )
        
        def forward(self, x, t):
            t_emb = t.float().unsqueeze(-1) / 100
            return self.net(torch.cat([x, t_emb], dim=-1))
    
    model = SimpleModel()
    sampler = DDPMSampler(model, n_timesteps=100)
    
    # Sample with trajectory
    samples, trajectory = sampler.sample((500, 2), return_trajectory=True)
    
    # Plot trajectory
    fig, axes = plt.subplots(1, len(trajectory), figsize=(3*len(trajectory), 3))
    
    for ax, (i, x) in zip(axes, enumerate(trajectory)):
        ax.scatter(x[:, 0], x[:, 1], alpha=0.5, s=5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Step {i}')
    
    plt.tight_layout()
    plt.savefig('sampling_trajectory.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    visualize_sampling()
```

## Sampling Speed

### Problem

Standard DDPM requires 1000 steps → slow generation

### Solutions

1. **DDIM**: Deterministic sampling with fewer steps
2. **Progressive distillation**: Train faster models
3. **Consistency models**: Single-step generation

## Summary

DDPM sampling:
1. **Start from noise**: $x_T \sim \mathcal{N}(0, I)$
2. **Iteratively denoise**: Use learned $\epsilon_\theta$
3. **Add stochasticity**: Noise at each step (except last)
4. **Slow but high-quality**: 1000 steps for best results

## Navigation

- **Previous**: [DDPM Training](training.md)
- **Next**: [DDIM](../fast_sampling/ddim.md)
