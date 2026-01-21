# Variance Exploding vs Variance Preserving

## Introduction

The **SDE framework** (Song et al., 2021) unifies different diffusion formulations under a common mathematical structure. Two primary variants are **Variance Exploding (VE)** and **Variance Preserving (VP)** SDEs.

## Variance Preserving (VP) SDE

### Formulation

The VP-SDE maintains approximately unit variance throughout:

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dw$$

where $\beta(t)$ is the noise schedule and $dw$ is the Wiener process.

### Properties

- **Variance bounded**: $\text{Var}[x_t] \approx 1$ for all $t$
- **Signal decay**: $x_0$ contribution shrinks as $e^{-\int_0^t \beta(s)/2 \, ds}$
- **DDPM connection**: Discrete VP-SDE yields the DDPM forward process

### Marginal Distribution

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

## Variance Exploding (VE) SDE

### Formulation

The VE-SDE lets variance grow unbounded:

$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dw$$

where $\sigma(t)$ is an increasing function.

### Properties

- **No signal shrinkage**: $x_0$ coefficient remains 1
- **Growing variance**: $\text{Var}[x_t] = \sigma^2(t)$
- **NCSN connection**: Corresponds to adding noise at multiple scales

### Marginal Distribution

$$q(x_t | x_0) = \mathcal{N}(x_t; x_0, \sigma^2(t) I)$$

## Comparison

| Aspect | VP-SDE | VE-SDE |
|--------|--------|--------|
| Signal scaling | Decreasing | Constant |
| Variance | Bounded (~1) | Exploding |
| Prior at $t=T$ | $\mathcal{N}(0, I)$ | $\mathcal{N}(0, \sigma_{\max}^2 I)$ |
| Training stability | More stable | May need normalization |
| DDPM/NCSN | DDPM | NCSN |

## PyTorch Implementation

```python
"""
VP-SDE and VE-SDE Implementations
=================================
"""

import torch
import numpy as np


class VPSDE:
    """Variance Preserving SDE."""
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def marginal_params(self, t: torch.Tensor):
        """Get mean coefficient and std for q(x_t|x_0)."""
        log_mean_coef = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coef = torch.exp(log_mean_coef)
        std = torch.sqrt(1 - mean_coef**2)
        return mean_coef, std
    
    def sample(self, x_0: torch.Tensor, t: torch.Tensor):
        """Sample from q(x_t|x_0)."""
        mean_coef, std = self.marginal_params(t)
        eps = torch.randn_like(x_0)
        x_t = mean_coef.view(-1, 1) * x_0 + std.view(-1, 1) * eps
        return x_t, eps


class VESDE:
    """Variance Exploding SDE."""
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Geometric sigma schedule."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def sample(self, x_0: torch.Tensor, t: torch.Tensor):
        """Sample from q(x_t|x_0)."""
        sigma = self.sigma(t)
        eps = torch.randn_like(x_0)
        x_t = x_0 + sigma.view(-1, 1) * eps
        return x_t, eps


def compare_sde_types():
    """Visualize VP vs VE behavior."""
    import matplotlib.pyplot as plt
    
    vp = VPSDE()
    ve = VESDE()
    
    x_0 = torch.ones(1, 2)
    times = torch.linspace(0, 1, 100)
    
    # Compute statistics
    vp_means, vp_stds = [], []
    ve_means, ve_stds = [], []
    
    for t in times:
        t_tensor = t.unsqueeze(0)
        mean_coef, std = vp.marginal_params(t_tensor)
        vp_means.append(mean_coef.item())
        vp_stds.append(std.item())
        ve_stds.append(ve.sigma(t_tensor).item())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(times, vp_means, label='VP mean coef')
    axes[0].axhline(1, linestyle='--', color='gray', label='VE mean coef (=1)')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Mean Coefficient')
    axes[0].set_title('Signal Scaling')
    axes[0].legend()
    
    axes[1].plot(times, vp_stds, label='VP std')
    axes[1].plot(times, ve_stds, label='VE std (σ)')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('Noise Level')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('ve_vs_vp.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    compare_sde_types()
```

## Which to Use?

### VP-SDE (DDPM-style)

- Default choice for most applications
- More stable training
- Natural unit-variance prior

### VE-SDE (NCSN-style)

- When signal preservation is important
- Multi-scale noise modeling
- Some audio applications

## Summary

Both formulations are mathematically equivalent under appropriate transformations, but have different numerical properties. VP-SDE is more common in practice due to its bounded variance and connection to DDPM.

## Navigation

- **Previous**: [Noise Schedules](noise_schedules.md)
- **Next**: [SDE Formulation](sde_formulation.md)
