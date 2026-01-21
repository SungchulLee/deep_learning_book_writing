# SDE Formulation

## Introduction

The **SDE (Stochastic Differential Equation) framework** provides a unified continuous-time perspective on diffusion models. This formulation, developed by Song et al. (2021), connects discrete diffusion to continuous stochastic processes.

!!! info "Key Benefit"
    The SDE framework enables: (1) unified analysis of different diffusion variants, (2) probability flow ODEs for deterministic sampling, (3) exact likelihood computation.

## Forward SDE

### General Form

The forward diffusion process is described by:

$$dx = f(x, t) \, dt + g(t) \, dw$$

where:
- $f(x, t)$: **Drift coefficient** (deterministic tendency)
- $g(t)$: **Diffusion coefficient** (noise magnitude)
- $dw$: **Wiener process** (Brownian motion increment)

### VP-SDE

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dw$$

- $f(x,t) = -\frac{1}{2}\beta(t)x$ (pulls toward origin)
- $g(t) = \sqrt{\beta(t)}$

### VE-SDE

$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dw$$

- $f(x,t) = 0$ (no drift)
- $g(t) = \sqrt{\frac{d[\sigma^2(t)]}{dt}}$

## Reverse-Time SDE

### Anderson's Theorem

The reverse of a diffusion SDE is also an SDE (Anderson, 1982):

$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)] \, dt + g(t) \, d\bar{w}$$

where $d\bar{w}$ is a reverse-time Wiener process and $\nabla_x \log p_t(x)$ is the **score function**.

!!! success "Key Insight"
    The reverse process requires only the score function—exactly what score matching teaches us to estimate!

## Probability Flow ODE

### Deterministic Alternative

There exists a deterministic ODE with the same marginals:

$$dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right] dt$$

This **probability flow ODE** enables:
- Deterministic sampling (no randomness)
- Exact likelihood computation
- Interpolation in latent space

## PyTorch Implementation

```python
"""
SDE Framework for Diffusion Models
==================================
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class SDE(ABC):
    """Abstract base class for SDEs."""
    
    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient f(x, t)."""
        pass
    
    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient g(t)."""
        pass
    
    def reverse_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor
    ) -> torch.Tensor:
        """Reverse SDE drift: f - g² ∇log p."""
        g = self.diffusion(t)
        return self.drift(x, t) - g**2 * score
    
    def ode_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor
    ) -> torch.Tensor:
        """Probability flow ODE drift: f - (1/2)g² ∇log p."""
        g = self.diffusion(t)
        return self.drift(x, t) - 0.5 * g**2 * score


class VPSDE(SDE):
    """Variance Preserving SDE."""
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.beta(t).view(-1, 1) * x
    
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))


class VESDE(SDE):
    """Variance Exploding SDE."""
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma(t)
        return sigma * torch.sqrt(
            2 * (torch.log(torch.tensor(self.sigma_max)) - 
                 torch.log(torch.tensor(self.sigma_min)))
        )


def euler_maruyama_step(
    sde: SDE,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    score_fn: callable = None,
    reverse: bool = False
) -> torch.Tensor:
    """
    Single Euler-Maruyama integration step.
    """
    if reverse and score_fn is not None:
        score = score_fn(x, t)
        drift = sde.reverse_drift(x, t, score)
    else:
        drift = sde.drift(x, t)
    
    g = sde.diffusion(t)
    noise = torch.randn_like(x)
    
    x_next = x + drift * dt + g.view(-1, 1) * torch.sqrt(torch.abs(torch.tensor(dt))) * noise
    return x_next


def probability_flow_step(
    sde: SDE,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    score_fn: callable
) -> torch.Tensor:
    """
    Single step of probability flow ODE (deterministic).
    """
    score = score_fn(x, t)
    drift = sde.ode_drift(x, t, score)
    return x + drift * dt


if __name__ == "__main__":
    # Demo
    vp_sde = VPSDE()
    x = torch.randn(10, 2)
    t = torch.ones(10) * 0.5
    
    print("VP-SDE drift shape:", vp_sde.drift(x, t).shape)
    print("VP-SDE diffusion:", vp_sde.diffusion(t))
```

## Numerical Integration

### Euler-Maruyama Method

For SDE $dx = f \, dt + g \, dw$:

$$x_{t+\Delta t} = x_t + f(x_t, t) \Delta t + g(t) \sqrt{\Delta t} \, z, \quad z \sim \mathcal{N}(0, I)$$

### Higher-Order Methods

- **Heun's method**: Predictor-corrector for better accuracy
- **Runge-Kutta**: For probability flow ODE

## Connection to Discrete Diffusion

The discrete DDPM process:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

Is the exact solution of the VP-SDE marginal distribution.

## Summary

The SDE framework provides:

1. **Unified theory**: VP, VE, and other variants under one framework
2. **Reverse formula**: Score function enables time reversal
3. **ODE alternative**: Deterministic sampling option
4. **Analysis tools**: Continuous-time probability theory

## Navigation

- **Previous**: [Variance Exploding vs Preserving](ve_vs_vp.md)
- **Next**: [Reverse SDE](../reverse_process/reverse_sde.md)
