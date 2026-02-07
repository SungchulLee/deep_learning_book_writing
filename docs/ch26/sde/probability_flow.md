# Probability Flow ODE

## Introduction

The **Probability Flow ODE** provides a deterministic alternative to stochastic sampling. It shares the same marginal distributions as the reverse SDE but without randomness.

!!! success "Key Property"
    The probability flow ODE enables: (1) exact likelihood computation, (2) deterministic sampling, (3) latent space interpolation.

## Mathematical Formulation

### From SDE to ODE

For any SDE with the same marginals, there exists an ODE:

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

This has **no noise term** — it's fully deterministic!

### VP-SDE Case

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]$$

### Equivalence

**Theorem**: If $x(0) \sim p_0(x)$ and we integrate the ODE, then $x(t) \sim p_t(x)$ for all $t$.

The SDE and ODE define different sample paths but identical marginal distributions.

## Likelihood Computation

### Change of Variables

Using the instantaneous change of variables formula:

$$\log p_0(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot f_\theta(x_t, t) \, dt$$

where $f_\theta$ is the ODE drift and $\nabla \cdot$ is the divergence.

### Practical Computation

Using Hutchinson's trace estimator:

$$\nabla \cdot f = \mathbb{E}_\epsilon[\epsilon^\top \nabla_x (f^\top \epsilon)]$$

## PyTorch Implementation

```python
"""
Probability Flow ODE
====================
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple


class ProbabilityFlowODE:
    """Probability flow ODE for diffusion models."""
    
    def __init__(
        self,
        score_fn: Callable,
        beta_min: float = 0.1,
        beta_max: float = 20.0
    ):
        self.score_fn = score_fn
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ODE drift: f(x,t) - (1/2)g²∇log p."""
        beta_t = self.beta(t)
        score = self.score_fn(x, t)
        
        # f(x,t) = -β(t)x/2
        f = -0.5 * beta_t.view(-1, 1) * x
        
        # g(t)² = β(t)
        g_sq = beta_t
        
        return f - 0.5 * g_sq.view(-1, 1) * score
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        n_steps: int = 1000,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Sample using ODE integration (Euler method)."""
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps, 0, -1):
            t = torch.full((shape[0],), i / n_steps, device=device)
            x = x - self.drift(x, t) * dt
        
        return x
    
    def encode(
        self,
        x_0: torch.Tensor,
        n_steps: int = 1000
    ) -> torch.Tensor:
        """Encode data to latent space (forward ODE)."""
        x = x_0
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i / n_steps, device=x.device)
            x = x + self.drift(x, t) * dt
        
        return x
    
    def compute_log_likelihood(
        self,
        x_0: torch.Tensor,
        n_steps: int = 100,
        n_hutchinson: int = 1
    ) -> torch.Tensor:
        """
        Compute log p(x_0) using change of variables.
        """
        x = x_0.clone().requires_grad_(True)
        batch_size = x.shape[0]
        device = x.device
        
        # Accumulate log determinant
        log_det = torch.zeros(batch_size, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps, device=device)
            
            # Compute divergence via Hutchinson
            div = torch.zeros(batch_size, device=device)
            for _ in range(n_hutchinson):
                eps = torch.randn_like(x)
                
                with torch.enable_grad():
                    x_grad = x.requires_grad_(True)
                    drift = self.drift(x_grad, t)
                    
                    # Vector-Jacobian product
                    vjp = torch.autograd.grad(
                        (drift * eps).sum(),
                        x_grad,
                        create_graph=False
                    )[0]
                    
                    div += (vjp * eps).sum(dim=-1)
            
            div /= n_hutchinson
            log_det += div * dt
            
            # Step forward
            x = x + self.drift(x.detach(), t) * dt
        
        # Prior log probability
        log_prior = -0.5 * (x ** 2).sum(dim=-1) - 0.5 * x.shape[-1] * torch.log(torch.tensor(2 * 3.14159))
        
        return log_prior + log_det


def demonstrate_ode_vs_sde():
    """Compare ODE and SDE sampling."""
    import matplotlib.pyplot as plt
    
    # Simple score function
    def score_fn(x, t):
        sigma = 0.1 + t.view(-1, 1) * 2
        return -x / (sigma ** 2 + 0.1)
    
    ode = ProbabilityFlowODE(score_fn)
    
    # ODE samples (deterministic)
    torch.manual_seed(42)
    init = torch.randn(500, 2)
    
    samples_ode = ode.sample((500, 2), n_steps=500)
    
    # Visualize
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(init[:, 0], init[:, 1], alpha=0.3, s=5)
    plt.title('Initial (Prior)')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    plt.subplot(1, 2, 2)
    plt.scatter(samples_ode[:, 0], samples_ode[:, 1], alpha=0.5, s=5)
    plt.title('ODE Samples')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('ode_samples.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    demonstrate_ode_vs_sde()
```

## Advantages and Disadvantages

### Advantages

| Feature | Benefit |
|---------|---------|
| Deterministic | Reproducible outputs |
| Exact likelihood | Enables NLL evaluation |
| Encoding | Map data to latent space |
| Interpolation | Smooth latent trajectories |

### Disadvantages

| Feature | Issue |
|---------|-------|
| Sample quality | Sometimes worse than SDE |
| Numerical errors | Accumulate without noise correction |
| Speed | Similar to SDE for same steps |

## Connection to Normalizing Flows

The probability flow ODE defines a **continuous normalizing flow**:
- Invertible transformation: $x_0 \leftrightarrow x_T$
- Exact likelihood via change of variables
- But: learned implicitly via score matching, not explicit Jacobian

## Summary

The probability flow ODE provides:

1. **Deterministic sampling**: Same model, no randomness
2. **Exact likelihood**: Enable density evaluation
3. **Encoding capability**: Invert generation process
4. **Theoretical foundation**: Connection to normalizing flows

