# Score-Based Stochastic Differential Equations

## Learning Objectives

By the end of this section, you will be able to:

1. Formulate score-based generative modeling as SDEs
2. Derive the reverse-time SDE from the forward process
3. Understand Variance Exploding (VE) and Variance Preserving (VP) formulations
4. Implement the probability flow ODE for deterministic sampling
5. Connect discrete diffusion models to continuous SDEs

## Prerequisites

- NCSN and noise scheduling
- Basic stochastic calculus (optional but helpful)
- ODE/SDE numerical solvers

---

## 1. From Discrete to Continuous

### 1.1 Discrete Noise Levels

NCSN uses discrete noise levels $\{\sigma_1 > \sigma_2 > \cdots > \sigma_L\}$. As $L \to \infty$:

$$
\text{Discrete steps} \to \text{Continuous diffusion process}
$$

### 1.2 The Continuous Framework

Instead of discrete noise levels, consider a continuous-time process:

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}
$$

where:
- $\mathbf{f}(\mathbf{x}, t)$ is the drift coefficient
- $g(t)$ is the diffusion coefficient
- $d\mathbf{w}$ is a standard Wiener process (Brownian motion)

---

## 2. Forward SDE

### 2.1 General Form

The forward process gradually adds noise to data $\mathbf{x}_0 \sim p_{\text{data}}$:

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}, \quad t \in [0, T]
$$

At time $T$, $\mathbf{x}_T$ is approximately pure noise.

### 2.2 Variance Exploding (VE) SDE

**Definition:**
$$
d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, d\mathbf{w}
$$

**Properties:**
- $\mathbf{f}(\mathbf{x}, t) = \mathbf{0}$ (no drift)
- Variance grows: $\text{Var}[\mathbf{x}_t] = \text{Var}[\mathbf{x}_0] + \sigma^2(t)$
- Corresponds to NCSN/SMLD

**Typical schedule:** $\sigma(t) = \sigma_{\min} (\sigma_{\max}/\sigma_{\min})^t$

### 2.3 Variance Preserving (VP) SDE

**Definition:**
$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} \, dt + \sqrt{\beta(t)} \, d\mathbf{w}
$$

**Properties:**
- Has drift toward origin: $\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}$
- Variance is preserved: if $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, then $\mathbf{x}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Corresponds to DDPM

**Typical schedule:** Linear $\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$

### 2.4 Sub-VP SDE

**Definition:**
$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} \, dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})} \, d\mathbf{w}
$$

**Properties:**
- Bounded variance: always $\leq 1$
- Better numerical stability than VP

---

## 3. Reverse SDE

### 3.1 Anderson's Theorem

A forward SDE can be reversed in time! The reverse SDE is:

$$
\boxed{d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t) d\bar{\mathbf{w}}}
$$

where $d\bar{\mathbf{w}}$ is a reverse-time Wiener process and $p_t(\mathbf{x})$ is the marginal at time $t$.

### 3.2 The Score Function Appears!

The reverse SDE requires $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ — the **score function** at time $t$!

This is exactly what NCSN learns: $\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$

### 3.3 Reverse SDEs for VE and VP

**VE Reverse:**
$$
d\mathbf{x} = -\frac{d[\sigma^2(t)]}{dt} \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \, dt + \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, d\bar{\mathbf{w}}
$$

**VP Reverse:**
$$
d\mathbf{x} = \left[-\frac{1}{2}\beta(t)\mathbf{x} - \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + \sqrt{\beta(t)} \, d\bar{\mathbf{w}}
$$

---

## 4. Probability Flow ODE

### 4.1 Deterministic Alternative

Song et al. (2021) showed that the reverse SDE has an equivalent **ODE**:

$$
\boxed{\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})}
$$

This ODE:
- Has **no stochasticity**
- Produces the **same marginal distributions** as the SDE
- Enables **exact likelihood computation**

### 4.2 VE Probability Flow

$$
\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\frac{d[\sigma^2(t)]}{dt} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

### 4.3 VP Probability Flow

$$
\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\beta(t)\left[\mathbf{x} + \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]
$$

---

## 5. Training Objective

### 5.1 Continuous-Time DSM

The training objective generalizes DSM to continuous time:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, T)} \mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{\mathbf{x}_t | \mathbf{x}_0} \left[\lambda(t) \|\mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t | \mathbf{x}_0)\|^2\right]
$$

### 5.2 Transition Kernels

For VE-SDE: $p_{0t}(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t | \mathbf{x}_0, \sigma^2(t)\mathbf{I})$

For VP-SDE: $p_{0t}(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t | \alpha(t)\mathbf{x}_0, (1 - \alpha^2(t))\mathbf{I})$

where $\alpha(t) = e^{-\frac{1}{2}\int_0^t \beta(s)ds}$

### 5.3 Loss Weighting

Common choices:
- $\lambda(t) = \sigma^2(t)$ (VE)
- $\lambda(t) = 1 - \alpha^2(t)$ (VP)
- $\lambda(t) = 1$ (uniform)

---

## 6. PyTorch Implementation

### 6.1 SDE Base Class

```python
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class SDE(ABC):
    """Base class for SDEs."""
    
    def __init__(self, T: float = 1.0):
        self.T = T
    
    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient f(x, t)."""
        pass
    
    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient g(t)."""
        pass
    
    @abstractmethod
    def marginal_params(self, x_0: torch.Tensor, t: torch.Tensor):
        """Parameters of p(x_t | x_0)."""
        pass
    
    def sample_prior(self, shape: tuple, device: str = 'cpu') -> torch.Tensor:
        """Sample from p(x_T)."""
        return torch.randn(shape, device=device)
```

### 6.2 Variance Exploding SDE

```python
class VESDE(SDE):
    """Variance Exploding SDE (NCSN/SMLD style)."""
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, T: float = 1.0):
        super().__init__(T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise level at time t."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """VE has zero drift."""
        return torch.zeros_like(x)
    
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = σ(t) * sqrt(2 * log(σ_max/σ_min))."""
        sigma = self.sigma(t)
        return sigma * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
    
    def marginal_params(self, x_0: torch.Tensor, t: torch.Tensor):
        """p(x_t | x_0) = N(x_0, σ²(t) I)."""
        sigma = self.sigma(t)
        mean = x_0
        std = sigma
        return mean, std
    
    def sample_prior(self, shape: tuple, device: str = 'cpu') -> torch.Tensor:
        return torch.randn(shape, device=device) * self.sigma_max
```

### 6.3 Variance Preserving SDE

```python
class VPSDE(SDE):
    """Variance Preserving SDE (DDPM style)."""
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        super().__init__(T)
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α(t) = exp(-0.5 * ∫₀ᵗ β(s) ds)."""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x, t) = -0.5 * β(t) * x."""
        return -0.5 * self.beta(t).view(-1, 1) * x
    
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = sqrt(β(t))."""
        return torch.sqrt(self.beta(t))
    
    def marginal_params(self, x_0: torch.Tensor, t: torch.Tensor):
        """p(x_t | x_0) = N(α(t) x_0, (1 - α²(t)) I)."""
        alpha = self.alpha(t)
        mean = alpha.view(-1, 1) * x_0
        std = torch.sqrt(1 - alpha ** 2)
        return mean, std
```

### 6.4 Score Network Training

```python
def score_sde_loss(
    score_model: nn.Module,
    sde: SDE,
    x_0: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Continuous-time score matching loss.
    
    Args:
        score_model: Network s_θ(x, t)
        sde: SDE defining the forward process
        x_0: Clean data samples
        eps: Small value to avoid t=0
    
    Returns:
        Scalar loss
    """
    N = x_0.shape[0]
    device = x_0.device
    
    # Sample random time
    t = torch.rand(N, device=device) * (sde.T - eps) + eps
    
    # Get marginal distribution parameters
    mean, std = sde.marginal_params(x_0, t)
    
    # Sample x_t ~ p(x_t | x_0)
    noise = torch.randn_like(x_0)
    x_t = mean + std.view(-1, 1) * noise
    
    # Predict score
    pred_score = score_model(x_t, t)
    
    # Target score: ∇ log p(x_t | x_0) = -noise / std
    target_score = -noise / std.view(-1, 1)
    
    # MSE loss (optionally weighted)
    loss = torch.mean(torch.sum((pred_score - target_score) ** 2, dim=1))
    
    return loss
```

### 6.5 Probability Flow ODE Sampler

```python
from scipy.integrate import solve_ivp

@torch.no_grad()
def probability_flow_sample(
    score_model: nn.Module,
    sde: SDE,
    shape: tuple,
    n_steps: int = 1000,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample using probability flow ODE.
    
    dx/dt = f(x,t) - 0.5 * g(t)² * ∇log p_t(x)
    """
    # Initialize from prior
    x = sde.sample_prior(shape, device)
    
    dt = -sde.T / n_steps  # Negative for reverse time
    
    for i in range(n_steps):
        t = sde.T - i * sde.T / n_steps
        t_tensor = torch.full((shape[0],), t, device=device)
        
        # Compute drift and diffusion
        f = sde.drift(x, t_tensor)
        g = sde.diffusion(t_tensor)
        
        # Score
        score = score_model(x, t_tensor)
        
        # ODE step (Euler method)
        drift = f - 0.5 * g.view(-1, 1) ** 2 * score
        x = x + drift * dt
    
    return x
```

### 6.6 Reverse SDE Sampler

```python
@torch.no_grad()
def reverse_sde_sample(
    score_model: nn.Module,
    sde: SDE,
    shape: tuple,
    n_steps: int = 1000,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample using reverse SDE (stochastic).
    
    dx = [f(x,t) - g(t)² * ∇log p_t(x)] dt + g(t) dw̄
    """
    x = sde.sample_prior(shape, device)
    
    dt = sde.T / n_steps
    
    for i in range(n_steps):
        t = sde.T - i * sde.T / n_steps
        t_tensor = torch.full((shape[0],), t, device=device)
        
        # Drift and diffusion
        f = sde.drift(x, t_tensor)
        g = sde.diffusion(t_tensor)
        
        # Score
        score = score_model(x, t_tensor)
        
        # Reverse SDE step
        drift = f - g.view(-1, 1) ** 2 * score
        noise = torch.randn_like(x) * np.sqrt(dt)
        x = x - drift * dt + g.view(-1, 1) * noise
    
    return x
```

---

## 7. Predictor-Corrector Sampling

### 7.1 The Idea

Combine:
- **Predictor**: One step of reverse SDE/ODE
- **Corrector**: Langevin MCMC steps to refine

### 7.2 Implementation

```python
@torch.no_grad()
def predictor_corrector_sample(
    score_model: nn.Module,
    sde: SDE,
    shape: tuple,
    n_steps: int = 500,
    n_corrector: int = 1,
    snr: float = 0.16,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Predictor-Corrector sampling.
    """
    x = sde.sample_prior(shape, device)
    
    dt = sde.T / n_steps
    
    for i in range(n_steps):
        t = sde.T - i * sde.T / n_steps
        t_tensor = torch.full((shape[0],), t, device=device)
        
        # Predictor (Euler-Maruyama)
        f = sde.drift(x, t_tensor)
        g = sde.diffusion(t_tensor)
        score = score_model(x, t_tensor)
        
        drift = f - g.view(-1, 1) ** 2 * score
        noise = torch.randn_like(x) * np.sqrt(dt)
        x = x - drift * dt + g.view(-1, 1) * noise
        
        # Corrector (Langevin MCMC)
        for _ in range(n_corrector):
            score = score_model(x, t_tensor)
            noise = torch.randn_like(x)
            
            # Step size based on SNR
            grad_norm = torch.mean(torch.sum(score ** 2, dim=1)).sqrt()
            step_size = (snr * g.mean() / grad_norm) ** 2 * 2
            
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise
    
    return x
```

---

## 8. Summary

| SDE Type | Drift $\mathbf{f}(\mathbf{x}, t)$ | Diffusion $g(t)$ | Prior $p_T$ |
|----------|-----------------------------------|------------------|-------------|
| **VE** | $\mathbf{0}$ | $\sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$ | $\mathcal{N}(\mathbf{0}, \sigma_{\max}^2 \mathbf{I})$ |
| **VP** | $-\frac{1}{2}\beta(t)\mathbf{x}$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |

| Sampler | Stochastic? | Likelihood? | Quality |
|---------|-------------|-------------|---------|
| **Reverse SDE** | Yes | No | High |
| **Prob. Flow ODE** | No | Yes | Medium |
| **Predictor-Corrector** | Yes | No | Highest |

!!! tip "Key Takeaways"
    1. **Continuous-time formulation** unifies discrete diffusion models
    2. **Reverse SDE** requires the score function
    3. **Probability flow ODE** enables deterministic sampling and exact likelihood
    4. **Predictor-corrector** combines the best of both worlds

---

## References

1. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*.
2. Anderson, B. D. O. (1982). "Reverse-time diffusion equation models." *Stochastic Processes and their Applications*.
