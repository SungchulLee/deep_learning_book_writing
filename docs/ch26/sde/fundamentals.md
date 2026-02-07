# SDE Fundamentals

The **SDE (Stochastic Differential Equation) framework** (Song et al., 2021) provides a unified continuous-time perspective on diffusion models. This formulation subsumes DDPM, NCSN, and their variants as special cases of the same mathematical structure, enables new sampling algorithms, and connects to exact likelihood computation.

## From Discrete to Continuous

DDPM uses $T$ discrete steps; NCSN uses $L$ discrete noise levels. As these become fine-grained ($T, L \to \infty$), the processes converge to continuous-time SDEs. The continuous framework provides cleaner analysis and more flexible sampling.

## The Forward SDE

### General Form

The forward diffusion process is described by:

$$dx = f(x, t)\, dt + g(t)\, dW$$

where:

| Symbol | Name | Role |
|--------|------|------|
| $f(x, t)$ | Drift coefficient | Deterministic tendency (e.g., shrinkage toward origin) |
| $g(t)$ | Diffusion coefficient | Noise magnitude |
| $dW$ | Wiener process increment | Standard Brownian motion |

Time runs from $t=0$ (data) to $t=T$ (noise). The choice of $f$ and $g$ defines the specific SDE variant.

## The Reverse SDE

Anderson's time-reversal theorem (1982) gives the reverse process:

$$\boxed{dx = \bigl[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\bigr] dt + g(t)\, d\bar{W}}$$

where $d\bar{W}$ is a reverse-time Wiener process and $\nabla_x \log p_t(x)$ is the time-dependent **score function**.

The reverse drift has two components: the original drift $f(x,t)$ and a score-guided correction $-g(t)^2 \nabla_x \log p_t(x)$ that steers the process toward higher-probability regions. The key insight is that the reverse process requires only the score function—exactly what denoising score matching trains us to estimate.

## Marginal Distributions

For both VP and VE SDEs, the marginal $q(x_t|x_0)$ is Gaussian with known parameters, enabling the same efficient training procedure as discrete diffusion: sample a random time $t$, compute $x_t$ directly from $x_0$, and train the score network.

## Numerical Integration

### Euler–Maruyama Method

The simplest discretisation for the reverse SDE:

$$x_{t-\Delta t} = x_t - \bigl[f(x_t, t) - g(t)^2 s_\theta(x_t, t)\bigr]\Delta t + g(t)\sqrt{\Delta t}\, z, \quad z \sim \mathcal{N}(0, I)$$

### Predictor-Corrector Sampling

Combine a predictor step (Euler–Maruyama or ODE) with corrector steps (Langevin MCMC):

1. **Predict**: Take one reverse SDE/ODE step
2. **Correct**: Run $K$ Langevin steps at the current noise level to refine

This produces the highest-quality samples at the cost of additional function evaluations.

## PyTorch Implementation

```python
"""SDE Framework for Diffusion Models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SDE(ABC):
    """Abstract base class for forward SDEs."""

    def __init__(self, T: float = 1.0):
        self.T = T

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient f(x, t)."""

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient g(t)."""

    @abstractmethod
    def marginal_params(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of q(x_t | x_0)."""

    @abstractmethod
    def sample_prior(
        self, shape: tuple, device: torch.device
    ) -> torch.Tensor:
        """Sample from the prior distribution p(x_T)."""

    def reverse_drift(
        self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Reverse SDE drift: f - g² · score."""
        g = self.diffusion(t)
        return self.drift(x, t) - g.view(-1, 1) ** 2 * score

    def ode_drift(
        self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Probability flow ODE drift: f - (1/2) g² · score."""
        g = self.diffusion(t)
        return self.drift(x, t) - 0.5 * g.view(-1, 1) ** 2 * score


def continuous_score_matching_loss(
    score_model: nn.Module,
    sde: SDE,
    x_0: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Continuous-time denoising score matching loss.

    Args:
        score_model: Network s_θ(x, t) predicting the score.
        sde: Forward SDE defining the noise process.
        x_0: Clean data samples.
        eps: Small offset to avoid t=0.

    Returns:
        Scalar loss.
    """
    N = x_0.shape[0]
    device = x_0.device

    # Sample random time uniformly in [eps, T]
    t = torch.rand(N, device=device) * (sde.T - eps) + eps

    # Get marginal distribution parameters
    mean, std = sde.marginal_params(x_0, t)

    # Sample x_t ~ q(x_t | x_0)
    noise = torch.randn_like(x_0)
    x_t = mean + std.view(-1, 1) * noise

    # Predict score
    pred_score = score_model(x_t, t)

    # Target score: ∇ log q(x_t | x_0) = -noise / std
    target_score = -noise / std.view(-1, 1)

    # Weighted MSE loss
    loss = torch.mean(torch.sum((pred_score - target_score) ** 2, dim=1))
    return loss


@torch.no_grad()
def reverse_sde_sample(
    score_model: nn.Module,
    sde: SDE,
    shape: tuple,
    n_steps: int = 1000,
    device: str = "cpu",
) -> torch.Tensor:
    """Sample via reverse SDE (Euler–Maruyama)."""
    import numpy as np

    x = sde.sample_prior(shape, torch.device(device))
    dt = sde.T / n_steps

    for i in range(n_steps):
        t_val = sde.T - i * dt
        t = torch.full((shape[0],), t_val, device=device)

        f = sde.drift(x, t)
        g = sde.diffusion(t)
        score = score_model(x, t)

        # Reverse SDE step
        drift = f - g.view(-1, 1) ** 2 * score
        noise = torch.randn_like(x) * np.sqrt(dt)
        x = x - drift * dt + g.view(-1, 1) * noise

    return x
```

## Summary

The SDE framework unifies discrete diffusion models under continuous-time stochastic calculus. The forward SDE defines noise corruption; the reverse SDE defines generation via the score function; and the probability flow ODE provides a deterministic alternative. The next sections develop the two main SDE variants (VP and VE) and the probability flow ODE in detail.

## References

1. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*.
2. Anderson, B. D. O. (1982). "Reverse-Time Diffusion Equation Models." *Stochastic Processes and their Applications*.
