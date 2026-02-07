# Variance Exploding SDE (VE-SDE)

The **VE-SDE** adds noise without shrinking the signal, allowing the variance to grow unbounded. It is the continuous-time generalisation of NCSN (Noise Conditional Score Networks).

## Formulation

$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\, dW$$

| Component | Expression | Role |
|-----------|-----------|------|
| Drift $f(x,t)$ | $0$ | No signal shrinkage |
| Diffusion $g(t)$ | $\sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$ | Growing noise |
| Schedule $\sigma(t)$ | $\sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$ | Geometric increase |

Typical values: $\sigma_{\min} = 0.01$, $\sigma_{\max} = 50.0$.

## Marginal Distribution

$$q(x_t | x_0) = \mathcal{N}\bigl(x_t;\, x_0,\, \sigma^2(t)\, I\bigr)$$

The mean is simply $x_0$—the signal is preserved at all times. Only the noise variance grows.

## Key Properties

**No signal shrinkage.** Unlike VP-SDE, the coefficient on $x_0$ remains 1. The original data is always present; it is simply buried under increasing noise.

**Growing variance.** $\text{Var}[x_t] = \text{Var}[x_0] + \sigma^2(t)$. For large $t$, $\sigma(T) \gg 1$ and the distribution is dominated by noise.

**Scaled Gaussian prior.** At $t=T$, $p_T \approx \mathcal{N}(0, \sigma_{\max}^2 I)$, requiring a scale-dependent prior.

**NCSN connection.** The discrete VE-SDE corresponds to adding noise at multiple scales $\sigma_1 > \cdots > \sigma_L$ as in NCSN/SMLD (Score Matching with Langevin Dynamics).

## Comparison with VP-SDE

| Aspect | VP-SDE | VE-SDE |
|--------|--------|--------|
| Signal scaling | Decreasing ($\sqrt{\bar{\alpha}_t}$) | Constant (1) |
| Variance | Bounded (~1) | Exploding ($\sigma^2(t)$) |
| Prior at $t=T$ | $\mathcal{N}(0, I)$ | $\mathcal{N}(0, \sigma_{\max}^2 I)$ |
| Training stability | More stable | May need normalisation |
| Discrete analogue | DDPM | NCSN |
| Default recommendation | ✅ Most applications | Multi-scale noise modelling |

Both formulations produce equivalent marginal distributions under appropriate reparameterisation and have equivalent reverse processes. The practical differences are numerical: VP-SDE's bounded variance provides better conditioning.

## Reverse VE-SDE

$$dx = -\sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})} \cdot \nabla_x \log p_t(x)\, dt + \sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}\, d\bar{W}$$

Since the forward drift is zero, the entire reverse drift is score-guided.

## PyTorch Implementation

```python
import torch
import numpy as np

from .fundamentals import SDE


class VESDE(SDE):
    """Variance Exploding SDE (continuous-time NCSN)."""

    def __init__(
        self, sigma_min: float = 0.01, sigma_max: float = 50.0, T: float = 1.0
    ):
        super().__init__(T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Geometric sigma schedule."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma(t)
        return sigma * torch.sqrt(
            2 * torch.tensor(np.log(self.sigma_max / self.sigma_min))
        )

    def marginal_params(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std for q(x_t | x_0)."""
        mean = x_0  # No signal shrinkage
        std = self.sigma(t)
        return mean, std

    def sample_prior(
        self, shape: tuple, device: torch.device
    ) -> torch.Tensor:
        return torch.randn(shape, device=device) * self.sigma_max
```

## When to Use VE-SDE

VE-SDE is appropriate when signal preservation is important or when working with multi-scale noise models. In practice, VP-SDE is preferred for most image generation tasks due to its numerical stability and simpler prior.
