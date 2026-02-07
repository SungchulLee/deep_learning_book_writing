# Variance Preserving SDE (VP-SDE)

The **VP-SDE** maintains approximately unit variance throughout the diffusion process. It is the continuous-time generalisation of DDPM and the default choice for most applications.

## Formulation

$$dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dW$$

| Component | Expression | Role |
|-----------|-----------|------|
| Drift $f(x,t)$ | $-\frac{1}{2}\beta(t)\,x$ | Pulls toward origin (shrinks signal) |
| Diffusion $g(t)$ | $\sqrt{\beta(t)}$ | Injects noise |
| Schedule $\beta(t)$ | Linear: $\beta_{\min} + t(\beta_{\max} - \beta_{\min})$ | Controls rate |

Typical values: $\beta_{\min} = 0.1$, $\beta_{\max} = 20.0$, $t \in [0, 1]$.

## Marginal Distribution

The marginal $q(x_t | x_0)$ is Gaussian:

$$q(x_t | x_0) = \mathcal{N}\bigl(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t)\, I\bigr)$$

where $\bar{\alpha}_t = \exp\!\left(-\int_0^t \beta(s)\, ds\right) = \exp\!\left(-\frac{1}{2}\beta_{\min} t - \frac{1}{4}(\beta_{\max} - \beta_{\min})t^2\right)$.

## Key Properties

**Bounded variance.** If $x_0$ has unit variance, then $\text{Var}[x_t] \approx 1$ for all $t$. The drift $-\frac{1}{2}\beta(t)x$ shrinks the signal while the noise $\sqrt{\beta(t)}\,dW$ adds variance, keeping the total approximately constant.

**Standard Gaussian prior.** As $t \to T$, $\bar{\alpha}_t \to 0$ and $x_T \sim \mathcal{N}(0, I)$. This simplifies the prior and avoids the need for a scale-dependent prior.

**DDPM connection.** The discrete VP-SDE with $\Delta t = 1/T$ yields the DDPM forward process $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$.

## Reverse VP-SDE

Applying Anderson's theorem:

$$dx = \left[-\frac{1}{2}\beta(t)\, x - \beta(t)\, \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\, d\bar{W}$$

The score-guided drift $-\beta(t) \nabla_x \log p_t(x)$ dominates the reverse dynamics, steering samples from noise toward data.

## PyTorch Implementation

```python
import torch
import numpy as np

from .fundamentals import SDE


class VPSDE(SDE):
    """Variance Preserving SDE (continuous-time DDPM)."""

    def __init__(
        self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0
    ):
        super().__init__(T)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.beta(t).view(-1, 1) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))

    def marginal_params(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean coefficient and std for q(x_t | x_0)."""
        log_mean_coef = (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min)
            - 0.5 * t * self.beta_min
        )
        mean_coef = torch.exp(log_mean_coef)
        std = torch.sqrt(1.0 - mean_coef ** 2)
        mean = mean_coef.view(-1, 1) * x_0
        return mean, std

    def sample_prior(
        self, shape: tuple, device: torch.device
    ) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio at time t."""
        log_mean_coef = (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min)
            - 0.5 * t * self.beta_min
        )
        alpha_bar = torch.exp(2 * log_mean_coef)
        return alpha_bar / (1 - alpha_bar)
```

## When to Use VP-SDE

VP-SDE is the default recommendation for most applications due to bounded variance and numerical stability. It is the natural continuous-time extension of DDPM and the basis for most production diffusion models (Stable Diffusion, DALL-E, Imagen).
