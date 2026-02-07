# Deterministic Sampling

DDIM with $\eta = 0$ provides **fully deterministic** generation: the same initial noise always produces the same output. This section develops the deterministic sampling formula, contrasts it with stochastic (ancestral) sampling, and explores the resulting latent space.

## The Deterministic Update Rule

Setting $\sigma_t = 0$ in the DDIM update eliminates all stochasticity:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0(x_t, t) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

where $\hat{x}_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta) / \sqrt{\bar{\alpha}_t}$ is the predicted clean data.

This can be rewritten as:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

The first term rescales the denoised prediction; the second adds back noise at the appropriate level for timestep $t-1$. No random sampling occurs.

## Comparison with Ancestral (Stochastic) Sampling

| Property | DDPM (ancestral) | DDIM ($\eta = 0$) | DDIM ($0 < \eta < 1$) |
|----------|------------------|--------------------|-----------------------|
| Stochasticity | Every step | None | Partial |
| Reproducibility | Different each run | Same noise → same output | Partially reproducible |
| Sample diversity | Higher | Lower | Intermediate |
| Sample quality | Slightly better | Slightly worse | Tuneable |
| Encoding | Not possible | Invertible | Not cleanly invertible |

Ancestral sampling adds fresh noise at every step:

$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t\, z, \qquad z \sim \mathcal{N}(0, I)$$

This stochastic exploration can correct score-estimation errors but prevents exact inversion. The $\eta$ parameter interpolates between the two extremes.

## Connection to Probability Flow ODE

Deterministic DDIM is the Euler discretisation of the probability flow ODE:

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

This connection is more than analogy—in the continuous-time limit, DDIM sampling and probability flow ODE integration converge. This means DDIM inherits the theoretical properties of the ODE: same marginal distributions as the SDE, exact likelihood computation via change of variables, and an invertible data-to-noise mapping.

## Latent Space Properties

The deterministic mapping $x_0 \mapsto x_T$ (encoding) and $x_T \mapsto x_0$ (decoding) define an invertible correspondence between data and noise:

**Encoding.** Run the DDIM update forward in time to map clean data to latent noise:

$$x_{t+1} = \sqrt{\bar{\alpha}_{t+1}}\, \hat{x}_0(x_t, t) + \sqrt{1-\bar{\alpha}_{t+1}} \cdot \epsilon_\theta(x_t, t)$$

**Interpolation.** Given two data points $x_0^{(a)}$ and $x_0^{(b)}$, encode to latents $x_T^{(a)}$ and $x_T^{(b)}$, interpolate $x_T^{(\lambda)} = (1-\lambda)\, x_T^{(a)} + \lambda\, x_T^{(b)}$, and decode. The results are semantically meaningful interpolations—far smoother than pixel-space or VAE latent-space interpolation.

**Editing.** Encode a real image, modify the latent representation (e.g., by adding a direction corresponding to a semantic attribute), and decode to produce a coherent edit.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import List


@torch.no_grad()
def ddim_deterministic_sample(
    model: nn.Module,
    shape: tuple,
    timesteps: List[int],
    alpha_bars: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Deterministic DDIM sampling (eta=0).

    Args:
        model: Noise prediction network ε_θ(x_t, t).
        shape: Output shape (batch_size, ...).
        timesteps: Subsequence of timesteps (descending).
        alpha_bars: Cumulative product schedule.
        device: Compute device.

    Returns:
        Generated samples.
    """
    x = torch.randn(shape, device=device)

    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        eps_pred = model(x, t_batch)

        # Predict x_0
        alpha_bar_t = alpha_bars[t]
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(
            alpha_bar_t
        )

        # Get alpha_bar for next timestep
        if i + 1 < len(timesteps):
            alpha_bar_prev = alpha_bars[timesteps[i + 1]]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # Deterministic update (eta=0)
        x = (
            torch.sqrt(alpha_bar_prev) * x0_pred
            + torch.sqrt(1 - alpha_bar_prev) * eps_pred
        )

    return x


@torch.no_grad()
def ddim_encode(
    model: nn.Module,
    x_0: torch.Tensor,
    timesteps: List[int],
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Encode data to latent space via DDIM inversion.

    Args:
        model: Noise prediction network.
        x_0: Clean data to encode.
        timesteps: Timestep subsequence (ascending for encoding).

    Returns:
        Latent representation x_T.
    """
    x = x_0

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        eps_pred = model(x, t_batch)

        alpha_bar_t = alpha_bars[t]
        alpha_bar_next = alpha_bars[t_next]

        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(
            alpha_bar_t
        )
        x = (
            torch.sqrt(alpha_bar_next) * x0_pred
            + torch.sqrt(1 - alpha_bar_next) * eps_pred
        )

    return x
```

## Summary

Deterministic DDIM sampling eliminates stochasticity by setting $\eta = 0$, creating a bijective mapping between noise and data. This enables reproducible generation, latent-space encoding, semantic interpolation, and image editing. The connection to the probability flow ODE provides theoretical grounding and links to exact likelihood computation.
