# Training Objective

The training objective connects the forward and reverse processes. Starting from the variational lower bound, we derive the simplified noise-prediction loss used in practice and show its equivalence to denoising score matching.

## Variational Lower Bound

Diffusion models are latent variable models with latents $x_{1:T}$. The evidence lower bound (ELBO) gives:

$$\log p_\theta(x_0) \geq \mathbb{E}_q\!\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -\mathcal{L}_{\text{VLB}}$$

### Decomposition

The VLB decomposes into interpretable terms:

$$\mathcal{L}_{\text{VLB}} = \underbrace{D_{\text{KL}}\bigl(q(x_T|x_0) \,\|\, p(x_T)\bigr)}_{\mathcal{L}_T} + \sum_{t=2}^{T} \underbrace{D_{\text{KL}}\bigl(q(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t)\bigr)}_{\mathcal{L}_{t-1}} + \underbrace{(-\log p_\theta(x_0|x_1))}_{\mathcal{L}_0}$$

**$\mathcal{L}_T$ (prior matching):** Measures how well the forward process terminal distribution matches the prior $\mathcal{N}(0, I)$. This is constant with respect to $\theta$—no gradients flow through it.

**$\mathcal{L}_{t-1}$ (denoising matching):** The core training signal. Each term measures the KL divergence between the true posterior $q(x_{t-1}|x_t, x_0)$ and the learned reverse transition $p_\theta(x_{t-1}|x_t)$. Both are Gaussian, so the KL has a closed form.

**$\mathcal{L}_0$ (reconstruction):** Measures reconstruction quality at the final denoising step.

### KL Between Gaussians

Since both $q(x_{t-1}|x_t, x_0)$ and $p_\theta(x_{t-1}|x_t)$ are Gaussian with the same (fixed) variance $\sigma_t^2$, the KL simplifies to a squared distance between means:

$$\mathcal{L}_{t-1} = \frac{1}{2\sigma_t^2} \left\|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\right\|^2 + C$$

where $\tilde{\mu}_t$ is the true posterior mean and $\mu_\theta$ is the learned mean.

## The Simplified Loss

Ho et al. (2020) showed that substituting the noise-prediction parameterisation into the KL divergence yields:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t,\, x_0,\, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

where $t \sim \text{Uniform}(1, T)$, $x_0 \sim p_{\text{data}}$, $\epsilon \sim \mathcal{N}(0, I)$, and $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$.

This drops the per-timestep weighting $1/(2\sigma_t^2)$ from the VLB. While this means $\mathcal{L}_{\text{simple}}$ is no longer a valid lower bound on the log-likelihood, it works better in practice because it up-weights the loss at small $t$ (where fine details matter most).

### The Training Algorithm

```
repeat:
    x_0 ~ p_data                           # Sample clean data
    t ~ Uniform({1, ..., T})                # Random timestep
    ε ~ N(0, I)                             # Sample noise
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε      # Create noisy sample
    loss = ‖ε - ε_θ(x_t, t)‖²             # Predict noise
    θ ← θ - η ∇_θ loss                     # Gradient step
until converged
```

## Equivalence to Denoising Score Matching

The simplified loss is denoising score matching (Vincent, 2011) with $\sigma^2$ weighting. Under the reparameterisation $s_\theta(x_t, t) = -\epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}$:

$$\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2}\,\mathbb{E}_{x_0, \epsilon}\!\left[\left\|s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}\right\|^2\right]$$

The target $-\epsilon/\sqrt{1-\bar{\alpha}_t}$ is the score of the noise kernel $\nabla_{x_t} \log q(x_t | x_0)$. Multiplying by $(1-\bar{\alpha}_t) = \sigma_t^2$ recovers the noise-prediction loss.

| DDPM | DSM | Score SDE |
|------|-----|-----------|
| Predict noise $\epsilon$ | Predict score $s = -\epsilon/\sigma_t$ | Match $\nabla_x \log p_t(x)$ |
| $\|\epsilon - \epsilon_\theta\|^2$ | $\|s_\theta + \epsilon/\sigma_t\|^2$ | $\lambda(t)\|s_\theta - \nabla \log p_t\|^2$ |

All three describe the same optimisation problem.

## Loss Weighting Strategies

The general weighted objective is:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\!\left[w(t)\, \|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

| Weighting | $w(t)$ | Properties |
|-----------|--------|------------|
| Simple (DDPM) | 1 | Best sample quality in practice |
| VLB | $\beta_t^2 / (2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t))$ | Valid lower bound on log-likelihood |
| SNR | $\bar{\alpha}_t / (1-\bar{\alpha}_t)$ | Emphasises low-noise regime |
| Min-SNR-$\gamma$ | $\min(\text{SNR}(t), \gamma) / \text{SNR}(t)$ | Clips high-SNR contribution |

The min-SNR-$\gamma$ weighting (Hang et al., 2023) with $\gamma=5$ provides a good balance between sample quality and likelihood.

## Prediction Targets

The network can predict different quantities, each offering advantages:

$$\begin{aligned}
\text{Noise: } & \mathcal{L}_\epsilon = \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \\
\text{Clean data: } & \mathcal{L}_{x_0} = \|x_0 - \hat{x}_\theta(x_t, t)\|^2 \\
\text{Velocity: } & \mathcal{L}_v = \|v_t - v_\theta(x_t, t)\|^2, \quad v_t = \sqrt{\bar{\alpha}_t}\,\epsilon - \sqrt{1-\bar{\alpha}_t}\,x_0
\end{aligned}$$

The velocity parameterisation (Salimans & Ho, 2022) interpolates between noise and data prediction, providing more uniform gradient magnitudes across timesteps.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss:
    """Compute diffusion training loss with configurable weighting."""

    def __init__(
        self,
        alpha_bars: torch.Tensor,
        prediction: str = "noise",
        weighting: str = "simple",
    ):
        self.alpha_bars = alpha_bars
        self.prediction = prediction
        self.weighting = weighting
        self.T = len(alpha_bars)

    def __call__(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample timesteps and noise
        t = torch.randint(0, self.T, (batch_size,), device=device)
        eps = torch.randn_like(x_0)

        # Create noisy samples
        a_bar = self.alpha_bars[t]
        while a_bar.dim() < x_0.dim():
            a_bar = a_bar.unsqueeze(-1)

        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * eps

        # Network prediction
        pred = model(x_t, t)

        # Compute target based on prediction type
        if self.prediction == "noise":
            target = eps
        elif self.prediction == "x0":
            target = x_0
        elif self.prediction == "velocity":
            target = torch.sqrt(a_bar) * eps - torch.sqrt(1 - a_bar) * x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction}")

        # Per-sample loss
        loss = ((pred - target) ** 2).flatten(1).mean(1)

        # Apply weighting
        if self.weighting == "simple":
            return loss.mean()
        elif self.weighting == "snr":
            snr = a_bar.squeeze() / (1 - a_bar.squeeze())
            return (snr * loss).mean()
        elif self.weighting == "min_snr":
            snr = a_bar.squeeze() / (1 - a_bar.squeeze())
            w = torch.clamp(snr, max=5.0) / snr
            return (w * loss).mean()
        else:
            return loss.mean()
```

## Practical Training Considerations

**Exponential moving average (EMA).** Maintain an exponential moving average of model weights with decay $\approx 0.9999$ and use the EMA weights for sampling. This is critical for stable generation.

**Gradient clipping.** Clip gradient norms to 1.0 to prevent training instabilities, especially at high resolutions.

**Timestep sampling.** Uniform sampling is standard, but importance sampling based on per-timestep loss magnitudes can accelerate convergence.

**Learning rate.** Typical values are $10^{-4}$ to $3 \times 10^{-4}$ with Adam or AdamW. Warmup over the first 5,000–10,000 steps is common.

## Summary

The training objective for diffusion models reduces to a denoising regression problem: predict the noise $\epsilon$ that was added to create $x_t$ from $x_0$. This objective is theoretically grounded in the variational lower bound and equivalent to denoising score matching. The simplified (unweighted) loss provides the best sample quality, while VLB weighting gives valid log-likelihoods. All modern diffusion models—DDPM, score SDEs, latent diffusion—share this same core training procedure.
