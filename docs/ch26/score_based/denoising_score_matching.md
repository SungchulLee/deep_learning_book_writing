# Denoising Score Matching

Denoising score matching (Vincent, 2011) replaces the expensive Hessian trace of explicit score matching with a simple regression loss against a known target. By learning the score of a noise-perturbed distribution, it simultaneously solves the computational cost problem and the low-density estimation problem. DSM is the training objective underlying all modern diffusion models.

## Motivation

Explicit score matching has two critical limitations in practice.

| Problem | Description | Impact |
|---------|-------------|--------|
| **Computational cost** | The Laplacian $\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta)$ requires $D$ backward passes | Impractical for images ($D > 10^4$) |
| **Low-density estimation** | Score estimates are unreliable where $p_{\text{data}}(\mathbf{x}) \approx 0$ | Sampling fails between modes |

DSM solves both by learning the score of a **noise-perturbed** distribution.

## The Perturbed Distribution

Given clean data $\mathbf{x} \sim p_{\text{data}}$ and a Gaussian noise kernel

$$q(\tilde{\mathbf{x}} | \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}} \,|\, \mathbf{x}, \, \sigma^2 \mathbf{I})$$

the marginal distribution of noisy samples is:

$$q_\sigma(\tilde{\mathbf{x}}) = \int p_{\text{data}}(\mathbf{x}) \, q(\tilde{\mathbf{x}} | \mathbf{x}) \, d\mathbf{x}$$

This smoothed distribution has full support (defined everywhere), making the score well-defined even far from the data manifold.

## The Known Target Score

The key insight is that the score of the noise kernel is available in closed form:

$$\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}$$

Writing $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, this becomes $-\boldsymbol{\epsilon} / \sigma$. The target score at each noisy point simply points back toward the clean data, scaled by the inverse noise level.

## The DSM Objective

**Theorem (Vincent, 2011).** The optimal score network for the perturbed distribution $q_\sigma$ is found by minimising:

$$\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \, \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \, \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\cdot | \mathbf{x})} \!\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x})\|^2\right]$$

Substituting the known target:

$$\boxed{\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \, \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \!\left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2\right]}$$

Training becomes a regression problem: predict the negative noise direction at each noisy sample.

## Equivalence to Explicit Score Matching

DSM and ESM objectives differ only by a constant that does not depend on $\theta$:

$$\mathcal{L}_{\text{DSM}}(\theta) = \mathcal{L}_{\text{ESM}}(\theta;\, q_\sigma) + C_\sigma$$

The proof proceeds by expanding the Fisher divergence between $\mathbf{s}_\theta$ and $\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}})$, then using the law of total expectation to replace the marginal score $\nabla \log q_\sigma$ with the conditional score $\nabla \log q(\tilde{\mathbf{x}} | \mathbf{x})$. The cross-term that eliminates $\mathbf{s}_{\text{data}}$ in ESM is replaced here by the known conditional score, achieving the same effect without integration by parts.

As $\sigma \to 0$ the perturbed distribution converges to the data distribution, so $\lim_{\sigma \to 0} \mathcal{L}_{\text{DSM}} = \mathcal{L}_{\text{ESM}} + C$. DSM with small $\sigma$ therefore approximates ESM without computing any Jacobians.

## Multi-Scale Denoising Score Matching

### The Single-Scale Limitation

A single noise level $\sigma$ introduces a bias–coverage trade-off. Small $\sigma$ captures fine details near the data but leaves low-density regions poorly covered. Large $\sigma$ fills the space but destroys data structure.

### Noise Conditional Score Networks (NCSN)

The solution is to learn scores across a range of noise levels $\{\sigma_i\}_{i=1}^L$ simultaneously:

$$\mathcal{L}_{\text{NCSN}}(\theta) = \sum_{i=1}^L \lambda(\sigma_i) \, \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \!\left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma_i \boldsymbol{\epsilon},\, \sigma_i) + \frac{\boldsymbol{\epsilon}}{\sigma_i}\right\|^2\right]$$

The score network $\mathbf{s}_\theta(\cdot, \sigma)$ is conditioned on the noise level, so it learns a family of scores indexed by $\sigma$. Large $\sigma$ provides global structure; small $\sigma$ captures fine details. During sampling (annealed Langevin dynamics), the noise level is gradually decreased.

### Noise Schedule

The standard choice is a **geometric schedule**:

$$\sigma_i = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{(i-1)/(L-1)}, \quad i = 1, \ldots, L$$

Typical values: $\sigma_{\max} \approx$ maximum pairwise distance in the data, $\sigma_{\min} \approx 0.01$, $L \in [10, 1000]$.

## Weighting Strategies

The choice of $\lambda(\sigma)$ affects training dynamics:

| Weighting | Formula | Rationale |
|-----------|---------|-----------|
| Uniform | $\lambda = 1$ | Baseline; low-noise scales dominate gradient |
| $\sigma^2$ | $\lambda = \sigma^2$ | Balances gradient magnitudes across scales (NCSN default) |
| SNR | $\lambda = 1/(1 + \sigma^2)$ | Emphasises low-noise regime for fine details |

### Why $\sigma^2$ Weighting Works

The score magnitude scales as $\|\mathbf{s}\| \sim 1/\sigma$, so the unweighted loss at scale $\sigma$ is $O(1/\sigma^2)$. Multiplying by $\sigma^2$ normalises contributions:

$$\sigma^2 \left\|\mathbf{s}_\theta + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2 = \|\sigma \, \mathbf{s}_\theta + \boldsymbol{\epsilon}\|^2$$

This is equivalent to predicting the noise $\boldsymbol{\epsilon}$ directly, which is exactly the DDPM training objective.

## Connection to Diffusion Models

### DSM Is DDPM Training

The DDPM objective is:

$$\mathcal{L}_{\text{DDPM}}(t) = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon}\|^2\right]$$

where $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}$. This is exactly DSM with $\sigma^2$ weighting under the correspondence:

| DDPM | DSM |
|------|-----|
| Noise predictor $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | $\mathbf{s}_\theta = -\boldsymbol{\epsilon}_\theta / \sqrt{1 - \bar{\alpha}_t}$ |
| Forward process $\mathbf{x}_t$ | Noisy sample $\tilde{\mathbf{x}} = \mathbf{x} + \sigma_t \boldsymbol{\epsilon}$ |
| Noise schedule $\bar{\alpha}_t$ | $\sigma_t = \sqrt{(1 - \bar{\alpha}_t) / \bar{\alpha}_t}$ |

All three frameworks—DSM, DDPM, and score-based SDEs—describe the same mathematical object.

## Alternative Noise Kernels

While Gaussian noise is standard, other kernels can be useful:

**Laplace kernel** $q(\tilde{\mathbf{x}} | \mathbf{x}) = \text{Laplace}(\tilde{\mathbf{x}} | \mathbf{x}, b)$: target score is $-\text{sign}(\tilde{\mathbf{x}} - \mathbf{x})/b$. Useful for heavy-tailed data.

**Student-$t$ kernel** $q(\tilde{\mathbf{x}} | \mathbf{x}) = t_\nu(\tilde{\mathbf{x}} | \mathbf{x}, \sigma^2)$: target score is $-(\nu + D)(\tilde{\mathbf{x}} - \mathbf{x}) / (\nu\sigma^2 + \|\tilde{\mathbf{x}} - \mathbf{x}\|^2)$. Provides robustness to outliers.

## PyTorch Implementation

### Basic DSM Loss

```python
import torch
import torch.nn as nn


def dsm_loss(score_net: nn.Module, x: torch.Tensor,
             sigma: float) -> torch.Tensor:
    """DSM loss for a single noise level.

    Args:
        score_net: Network s(x) mapping [batch, dim] → [batch, dim].
        x: Clean data [batch, dim].
        sigma: Noise standard deviation.

    Returns:
        Scalar loss.
    """
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score = score_net(x_noisy)
    target = -noise / sigma
    return 0.5 * ((score - target)**2).sum(dim=-1).mean()
```

### Energy-Based DSM Loss

When the model is parameterised as an energy function $E_\theta(\mathbf{x})$ rather than a direct score network:

```python
def dsm_loss_energy(energy_net: nn.Module, x: torch.Tensor,
                    sigma: float) -> torch.Tensor:
    """DSM loss for an energy-based model.

    The score is computed as ∇_x E_θ(x) via autograd.

    Args:
        energy_net: Scalar energy network E(x).
        x: Clean data [batch, dim].
        sigma: Noise standard deviation.

    Returns:
        Scalar loss.
    """
    noise = torch.randn_like(x)
    x_noisy = (x + sigma * noise).requires_grad_(True)

    energy = energy_net(x_noisy)
    score = torch.autograd.grad(
        energy.sum(), x_noisy, create_graph=True
    )[0]

    target = -noise / sigma
    return ((score - target)**2).sum(dim=1).mean()
```

### Multi-Scale DSM (NCSN-Style)

```python
def multi_scale_dsm_loss(
    score_net: nn.Module, x: torch.Tensor,
    sigmas: list[float], weights: list[float] | None = None
) -> torch.Tensor:
    """Multi-scale DSM loss with σ² weighting by default.

    Args:
        score_net: Network s(x, sigma) with noise conditioning.
        x: Clean data [batch, dim].
        sigmas: Noise levels.
        weights: Per-level weights (default: σ²-normalised).

    Returns:
        Weighted DSM loss across scales.
    """
    if weights is None:
        raw = [s**2 for s in sigmas]
        total = sum(raw)
        weights = [w / total for w in raw]

    loss = torch.tensor(0.0, device=x.device)
    for sigma, w in zip(sigmas, weights):
        noise = torch.randn_like(x)
        x_noisy = x + sigma * noise
        sigma_t = torch.full((x.shape[0], 1), sigma, device=x.device)
        score = score_net(x_noisy, sigma_t)
        target = -noise / sigma
        loss = loss + w * 0.5 * ((score - target)**2).sum(dim=-1).mean()

    return loss
```

### Efficient Random-Sigma Training

```python
def dsm_loss_random_sigma(
    score_net: nn.Module, x: torch.Tensor,
    sigma_min: float = 0.01, sigma_max: float = 10.0
) -> torch.Tensor:
    """DSM loss with per-sample random sigma (log-uniform).

    More efficient than iterating over all sigma levels.
    """
    log_sigma = (
        torch.rand(x.shape[0], 1, device=x.device)
        * (torch.log(torch.tensor(sigma_max)) - torch.log(torch.tensor(sigma_min)))
        + torch.log(torch.tensor(sigma_min))
    )
    sigma = torch.exp(log_sigma)

    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score = score_net(x_noisy, sigma)
    target = -noise / sigma

    # σ²-weighted loss
    return 0.5 * (sigma**2 * (score - target)**2).sum(dim=-1).mean()
```

### Score Network with Noise Conditioning

```python
class ScoreNet(nn.Module):
    """MLP score network conditioned on noise level."""

    def __init__(self, data_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Args: x [batch, dim], sigma [batch, 1]."""
        inp = torch.cat([x, torch.log(sigma) / 4.0], dim=-1)
        return self.net(inp)
```

### Training Loop

```python
import numpy as np

def train_dsm(data: torch.Tensor, n_epochs: int = 5000,
              batch_size: int = 256, lr: float = 1e-3):
    """Train a score network with random-sigma DSM."""
    model = ScoreNet(data_dim=data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        idx = torch.randperm(len(data))[:batch_size]
        batch = data[idx]

        loss = dsm_loss_random_sigma(model, batch,
                                     sigma_min=0.01, sigma_max=5.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

    return model
```

## Noise Level Selection Guidelines

1. **Normalise data first** to zero mean and unit variance.
2. **Single-scale DSM:** $\sigma \approx 0.5$ for normalised data, or $0.5 \times$ median nearest-neighbour distance.
3. **Multi-scale DSM:** geometric schedule from $\sigma_{\min} = 0.01$ to $\sigma_{\max} \approx \max_{i,j} \|\mathbf{x}_i - \mathbf{x}_j\|$.

## Summary

| Aspect | Description |
|--------|-------------|
| **Objective** | $\frac{1}{2}\mathbb{E}[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \boldsymbol{\epsilon}/\sigma\|^2]$ |
| **Target** | $-\boldsymbol{\epsilon}/\sigma$ (negative noise direction) |
| **Cost** | Single forward + backward pass (no Hessian) |
| **Equivalence** | $\mathcal{L}_{\text{DSM}} \to \mathcal{L}_{\text{ESM}}$ as $\sigma \to 0$ |
| **Multi-scale** | NCSN: learn scores at multiple $\sigma$ levels |
| **Diffusion link** | DDPM training = DSM with $\sigma^2$ weighting |

## Exercises

1. **Laplace kernel target.** For $q(\tilde{x}|x) = \frac{1}{2b}e^{-|\tilde{x}-x|/b}$, derive $\nabla_{\tilde{x}} \log q = -\text{sign}(\tilde{x} - x)/b$.

2. **Multi-noise experiment.** Train models with single $\sigma \in \{0.1, 0.3, 0.5, 1.0\}$ on a 2-D Swiss roll. Plot learned score fields and compare sample quality via Langevin dynamics.

3. **Weighting study.** Compare uniform vs $\sigma^2$ weighting for multi-scale DSM. Measure per-scale loss at convergence.

4. **DDPM equivalence.** Starting from $\mathcal{L}_{\text{DDPM}}$, derive the equivalent DSM formulation. Show the correspondence between $\bar{\alpha}_t$ and $\sigma_t$.

## References

1. Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." *Neural Computation*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Song, Y., & Ermon, S. (2020). "Improved Techniques for Training Score-Based Generative Models." *NeurIPS*.
4. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
