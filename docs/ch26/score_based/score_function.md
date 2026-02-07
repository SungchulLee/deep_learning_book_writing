# Score Function

The **score function** is the gradient of the log-density. It is the mathematical foundation of score-based generative modelling: because the gradient eliminates the intractable normalisation constant, a model that learns the score can generate samples without ever evaluating the partition function. This section defines the score, develops its key properties, computes it analytically for common distributions, and connects it to energy-based models and Langevin sampling.

## Definition

For a probability distribution with density $p(\mathbf{x})$, the **score function** is:

$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})}$$

The score is a vector field over the data space: at every point $\mathbf{x}$ it points in the direction of steepest ascent of the log-density.

### Geometric Interpretation

At a mode $\mathbf{x}^*$ the score is zero (no direction of ascent). Away from modes the score points toward higher-density regions, with magnitude proportional to the steepness of the log-density landscape. The score field can be thought of as a "flow" that, if followed, carries probability mass toward the modes.

## Why the Score Matters: Normalisation-Free Modelling

In many models the density is specified only up to a normalisation constant:

$$p(\mathbf{x}) = \frac{\tilde{p}(\mathbf{x})}{Z}, \quad Z = \int \tilde{p}(\mathbf{x}) \, d\mathbf{x}$$

Computing $Z$ is intractable for complex models (e.g., deep energy-based models). The score eliminates this problem:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}} \log Z}_{= \, \mathbf{0}} = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x})$$

Since $Z$ does not depend on $\mathbf{x}$, it vanishes under differentiation. Score-based methods therefore work directly with unnormalised densities.

### Connection to Energy-Based Models

An energy-based model defines $p(\mathbf{x}) = Z^{-1} \exp(-E(\mathbf{x}))$. Its score is the negative energy gradient:

$$\mathbf{s}(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$$

The score points "downhill" in the energy landscape—toward lower energy and higher probability.

## Key Properties

### Zero Score at Modes

At any local maximum $\mathbf{x}^*$ of the density, $\mathbf{s}(\mathbf{x}^*) = \mathbf{0}$ by the first-order optimality condition.

### Zero Expected Score

The expected score under the data distribution is always zero:

$$\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})] = \int p(\mathbf{x}) \, \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})} \, d\mathbf{x} = \int \nabla_{\mathbf{x}} p(\mathbf{x}) \, d\mathbf{x} = \nabla_{\mathbf{x}} \int p(\mathbf{x}) \, d\mathbf{x} = \mathbf{0}$$

The last step exchanges integration and differentiation and uses $\int p = 1$. Intuitively, the "pushes" toward high-density regions are balanced by the "pushes" away from them when averaged over the whole distribution.

### Fisher Information

The expected squared norm of the score is the **Fisher information**:

$$\mathcal{I}(p) = \mathbb{E}_{p(\mathbf{x})}\!\left[\|\mathbf{s}(\mathbf{x})\|^2\right]$$

High Fisher information means a sharply peaked distribution with large score magnitudes; low Fisher information means a flat landscape. For an isotropic Gaussian $\mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$ in $D$ dimensions, $\mathcal{I} = D / \sigma^2$.

### Temperature Scaling

For a tempered distribution $p_T(\mathbf{x}) \propto p(\mathbf{x})^{1/T}$ the score scales linearly: $\mathbf{s}_T(\mathbf{x}) = \mathbf{s}(\mathbf{x}) / T$. Low temperature ($T < 1$) sharpens the landscape and amplifies scores; high temperature ($T > 1$) flattens it.

### Stein's Identity

For any smooth vector-valued function $\boldsymbol{\phi}(\mathbf{x})$ with suitable decay:

$$\mathbb{E}_{p}\!\left[\mathbf{s}(\mathbf{x}) \, \boldsymbol{\phi}(\mathbf{x})^\top + \nabla_{\mathbf{x}} \boldsymbol{\phi}(\mathbf{x})\right] = \mathbf{0}$$

This identity is the basis of **Stein discrepancies** and **kernelised Stein discrepancies**, which provide goodness-of-fit tests that depend only on the score (not the normalisation constant).

## Score Functions for Common Distributions

### Univariate Gaussian

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$s(x) = -\frac{x - \mu}{\sigma^2}$$

The score is linear in $x$, pointing toward the mean with magnitude inversely proportional to the variance.

### Multivariate Gaussian

For $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$\boxed{\mathbf{s}(\mathbf{x}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$$

For the isotropic case $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ this simplifies to $\mathbf{s}(\mathbf{x}) = -(\mathbf{x} - \boldsymbol{\mu}) / \sigma^2$.

### Gaussian Mixture Model

For $p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ the score is a **posterior-weighted** average of component scores:

$$\mathbf{s}(\mathbf{x}) = \sum_{k=1}^K w_k(\mathbf{x}) \, \mathbf{s}_k(\mathbf{x})$$

where $w_k(\mathbf{x}) = p(k | \mathbf{x})$ is the responsibility of component $k$ and $\mathbf{s}_k(\mathbf{x}) = -\boldsymbol{\Sigma}_k^{-1}(\mathbf{x} - \boldsymbol{\mu}_k)$. Unlike single-Gaussian scores, mixture scores are nonlinear: between modes the score interpolates between component scores based on proximity.

## Connection to Langevin Dynamics

Given access to the score, we can generate samples via **Langevin dynamics**:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\epsilon}{2} \, \mathbf{s}(\mathbf{x}_t) + \sqrt{\epsilon} \, \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The gradient term pushes samples toward high-probability regions; the noise term ensures exploration. As $\epsilon \to 0$ and $t \to \infty$, samples converge to the target distribution. For a detailed treatment see [Langevin Dynamics](../../ch13/langevin/langevin_dynamics.md).

## Score Networks

A **score network** $\mathbf{s}_\theta(\mathbf{x})$ is a neural network trained to approximate $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. Unlike density-estimation networks, score networks output a vector of the same dimension as the input and require no normalisation constraint. Training a score network is the subject of [Score Matching](score_matching.md).

## Practical Challenges

In low-density regions where $p(\mathbf{x}) \approx 0$, few training samples exist and score estimates are unreliable. When data concentrates on a low-dimensional manifold, the score is undefined off the manifold entirely. The solution is to work with a **noise-perturbed distribution** $p_\sigma(\mathbf{x}) = \int p(\mathbf{y}) \, \mathcal{N}(\mathbf{x}; \mathbf{y}, \sigma^2 \mathbf{I}) \, d\mathbf{y}$, which has full support and well-defined scores everywhere. This is the foundation of [Denoising Score Matching](denoising_score_matching.md) and diffusion models.

## PyTorch Implementation

### Analytical Scores

```python
import torch
import numpy as np


def score_gaussian(
    x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor
) -> torch.Tensor:
    """Score for multivariate Gaussian: s(x) = -Σ⁻¹(x - μ).

    Args:
        x: Input points [N, D].
        mu: Mean vector [D].
        cov: Covariance matrix [D, D].

    Returns:
        Score vectors [N, D].
    """
    cov_inv = torch.linalg.inv(cov)
    return -torch.matmul(x - mu, cov_inv.T)


def score_gmm(
    x: torch.Tensor, weights: torch.Tensor,
    means: torch.Tensor, covs: torch.Tensor
) -> torch.Tensor:
    """Score for Gaussian mixture: s(x) = Σ_k p(k|x) s_k(x).

    Args:
        x: Input points [N, D].
        weights: Mixture weights [K].
        means: Component means [K, D].
        covs: Component covariances [K, D, D].

    Returns:
        Mixture score [N, D].
    """
    N, D = x.shape
    K = len(weights)

    log_resps = torch.zeros(N, K)
    comp_scores = torch.zeros(N, K, D)

    for k in range(K):
        diff = x - means[k]
        cov_inv = torch.linalg.inv(covs[k])
        log_det = torch.linalg.slogdet(covs[k])[1]
        mahal = torch.sum(diff @ cov_inv * diff, dim=1)
        log_resps[:, k] = (
            torch.log(weights[k])
            - 0.5 * D * np.log(2 * np.pi)
            - 0.5 * log_det
            - 0.5 * mahal
        )
        comp_scores[:, k] = -diff @ cov_inv.T

    log_resps = log_resps - torch.logsumexp(log_resps, dim=1, keepdim=True)
    resps = torch.exp(log_resps)
    return torch.sum(resps.unsqueeze(-1) * comp_scores, dim=1)
```

### Score via Automatic Differentiation

```python
def compute_score_autograd(
    x: torch.Tensor, log_prob_fn: callable
) -> torch.Tensor:
    """Compute s(x) = ∇_x log p(x) using autograd.

    Args:
        x: Input points [N, D].
        log_prob_fn: Maps [N, D] → [N] (log-density).

    Returns:
        Score vectors [N, D].
    """
    x = x.clone().requires_grad_(True)
    log_prob = log_prob_fn(x)
    score = torch.autograd.grad(log_prob.sum(), x, create_graph=True)[0]
    return score
```

### Visualising Score Fields

```python
import matplotlib.pyplot as plt

def plot_score_field(score_fn, xlim=(-3, 3), ylim=(-3, 3), n_points=20):
    """Visualise a 2-D score function as a quiver plot."""
    gx = torch.linspace(*xlim, n_points)
    gy = torch.linspace(*ylim, n_points)
    X, Y = torch.meshgrid(gx, gy, indexing='xy')
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        S = score_fn(pts)

    U = S[:, 0].reshape(n_points, n_points).numpy()
    V = S[:, 1].reshape(n_points, n_points).numpy()
    mag = np.sqrt(U**2 + V**2)

    plt.figure(figsize=(8, 6))
    plt.quiver(X.numpy(), Y.numpy(), U, V, mag, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Score magnitude')
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.title('Score Field')
    plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
```

## Exercises

1. **Laplace distribution.** Derive the score for $p(x) = \frac{1}{2b}\exp(-|x - \mu|/b)$. Note the discontinuity at $x = \mu$.

2. **Verify zero mean.** Numerically verify $\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})] \approx \mathbf{0}$ for a 2-D Gaussian mixture by Monte Carlo sampling.

3. **Temperature effects.** Implement temperature scaling and visualise score fields for $T \in \{0.5, 1.0, 2.0\}$.

4. **Fisher information.** Compute $\mathcal{I}(p)$ analytically for an isotropic Gaussian in $D$ dimensions and verify it equals $D / \sigma^2$.

## References

1. Hyvärinen, A. (2005). "Estimation of Non-Normalized Statistical Models by Score Matching." *JMLR*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*.
