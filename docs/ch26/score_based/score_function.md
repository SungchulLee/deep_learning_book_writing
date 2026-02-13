# Score Function

## Overview

The **score function** is the gradient of the log-density. It is the mathematical foundation of score-based generative modeling: because the gradient eliminates the intractable normalization constant, a model that learns the score can generate samples without ever evaluating the partition function. This section defines the score, develops its key properties, computes it analytically for common distributions, connects it to energy-based models and Langevin sampling, and explains why noise perturbation is essential for making score estimation practical—setting up the training methods in [Score Matching](score_matching.md), [Denoising Score Matching](denoising_score_matching.md), and [Sliced Score Matching](sliced_score_matching.md).

---

## 1. Definition

For a probability distribution with density $p(\mathbf{x})$, the **score function** (or simply "score") is:

$$\mathbf{s}(\mathbf{x}) \;\triangleq\; \nabla_{\mathbf{x}} \log p(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})}$$

The score is a **vector field** over the data space $\mathbb{R}^D$: at every point $\mathbf{x}$ it returns a $D$-dimensional vector pointing in the direction of steepest ascent of the log-density.

### Geometric Interpretation

- At a **mode** $\mathbf{x}^*$ (local maximum of $p$), the score is zero — there is no direction of further ascent.
- **Away from modes**, the score points toward higher-density regions, with magnitude proportional to the steepness of the log-density landscape.
- The score field can be visualized as a "flow" that carries probability mass toward the modes. Following this flow with added noise is precisely how Langevin dynamics generates samples.

---

## 2. Why the Score Matters: Normalization-Free Modeling

### 2.1 Eliminating the Partition Function

In many models the density is specified only up to a normalization constant:

$$p(\mathbf{x}) = \frac{\tilde{p}(\mathbf{x})}{Z}, \quad Z = \int \tilde{p}(\mathbf{x}) \, d\mathbf{x}$$

Computing $Z$ is intractable for complex models (energy-based models, Boltzmann machines, deep latent variable models). The score eliminates this problem entirely:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}} \log Z}_{=\, \mathbf{0}} = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x})$$

Since $Z$ does not depend on $\mathbf{x}$, it vanishes under differentiation. This is the fundamental reason score-based methods can work with unnormalized densities — and it is the same reason that [MCMC methods](../../ch15/mcmc/metropolis_hastings.md) use acceptance ratios (which also cancel $Z$) and why the DDPM [training objective](../foundations/training_objective.md) avoids computing the marginal likelihood.

### 2.2 Connection to Energy-Based Models

An energy-based model defines $p(\mathbf{x}) = Z^{-1} \exp(-E(\mathbf{x}))$. Its score is simply the negative energy gradient:

$$\mathbf{s}(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$$

The score points "downhill" in the energy landscape — toward lower energy and higher probability. Training a score network is therefore equivalent to learning the gradient of an implicit energy function, without ever specifying or normalizing the energy itself.

---

## 3. Key Properties

### 3.1 Zero Score at Modes

At any local maximum $\mathbf{x}^*$ of the density, $\mathbf{s}(\mathbf{x}^*) = \mathbf{0}$ by the first-order optimality condition. More generally, the score is zero at any critical point (modes, saddle points), but only modes are stable fixed points of Langevin dynamics.

### 3.2 Zero Expected Score

The expected score under the data distribution is always zero:

$$\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})] = \int p(\mathbf{x}) \, \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})} \, d\mathbf{x} = \int \nabla_{\mathbf{x}} p(\mathbf{x}) \, d\mathbf{x} = \nabla_{\mathbf{x}} \int p(\mathbf{x}) \, d\mathbf{x} = \mathbf{0}$$

The last step exchanges integration and differentiation (valid under mild regularity conditions) and uses $\int p = 1$. Intuitively, the "pushes" toward high-density regions are exactly balanced by "pushes" away from them when averaged over the entire distribution.

This property is used in [score matching](score_matching.md) to derive the integration-by-parts trick that eliminates the need for the unknown true score in the training objective.

### 3.3 Fisher Information

The expected squared norm of the score is the **(Fisher) information**:

$$\mathcal{I}(p) = \mathbb{E}_{p(\mathbf{x})}\!\left[\|\mathbf{s}(\mathbf{x})\|^2\right]$$

This scalar measures the "sharpness" of the distribution:

| Distribution | Fisher Information | Interpretation |
|-------------|-------------------|----------------|
| $\mathcal{N}(\mu, \sigma^2)$ in 1D | $1/\sigma^2$ | Sharper (smaller $\sigma$) = more information |
| $\mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$ in $D$ dimensions | $D/\sigma^2$ | Scales linearly with dimension |
| Uniform on $[a, b]$ | 0 (interior) | Flat density carries no score information |

Fisher information connects score-based methods to the Cramér-Rao bound in statistics and to the Fisher divergence used in [score matching](score_matching.md) objectives.

### 3.4 Score of the Hessian: Curvature Information

The Jacobian of the score gives curvature information:

$$\nabla_{\mathbf{x}} \mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}}^2 \log p(\mathbf{x})$$

This is the Hessian of the log-density. At a mode, it equals the negative precision matrix for Gaussians. This quantity appears explicitly in the [score matching](score_matching.md) objective, where its trace $\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}(\mathbf{x}))$ replaces the intractable comparison with the true score.

### 3.5 Temperature Scaling

For a tempered distribution $p_T(\mathbf{x}) \propto p(\mathbf{x})^{1/T}$, the score scales linearly:

$$\mathbf{s}_T(\mathbf{x}) = \frac{1}{T}\, \mathbf{s}(\mathbf{x})$$

- Low temperature ($T < 1$): sharpens the landscape, amplifies scores, concentrates samples near modes
- High temperature ($T > 1$): flattens the landscape, attenuates scores, increases sample diversity

This temperature-score relationship underlies [classifier guidance](../conditional/classifier_guidance.md), where the guidance scale effectively controls the temperature of the conditional distribution.

### 3.6 Stein's Identity

For any smooth vector-valued function $\boldsymbol{\phi}(\mathbf{x})$ with suitable decay at infinity:

$$\mathbb{E}_{p}\!\left[\mathbf{s}(\mathbf{x}) \, \boldsymbol{\phi}(\mathbf{x})^\top + \nabla_{\mathbf{x}} \boldsymbol{\phi}(\mathbf{x})\right] = \mathbf{0}$$

This identity is the basis of **Stein discrepancies** and **kernelized Stein discrepancies**, which provide goodness-of-fit tests that depend only on the score (not the normalization constant). Stein's identity also underlies the integration-by-parts trick used to derive the [score matching](score_matching.md) objective.

---

## 4. Score Functions for Common Distributions

### 4.1 Univariate Gaussian

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$s(x) = -\frac{x - \mu}{\sigma^2}$$

The score is linear in $x$, pointing toward the mean with magnitude inversely proportional to the variance. This is the simplest example of the general principle: the score "pulls" toward high-density regions.

### 4.2 Multivariate Gaussian

For $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$\boxed{\mathbf{s}(\mathbf{x}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$$

For the isotropic case $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$, this simplifies to $\mathbf{s}(\mathbf{x}) = -(\mathbf{x} - \boldsymbol{\mu})/\sigma^2$. This formula is central to DDPM: the [forward process](../foundations/forward_process.md) produces $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)\, \mathbf{I})$, whose score is:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

where $\epsilon$ is the noise added in the forward process. This directly connects the score to DDPM's noise prediction: $\mathbf{s}_\theta(x_t, t) = -\epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}$.

### 4.3 Gaussian Mixture Model

For $p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$, the score is a **responsibility-weighted** average of component scores:

$$\mathbf{s}(\mathbf{x}) = \sum_{k=1}^K w_k(\mathbf{x})\, \mathbf{s}_k(\mathbf{x})$$

where $w_k(\mathbf{x}) = p(k \mid \mathbf{x})$ is the posterior responsibility of component $k$ and $\mathbf{s}_k(\mathbf{x}) = -\boldsymbol{\Sigma}_k^{-1}(\mathbf{x} - \boldsymbol{\mu}_k)$.

Unlike single-Gaussian scores, mixture scores are **nonlinear**: between modes, the score interpolates between component scores based on proximity, creating curved flow lines that route samples toward the nearest mode. This nonlinearity is why neural networks are needed to approximate real-world score functions.

### 4.4 Laplace Distribution

For $X \sim \text{Laplace}(\mu, b)$ with $p(x) = \frac{1}{2b}\exp(-|x-\mu|/b)$:

$$s(x) = \begin{cases} +1/b & x < \mu \\ -1/b & x > \mu \end{cases}$$

The score is piecewise constant with a discontinuity at the mode. This illustrates that scores can be non-smooth, which is one reason neural score networks use smooth activations.

---

## 5. Connection to Langevin Dynamics

Given access to the score, we can generate samples via **Langevin Monte Carlo**:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\eta}{2}\, \mathbf{s}(\mathbf{x}_t) + \sqrt{\eta}\, \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The gradient term pushes samples toward high-probability regions; the noise term ensures exploration and prevents collapse to a single mode. As $\eta \to 0$ and $t \to \infty$, the distribution of $\mathbf{x}_t$ converges to $p(\mathbf{x})$.

This is the key insight connecting score functions to generative modeling: **if we can estimate the score, we can sample from the distribution**. The full development of Langevin-based sampling is in [Langevin Fundamentals](../sde/fundamentals.md), and its role in diffusion models is covered in the [SDE framework](../sde/vp_sde.md).

### Why Langevin Alone Is Not Enough

In practice, Langevin dynamics with the score of the raw data distribution fails for two reasons:

1. **Low-density regions**: The score is poorly estimated where few data points exist, causing Langevin chains to get "lost" between modes.
2. **Manifold hypothesis**: Real data often lies on a low-dimensional manifold where the ambient-space score is undefined.

The solution — adding noise at multiple scales — is the foundation of [Denoising Score Matching](denoising_score_matching.md) and the connection to diffusion models.

---

## 6. Noise-Perturbed Scores and Multi-Scale Estimation

### 6.1 The Noise Perturbation Strategy

To make score estimation practical, we work with a family of **noise-perturbed distributions**:

$$p_\sigma(\mathbf{x}) = \int p(\mathbf{y})\, \mathcal{N}(\mathbf{x};\, \mathbf{y},\, \sigma^2 \mathbf{I})\, d\mathbf{y}$$

The perturbed distribution $p_\sigma$ has full support in $\mathbb{R}^D$ (no manifold issues) and well-defined scores everywhere. Its score can be shown to equal:

$$\nabla_{\mathbf{x}} \log p_\sigma(\mathbf{x}) = \mathbb{E}_{p(\mathbf{y} \mid \mathbf{x})}\!\left[-\frac{\mathbf{x} - \mathbf{y}}{\sigma^2}\right]$$

This is the **expected denoising direction**: the score of the noisy distribution points toward the expected clean data point.

### 6.2 Connection to DDPM

The [DDPM forward process](../foundations/forward_process.md) creates a sequence of noise-perturbed distributions $q(x_t)$ indexed by timestep $t$ rather than noise level $\sigma$. The mapping is:

$$\sigma_t^2 = \frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}$$

Training a DDPM noise predictor $\epsilon_\theta(x_t, t)$ is equivalent to learning the score $\nabla_{x_t} \log q(x_t)$ at each noise level. The [SDE framework](../sde/fundamentals.md) formalizes this connection, showing that DDPM is a discretization of a continuous-time score-based diffusion process.

### 6.3 Multi-Scale Score Structure

| Noise Level | Score Behavior | What the Model Learns |
|-------------|---------------|----------------------|
| High $\sigma$ (large $t$) | Smooth, long-range, points toward data center of mass | Global structure: layout, dominant features |
| Medium $\sigma$ | Moderate-range, captures cluster structure | Medium-scale features: object shapes, regions |
| Low $\sigma$ (small $t$) | Sharp, short-range, captures local modes | Fine details: edges, textures, high-frequency content |

This multi-scale structure is why diffusion models generate samples coarse-to-fine during the [reverse process](../foundations/reverse_process.md).

---

## 7. Score Networks

A **score network** $\mathbf{s}_\theta(\mathbf{x}, t)$ is a neural network trained to approximate $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ at each noise level $t$. Key design considerations:

- **Output dimension**: The score has the same dimension as the input ($D$-in, $D$-out), unlike density estimators which output a scalar.
- **No normalization constraint**: Unlike normalizing flows or autoregressive models, score networks have no architectural constraints ensuring a valid probability distribution.
- **Time conditioning**: The network must handle all noise levels, typically via sinusoidal time embeddings added to intermediate features.
- **Architecture**: [U-Net](../architectures/unet.md) is standard for images; [DiT](../architectures/dit.md) (Diffusion Transformers) are a recent alternative.

Training a score network is the subject of [Score Matching](score_matching.md) (explicit matching), [Denoising Score Matching](denoising_score_matching.md) (implicit matching via denoising), and [Sliced Score Matching](sliced_score_matching.md) (scalable random projections).

---

## 8. Implementation

### 8.1 Analytical Score Functions

```python
import torch
import numpy as np


def score_gaussian(
    x: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
) -> torch.Tensor:
    """
    Exact score for multivariate Gaussian: s(x) = −Σ⁻¹(x − μ).

    Parameters
    ----------
    x : Tensor of shape (N, D)
        Query points.
    mu : Tensor of shape (D,)
        Mean vector.
    cov : Tensor of shape (D, D)
        Covariance matrix.

    Returns
    -------
    score : Tensor of shape (N, D).
    """
    precision = torch.linalg.inv(cov)
    return -(x - mu) @ precision.T


def score_gmm(
    x: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    covs: torch.Tensor,
) -> torch.Tensor:
    """
    Exact score for Gaussian mixture model: s(x) = Σ_k p(k|x) s_k(x).

    Parameters
    ----------
    x : Tensor of shape (N, D)
    weights : Tensor of shape (K,)
        Mixture weights (must sum to 1).
    means : Tensor of shape (K, D)
    covs : Tensor of shape (K, D, D)

    Returns
    -------
    score : Tensor of shape (N, D).
    """
    N, D = x.shape
    K = weights.shape[0]

    log_resps = torch.zeros(N, K, device=x.device)
    comp_scores = torch.zeros(N, K, D, device=x.device)

    for k in range(K):
        diff = x - means[k]  # (N, D)
        precision = torch.linalg.inv(covs[k])
        log_det = torch.linalg.slogdet(covs[k]).logabsdet

        # Log-responsibility (unnormalized)
        mahal = (diff @ precision * diff).sum(dim=1)  # (N,)
        log_resps[:, k] = (
            torch.log(weights[k])
            - 0.5 * D * np.log(2 * np.pi)
            - 0.5 * log_det
            - 0.5 * mahal
        )
        # Component score
        comp_scores[:, k] = -diff @ precision.T

    # Normalize responsibilities via log-sum-exp
    log_resps = log_resps - torch.logsumexp(log_resps, dim=1, keepdim=True)
    resps = torch.exp(log_resps)  # (N, K)

    # Responsibility-weighted average of component scores
    return (resps.unsqueeze(-1) * comp_scores).sum(dim=1)
```

### 8.2 Score via Automatic Differentiation

For arbitrary log-densities, compute the score using PyTorch autograd:

```python
def score_autograd(
    x: torch.Tensor,
    log_prob_fn: callable,
) -> torch.Tensor:
    """
    Compute s(x) = ∇_x log p(x) via automatic differentiation.

    Parameters
    ----------
    x : Tensor of shape (N, D)
        Query points (will be cloned and require grad).
    log_prob_fn : callable
        Maps (N, D) → (N,) returning log p(x).

    Returns
    -------
    score : Tensor of shape (N, D).
    """
    x = x.detach().requires_grad_(True)
    log_p = log_prob_fn(x)
    score = torch.autograd.grad(
        log_p.sum(), x, create_graph=True
    )[0]
    return score
```

### 8.3 Visualizing Score Fields

```python
import matplotlib.pyplot as plt


def plot_score_field(
    score_fn: callable,
    xlim: tuple = (-4, 4),
    ylim: tuple = (-4, 4),
    n_grid: int = 20,
    title: str = "Score Field",
):
    """
    Visualize a 2D score function as a quiver plot.

    Parameters
    ----------
    score_fn : callable
        Maps (N, 2) → (N, 2).
    xlim, ylim : tuple
        Plot bounds.
    n_grid : int
        Grid resolution per axis.
    title : str
    """
    gx = torch.linspace(*xlim, n_grid)
    gy = torch.linspace(*ylim, n_grid)
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        S = score_fn(pts)

    U = S[:, 0].reshape(n_grid, n_grid).numpy()
    V = S[:, 1].reshape(n_grid, n_grid).numpy()
    mag = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver(X.numpy(), Y.numpy(), U, V, mag, cmap="viridis", alpha=0.8)
    fig.colorbar(q, ax=ax, label="Score magnitude")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
```

### 8.4 Demonstration: Gaussian Mixture Score Field

```python
torch.manual_seed(42)

# Define a 2D Gaussian mixture
weights = torch.tensor([0.4, 0.6])
means = torch.tensor([[-2.0, 0.0], [2.0, 1.0]])
covs = torch.stack([
    torch.tensor([[0.5, 0.2], [0.2, 0.5]]),
    torch.tensor([[0.8, -0.3], [-0.3, 0.6]]),
])

score_fn = lambda x: score_gmm(x, weights, means, covs)
fig = plot_score_field(score_fn, xlim=(-5, 5), ylim=(-3, 4), title="GMM Score Field")
```

The resulting quiver plot shows arrows converging toward both modes, with a "saddle" region between them where the score transitions from pointing toward one mode to the other. This nonlinear interpolation is what score networks must learn.

---

## 9. Finance Applications

Score functions have natural interpretations in quantitative finance:

| Financial Concept | Score Analogue | Application |
|------------------|---------------|-------------|
| Return distribution gradient | Score of return density | Direction of maximum likelihood movement |
| Risk landscape | Energy gradient | Portfolio risk sensitivity |
| Tail behavior | Score magnitude in tails | Extreme event characterization |
| Regime transitions | Score between mixture modes | Market regime change dynamics |
| Implied density from options | Score of risk-neutral density | Model-free Greeks and hedging |

The score of the return distribution tells us how the log-likelihood changes with small perturbations — essentially a sensitivity analysis of the distributional model. In [scenario generation](../finance/scenarios.md), score-based sampling produces realistic market scenarios by following the score field of the fitted return distribution, naturally respecting the correlation structure and tail properties.

---

## 10. Key Takeaways

1. **The score $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ eliminates the normalization constant**, enabling generative modeling without computing intractable partition functions.

2. **The expected score is zero** ($\mathbb{E}_p[\mathbf{s}] = \mathbf{0}$), and its expected squared norm equals the Fisher information — connecting score estimation to classical statistics.

3. **For Gaussians, the score is linear** ($\mathbf{s}(\mathbf{x}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$), which directly connects to DDPM's noise prediction: $\mathbf{s}_\theta \propto -\epsilon_\theta$.

4. **Mixture scores are nonlinear**, interpolating between component scores based on posterior responsibilities — this is why neural networks are needed for real-world score estimation.

5. **Noise perturbation is essential**: raw data scores are unreliable in low-density regions and undefined on manifolds. Adding noise at multiple scales creates well-defined scores everywhere and produces the multi-scale coarse-to-fine generation observed in diffusion models.

6. **The score is the bridge** between energy-based models, Langevin sampling, score matching, and diffusion models — all of which are unified in the [SDE framework](../sde/fundamentals.md).

---

## Exercises

### Exercise 1: Laplace Distribution Score

Derive the score for $p(x) = \frac{1}{2b}\exp(-|x - \mu|/b)$. Note the discontinuity at $x = \mu$. What does this imply about using smooth neural networks to approximate scores of non-smooth distributions?

### Exercise 2: Zero Mean Verification

Numerically verify $\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})] \approx \mathbf{0}$ for a 2D Gaussian mixture by sampling 10,000 points and computing the sample mean of the score. How does the error scale with sample size?

### Exercise 3: Temperature Effects on Score Fields

Implement temperature scaling and visualize GMM score fields for $T \in \{0.3, 0.5, 1.0, 2.0, 5.0\}$. At what temperature do the two modes effectively "merge" into a single basin of attraction?

### Exercise 4: Fisher Information

Compute $\mathcal{I}(p)$ analytically for an isotropic Gaussian in $D$ dimensions and verify numerically that $\mathcal{I} = D/\sigma^2$. Then compute it for a 2-component GMM and show that it exceeds the Gaussian case — explaining why mixtures carry more score information.

### Exercise 5: Score and DDPM Connection

For a 1D Gaussian $\mathcal{N}(0, 1)$ with DDPM forward process noise schedule, compute the exact score $\nabla_{x_t} \log q(x_t)$ at $t \in \{100, 500, 900\}$ (out of $T = 1000$). Verify that $\mathbf{s}(x_t, t) = -\epsilon / \sqrt{1 - \bar{\alpha}_t}$ by comparing with the analytical noise $\epsilon$.

### Exercise 6: Langevin Sampling with Exact Scores

Implement Langevin dynamics using the exact GMM score function. Generate 5,000 samples with step sizes $\eta \in \{0.01, 0.1, 0.5\}$ and 1,000 steps each. Compare the empirical distribution with the true GMM. At what step size does the sampler diverge?

---

## References

1. Hyvärinen, A. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. *Journal of Machine Learning Research*, 6, 695–709.
2. Song, Y. & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *Advances in Neural Information Processing Systems (NeurIPS)*.
3. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *Proceedings of the 9th International Conference on Learning Representations (ICLR)*.
4. Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. *Neural Computation*, 23(7), 1661–1674.
5. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*.
