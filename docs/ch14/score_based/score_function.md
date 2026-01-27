# Score Function Definition

The **score function** is the mathematical foundation of score-based generative modeling and diffusion models. It provides an alternative approach to density estimation that elegantly sidesteps the intractable normalization constants that plague traditional probabilistic models.

## Learning Objectives

By the end of this section, you will be able to:

1. Define the score function mathematically and understand its geometric interpretation
2. Explain why score functions enable tractable generative modeling (normalization-free property)
3. Compute score functions analytically for common distributions
4. Implement score computation using both analytical formulas and automatic differentiation
5. Connect score functions to Langevin dynamics, diffusion models, and energy-based models

## Prerequisites

- Multivariate calculus (gradients, Jacobians)
- Probability theory (density functions, Gaussian distributions)
- PyTorch fundamentals (tensors, autograd)

---

## 1. Mathematical Definition

### 1.1 The Score Function

For a probability distribution $p(\mathbf{x})$, the **score function** is defined as the gradient of the log-density:

$$
\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})
$$

This vector field points in the direction of **increasing probability density** at each point in space.

### 1.2 Relationship to Density

The score function encodes the same information as the density $p(\mathbf{x})$ up to a normalization constant. Using the chain rule:

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})}
$$

The score captures the **relative rate of change** of the density—how quickly probability is changing at each point.

### 1.3 Geometric Interpretation

The score function has an elegant geometric meaning:

| Property | Description |
|----------|-------------|
| **Direction** | Points toward higher probability regions |
| **Magnitude** | $\|\mathbf{s}(\mathbf{x})\|$ indicates steepness of log-density |
| **At modes** | $\mathbf{s}(\mathbf{x}^*) = \mathbf{0}$ (zero gradient at peaks) |
| **Away from modes** | Larger magnitude, stronger "pull" toward high-density |

```
    Low p(x)                    High p(x)
        ·  →  →  →  →  →  ·  (mode, score ≈ 0)
              ↗        ↖
             ↗          ↖
    Scores point toward the mode from all directions
```

---

## 2. Why the Score Function Matters

### 2.1 The Normalization Problem

Traditional density estimation requires computing:

$$
p(\mathbf{x}) = \frac{\tilde{p}(\mathbf{x})}{Z}, \quad \text{where } Z = \int \tilde{p}(\mathbf{x}) \, d\mathbf{x}
$$

The normalization constant $Z$ is often **intractable** for complex models (e.g., deep energy-based models).

### 2.2 The Score Solution

The score function **does not depend on $Z$**:

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x}) - \nabla_{\mathbf{x}} \log Z = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x})
$$

Since $Z$ is constant with respect to $\mathbf{x}$, it vanishes under differentiation!

!!! success "Key Insight"
    Score-based models can work with **unnormalized** densities. We only need gradients, which are typically easy to compute via automatic differentiation.

### 2.3 Connection to Energy-Based Models

Energy-based models define:

$$
p(\mathbf{x}) = \frac{1}{Z} \exp(-E(\mathbf{x}))
$$

The score is simply the **negative energy gradient**:

$$
\mathbf{s}(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})
$$

This connects score-based methods to the broader class of energy-based models, with the score pointing "downhill" in the energy landscape.

---

## 3. Key Properties

### 3.1 Zero Score at Modes

At any local maximum (mode) $\mathbf{x}^*$ of the density:

$$
\mathbf{s}(\mathbf{x}^*) = \nabla_{\mathbf{x}} \log p(\mathbf{x})\big|_{\mathbf{x}=\mathbf{x}^*} = \mathbf{0}
$$

This follows from the first-order optimality condition—at a peak, there's no direction of ascent.

### 3.2 Zero Expected Score (Integral Constraint)

The expected score under the data distribution is always zero:

$$
\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})] = \int p(\mathbf{x}) \nabla_{\mathbf{x}} \log p(\mathbf{x}) \, d\mathbf{x} = \int \nabla_{\mathbf{x}} p(\mathbf{x}) \, d\mathbf{x} = \mathbf{0}
$$

**Proof**: Using $\nabla_{\mathbf{x}} \log p = \frac{\nabla_{\mathbf{x}} p}{p}$:

$$
\int p(\mathbf{x}) \cdot \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})} d\mathbf{x} = \int \nabla_{\mathbf{x}} p(\mathbf{x}) d\mathbf{x} = \nabla_{\mathbf{x}} \int p(\mathbf{x}) d\mathbf{x} = \nabla_{\mathbf{x}} 1 = \mathbf{0}
$$

!!! note "Intuition"
    The score "pushes" are balanced—probability flowing toward high-density regions equals probability flowing away.

### 3.3 Fisher Information

The **Fisher Information** measures the expected squared magnitude of the score:

$$
\mathcal{I}(p) = \mathbb{E}_{p(\mathbf{x})}\left[\|\mathbf{s}(\mathbf{x})\|^2\right]
$$

| Property | Interpretation |
|----------|----------------|
| High Fisher info | Sharp distribution, scores have large magnitudes |
| Low Fisher info | Flat distribution, scores are small everywhere |
| For $\mathcal{N}(\mu, \sigma^2)$ | $\mathcal{I} = 1/\sigma^2$ |

Fisher information quantifies how much information the data carries about the underlying distribution.

### 3.4 Temperature Scaling

For a "tempered" distribution $p_T(\mathbf{x}) \propto p(\mathbf{x})^{1/T}$:

$$
\mathbf{s}_T(\mathbf{x}) = \frac{1}{T} \mathbf{s}(\mathbf{x})
$$

| Temperature | Effect on Score | Distribution Shape |
|-------------|-----------------|-------------------|
| $T < 1$ (cold) | Larger scores | Sharper peaks |
| $T = 1$ | Original | Original |
| $T > 1$ (hot) | Smaller scores | Flatter landscape |

---

## 4. Score Functions for Common Distributions

### 4.1 Univariate Gaussian

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**Derivation:**

$$
\log p(x) = -\frac{(x-\mu)^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)
$$

$$
s(x) = \frac{d}{dx}\log p(x) = -\frac{x - \mu}{\sigma^2}
$$

**Interpretation:**
- Linear function of $x$
- Points toward the mean $\mu$
- Magnitude proportional to distance from mean
- Inversely proportional to variance (sharper distributions → larger scores)

### 4.2 Multivariate Gaussian

For $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$
\log p(\mathbf{x}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) + \text{const}
$$

$$
\boxed{\mathbf{s}(\mathbf{x}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}
$$

For isotropic Gaussian with $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$:

$$
\mathbf{s}(\mathbf{x}) = -\frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2}
$$

### 4.3 Gaussian Mixture Model

For a mixture $p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$:

$$
\mathbf{s}(\mathbf{x}) = \frac{\sum_k \pi_k p_k(\mathbf{x}) \mathbf{s}_k(\mathbf{x})}{\sum_k \pi_k p_k(\mathbf{x})} = \sum_{k=1}^K w_k(\mathbf{x}) \cdot \mathbf{s}_k(\mathbf{x})
$$

where:
- $w_k(\mathbf{x}) = p(k|\mathbf{x})$ is the **posterior probability** of component $k$
- $\mathbf{s}_k(\mathbf{x}) = -\boldsymbol{\Sigma}_k^{-1}(\mathbf{x} - \boldsymbol{\mu}_k)$ is the score of component $k$

The posterior is computed via Bayes' rule:

$$
w_k(\mathbf{x}) = \frac{\pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

!!! warning "Nonlinearity in Mixtures"
    Unlike single Gaussians, mixture scores are **nonlinear** functions. Between modes, the score interpolates between component scores based on proximity—it's a "soft" weighted average.

---

## 5. Connection to Generative Modeling

### 5.1 Langevin Dynamics Sampling

Given access to $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$, we can generate samples via **Langevin dynamics**:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\epsilon}{2} \mathbf{s}(\mathbf{x}_t) + \sqrt{\epsilon} \, \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

| Term | Role |
|------|------|
| $\frac{\epsilon}{2} \mathbf{s}(\mathbf{x}_t)$ | Gradient ascent toward high probability |
| $\sqrt{\epsilon} \, \mathbf{z}_t$ | Noise for exploration |

As $\epsilon \to 0$ and $t \to \infty$, samples converge to the target distribution $p(\mathbf{x})$.

### 5.2 Connection to Diffusion Models

In diffusion models, the score of the noisy distribution relates directly to the added noise:

$$
\mathbf{s}(\mathbf{x}_t, t) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\boldsymbol{\epsilon}}{\sigma_t}
$$

where $\boldsymbol{\epsilon}$ is the noise added at time $t$.

**This is the key insight:** The DDPM noise predictor $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ is equivalent to learning the score:

$$
\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

Predicting noise = Estimating score!

### 5.3 Score Networks

A **score network** $\mathbf{s}_\theta(\mathbf{x})$ is a neural network trained to approximate $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. Unlike density estimation networks:

| Property | Density Network | Score Network |
|----------|-----------------|---------------|
| Output | Scalar (log-prob) | Vector (same dim as input) |
| Constraint | Must integrate to 1 | No normalization needed |
| Training | Requires $Z$ | Works with unnormalized $\tilde{p}$ |

---

## 6. Practical Challenges

### 6.1 The Low-Density Problem

In low-density regions where $p(\mathbf{x}) \approx 0$:
- Few training samples exist
- Score estimates are unreliable
- Langevin sampling can get stuck

### 6.2 The Manifold Problem

Real data often lies on low-dimensional manifolds where $p(\mathbf{x}) = 0$ off the manifold. The score is **undefined** in these regions.

### 6.3 Solution: Noise Perturbation

The solution is to work with a **noise-perturbed distribution**:

$$
p_\sigma(\mathbf{x}) = \int p(\mathbf{y}) \mathcal{N}(\mathbf{x}; \mathbf{y}, \sigma^2 \mathbf{I}) \, d\mathbf{y}
$$

This smoothed distribution:
- Has **full support** (defined everywhere)
- Makes the score well-defined in all regions
- Is the foundation of **denoising score matching** and **diffusion models**

---

## 7. PyTorch Implementation

### 7.1 Analytical Score for Gaussian

```python
import torch
import numpy as np

def score_gaussian_1d(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """
    Analytical score for 1D Gaussian: s(x) = -(x - μ) / σ²
    """
    return -(x - mu) / (sigma ** 2)


def score_gaussian_nd(
    x: torch.Tensor, 
    mu: torch.Tensor, 
    cov: torch.Tensor
) -> torch.Tensor:
    """
    Analytical score for multivariate Gaussian: s(x) = -Σ⁻¹(x - μ)
    
    Args:
        x: Input points, shape (N, D)
        mu: Mean vector, shape (D,)
        cov: Covariance matrix, shape (D, D)
    
    Returns:
        Score vectors, shape (N, D)
    """
    cov_inv = torch.linalg.inv(cov)
    return -torch.matmul(x - mu, cov_inv.T)


def score_isotropic_gaussian(
    x: torch.Tensor, 
    mu: torch.Tensor, 
    sigma: float
) -> torch.Tensor:
    """
    Score for isotropic Gaussian N(μ, σ²I): s(x) = -(x - μ) / σ²
    """
    return -(x - mu) / (sigma ** 2)
```

### 7.2 Score via Automatic Differentiation

```python
def compute_score_autograd(
    x: torch.Tensor, 
    log_prob_fn: callable
) -> torch.Tensor:
    """
    Compute score using automatic differentiation.
    
    This demonstrates the definition: s(x) = ∇_x log p(x)
    
    Args:
        x: Input points, shape (N, D)
        log_prob_fn: Function computing log p(x)
    
    Returns:
        Score vectors, shape (N, D)
    """
    x = x.clone().requires_grad_(True)
    log_prob = log_prob_fn(x)
    score = torch.autograd.grad(
        log_prob.sum(), 
        x, 
        create_graph=True
    )[0]
    return score


# Example usage
def gaussian_log_prob(x, mu=0.0, sigma=1.0):
    return -0.5 * torch.sum((x - mu) ** 2, dim=-1) / (sigma ** 2)

# x = torch.randn(100, 2)
# score = compute_score_autograd(x, gaussian_log_prob)
```

### 7.3 Gaussian Mixture Score

```python
def score_gmm(
    x: torch.Tensor, 
    weights: torch.Tensor, 
    means: torch.Tensor, 
    covs: torch.Tensor
) -> torch.Tensor:
    """
    Analytical score for Gaussian Mixture Model.
    
    s(x) = Σ_k p(k|x) · s_k(x)
    
    Args:
        x: Input points, shape (N, D)
        weights: Mixture weights, shape (K,)
        means: Component means, shape (K, D)
        covs: Component covariances, shape (K, D, D)
    
    Returns:
        Mixture score, shape (N, D)
    """
    N, D = x.shape
    K = len(weights)
    
    # Compute log responsibilities and component scores
    log_resps = torch.zeros(N, K)
    component_scores = torch.zeros(N, K, D)
    
    for k in range(K):
        diff = x - means[k]
        cov_inv = torch.linalg.inv(covs[k])
        log_det = torch.linalg.slogdet(covs[k])[1]
        
        # Mahalanobis distance
        mahal = torch.sum(diff @ cov_inv * diff, dim=1)
        
        # Log responsibility (unnormalized)
        log_resps[:, k] = (
            torch.log(weights[k]) 
            - 0.5 * D * np.log(2 * np.pi) 
            - 0.5 * log_det 
            - 0.5 * mahal
        )
        
        # Component score
        component_scores[:, k] = -diff @ cov_inv.T
    
    # Normalize responsibilities
    log_resps = log_resps - torch.logsumexp(log_resps, dim=1, keepdim=True)
    resps = torch.exp(log_resps)
    
    # Weighted sum of component scores
    return torch.sum(resps.unsqueeze(-1) * component_scores, dim=1)
```

### 7.4 Visualizing Score Fields

```python
import matplotlib.pyplot as plt

def plot_score_field(
    score_fn: callable,
    xlim: tuple = (-3, 3),
    ylim: tuple = (-3, 3),
    n_points: int = 20,
    title: str = "Score Function Vector Field"
):
    """
    Visualize 2D score function as a vector field.
    
    Args:
        score_fn: Function mapping (N, 2) tensor to (N, 2) scores
        xlim, ylim: Plot limits
        n_points: Grid resolution
    """
    # Create grid
    x = torch.linspace(xlim[0], xlim[1], n_points)
    y = torch.linspace(ylim[0], ylim[1], n_points)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # Compute scores at grid points
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        scores = score_fn(grid_points)
    
    # Reshape for plotting
    U = scores[:, 0].reshape(n_points, n_points).numpy()
    V = scores[:, 1].reshape(n_points, n_points).numpy()
    magnitude = np.sqrt(U**2 + V**2)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.quiver(X.numpy(), Y.numpy(), U, V, magnitude, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Score Magnitude')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()
```

---

## 8. Summary

| Concept | Definition | Key Property |
|---------|------------|--------------|
| **Score Function** | $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$ | Points toward high probability |
| **Normalization-free** | $\nabla \log(p/Z) = \nabla \log p$ | No need to compute $Z$ |
| **Gaussian Score** | $-\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ | Linear, points toward mean |
| **Mixture Score** | $\sum_k w_k(\mathbf{x}) \mathbf{s}_k(\mathbf{x})$ | Posterior-weighted average |
| **Fisher Information** | $\mathbb{E}[\|\mathbf{s}(\mathbf{x})\|^2]$ | Measures distribution "sharpness" |
| **Zero mean** | $\mathbb{E}_p[\mathbf{s}(\mathbf{x})] = \mathbf{0}$ | Integral constraint |

!!! tip "Key Takeaways"
    1. **The score captures gradient structure** without the normalization constant
    2. **Learning scores enables sampling** via Langevin dynamics
    3. **Diffusion models are score learners**—predicting noise = estimating score
    4. **Noise perturbation solves** the low-density and manifold problems
    5. **Score networks output vectors**, not scalars, matching input dimension

---

## Exercises

1. **Laplace Distribution**: Derive the score function for $p(x) = \frac{1}{2b}\exp(-|x-\mu|/b)$. Note the discontinuity at $x = \mu$.

2. **Verify Zero Mean**: Numerically verify $\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})] \approx \mathbf{0}$ for a 2D Gaussian mixture by sampling.

3. **Temperature Effects**: Implement temperature scaling and visualize score fields for $T \in \{0.5, 1.0, 2.0\}$.

4. **Fisher Information**: Compute the Fisher information for a 2D isotropic Gaussian and verify it equals $D/\sigma^2 = 2/\sigma^2$.

5. **GMM Visualization**: Create an animated visualization showing how the score field changes as you move between mixture components.

---

## References

1. Hyvärinen, A. (2005). "Estimation of Non-Normalized Statistical Models by Score Matching." *JMLR*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*.
