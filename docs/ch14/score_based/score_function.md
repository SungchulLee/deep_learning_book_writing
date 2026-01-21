# Score Function Definition

The **score function** is a fundamental concept in score-based generative modeling that provides an alternative approach to density estimation and sampling. It is the mathematical foundation of diffusion models, enabling sampling from complex distributions without computing intractable normalizing constants.

## Definition

For a probability distribution $p(x)$, the **score function** is defined as the gradient of the log-density:

$$s(x) = \nabla_x \log p(x)$$

This vector field points in the direction of increasing probability density at each point in space.

## Key Properties

### Relationship to Density

The score function encodes the same information as the density $p(x)$ up to a normalization constant. Since:

$$\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$

the score captures the relative rate of change of the density.

### Normalization-Free

A crucial advantage: the score function **does not depend on the normalization constant** of $p(x)$. If $p(x) = \frac{\tilde{p}(x)}{Z}$ where $Z = \int \tilde{p}(x) dx$, then:

$$\nabla_x \log p(x) = \nabla_x \log \tilde{p}(x) - \nabla_x \log Z = \nabla_x \log \tilde{p}(x)$$

Since $Z$ is constant with respect to $x$, it vanishes under differentiation. This property makes score-based methods particularly useful when the normalization constant is intractable.

### Integral Constraint

For any valid probability density, the score satisfies:

$$\mathbb{E}_{p(x)}[s(x)] = \int p(x) \nabla_x \log p(x) dx = 0$$

### Fisher Information

The expected squared norm of the score defines the **Fisher information**:

$$\mathcal{I}(p) = \mathbb{E}_{p(x)}\left[\|s(x)\|^2\right]$$

Fisher information measures how much information the data carries about the underlying distribution.

## Score Function for Common Distributions

### For Gaussian Distributions

For $p(x) = \mathcal{N}(x; \mu, \Sigma)$:

$$s(x) = -\Sigma^{-1}(x - \mu)$$

The score points toward the mean, with magnitude inversely proportional to variance.

### For Mixture Models

For a mixture $p(x) = \sum_k \pi_k p_k(x)$:

$$s(x) = \frac{\sum_k \pi_k p_k(x) s_k(x)}{\sum_k \pi_k p_k(x)}$$

The mixture score is a weighted average of component scores.

## Why Learn the Score?

### Langevin Dynamics Sampling

Given access to $s(x) = \nabla_x \log p(x)$, we can generate samples via **Langevin dynamics**:

$$x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} z_t, \quad z_t \sim \mathcal{N}(0, I)$$

As $\epsilon \to 0$ and $t \to \infty$, samples converge to the target distribution.

### Connection to Diffusion Models

In diffusion models, the score function at different noise levels guides the reverse (denoising) process. The network $\epsilon_\theta(x_t, t)$ in DDPM is related to the score by:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

This establishes that diffusion models are fundamentally learning score functions at multiple noise scales.

## Score Networks

A **score network** $s_\theta(x)$ is a neural network trained to approximate $\nabla_x \log p(x)$. Unlike density estimation networks, score networks:

1. Output vectors (same dimension as input)
2. Do not need to integrate to 1
3. Can be trained without knowing the normalization constant

## PyTorch Implementation

```python
"""
Score Function Fundamentals
===========================
"""

import torch
import numpy as np


def score_gaussian_1d(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """Analytical score for 1D Gaussian: s(x) = -(x - μ) / σ²"""
    return -(x - mu) / (sigma ** 2)


def score_gaussian_nd(x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Analytical score for multivariate Gaussian: s(x) = -Σ⁻¹(x - μ)"""
    cov_inv = torch.linalg.inv(cov)
    return -torch.matmul(x - mu, cov_inv.T)


def score_gmm(x: torch.Tensor, weights: torch.Tensor, means: torch.Tensor, covs: torch.Tensor) -> torch.Tensor:
    """Analytical score for Gaussian Mixture Model."""
    batch_size, dim = x.shape
    K = len(weights)
    
    log_resps = torch.zeros(batch_size, K)
    component_scores = torch.zeros(batch_size, K, dim)
    
    for k in range(K):
        diff = x - means[k]
        cov_inv = torch.linalg.inv(covs[k])
        log_det = torch.linalg.slogdet(covs[k])[1]
        mahal = torch.sum(diff @ cov_inv * diff, dim=1)
        log_resps[:, k] = torch.log(weights[k]) - 0.5 * dim * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * mahal
        component_scores[:, k] = -diff @ cov_inv.T
    
    log_resps = log_resps - torch.logsumexp(log_resps, dim=1, keepdim=True)
    resps = torch.exp(log_resps)
    return torch.sum(resps.unsqueeze(-1) * component_scores, dim=1)


def compute_score_autograd(x: torch.Tensor, log_prob_fn: callable) -> torch.Tensor:
    """Compute score using automatic differentiation."""
    x = x.clone().requires_grad_(True)
    log_prob = log_prob_fn(x)
    return torch.autograd.grad(log_prob.sum(), x, create_graph=True)[0]
```

## Practical Considerations

### Challenges with Raw Score Estimation

In low-density regions, the score function is poorly defined and difficult to estimate. This motivates:

1. **Noise Conditional Score Networks (NCSN)**: Train separate scores for different noise levels
2. **Denoising Score Matching**: Estimate scores of noise-perturbed distributions
3. **Annealed Langevin Dynamics**: Sample through decreasing noise levels

### The Manifold Problem

Real data often lies on low-dimensional manifolds where $p(x) = 0$ off the manifold. The solution is **noise perturbation**:

$$p_\sigma(x) = \int p(y) \mathcal{N}(x; y, \sigma^2 I) dy$$

The perturbed distribution has full support, making the score well-defined everywhere.

## Connection to Energy-Based Models

Energy-based models define $p(x) = \frac{1}{Z} \exp(-E(x))$. The score is:

$$s(x) = -\nabla_x E(x)$$

This connects score-based methods to the broader class of energy-based models.

## Summary

The score function $s(x) = \nabla_x \log p(x)$ provides a normalization-free representation of probability distributions. Learning scores enables sampling via Langevin dynamics and forms the theoretical foundation for diffusion models. The key insight is that predicting noise in diffusion models is equivalent to estimating score functions at multiple noise scales.

## Exercises

1. **Score Derivation**: Derive the score function for the Laplace distribution $p(x) = \frac{1}{2b}\exp(-\frac{|x-\mu|}{b})$.

2. **Fisher Information**: Compute the Fisher information for a 1D Gaussian $\mathcal{N}(\mu, \sigma^2)$.

3. **GMM Visualization**: Implement score visualization for a mixture of three Gaussians.

## References

1. Hyvärinen, A. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. JMLR.
2. Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution.
