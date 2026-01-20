# Score Function

## Learning Objectives

By the end of this section, you will be able to:

- Define the score function and explain its geometric interpretation
- Compute score functions for common probability distributions
- Understand why the score is independent of the normalization constant
- Connect the score function to gradient-based sampling and modern generative models
- Implement score computation in PyTorch with automatic differentiation

## Definition and Geometric Interpretation

### The Score Function Defined

The **score function** is the gradient of the log-density with respect to the state variable:

$$
s(x) = \nabla_x \log p(x)
$$

For a $d$-dimensional distribution, the score is a vector field $s: \mathbb{R}^d \to \mathbb{R}^d$ that assigns a direction to every point in space.

!!! info "Physical Intuition"
    Think of $\log p(x)$ as a landscape where elevation corresponds to log-probability. The score function $s(x)$ is the **gradient** of this landscape—it points "uphill" toward regions of higher probability density.

### Geometric Properties

The score function has several important geometric properties:

**Direction**: At any point $x$, the score points toward the direction of steepest increase in $\log p(x)$, which is the direction toward higher probability density.

**Magnitude**: The magnitude $\|s(x)\|$ indicates how rapidly the log-density changes. Large magnitudes indicate steep regions; small magnitudes indicate flat regions.

**Zeros**: The score vanishes at critical points of $\log p(x)$:

$$
s(x^*) = 0 \quad \Leftrightarrow \quad x^* \text{ is a mode or saddle point}
$$

At modes (local maxima of $p$), the score is zero because you're at the peak of the probability landscape.

### Relation to the Energy Function

In statistical mechanics and machine learning, we often work with distributions of the Boltzmann form:

$$
p(x) = \frac{1}{Z} \exp(-U(x))
$$

where $U(x)$ is the **energy function** (or potential) and $Z = \int \exp(-U(x)) dx$ is the normalization constant (partition function).

The score function has a simple relation to the energy:

$$
s(x) = \nabla_x \log p(x) = \nabla_x \left[ -U(x) - \log Z \right] = -\nabla U(x)
$$

The score is the **negative gradient of the energy**. It points from high-energy (low probability) regions toward low-energy (high probability) regions.

## Independence from Normalization

### The Ratio Trick for Scores

A crucial property of the score function is that it doesn't depend on the normalization constant:

$$
s(x) = \nabla_x \log p(x) = \nabla_x \log \frac{\tilde{p}(x)}{Z} = \nabla_x \log \tilde{p}(x) - \nabla_x \log Z = \nabla_x \log \tilde{p}(x)
$$

Since $Z$ is a constant (doesn't depend on $x$), its gradient vanishes: $\nabla_x \log Z = 0$.

!!! success "Computational Implication"
    We can compute the score from the **unnormalized density** $\tilde{p}(x)$ alone:
    
    $$
    s(x) = \nabla_x \log \tilde{p}(x)
    $$
    
    This is why gradient-based MCMC methods inherit the "ratio trick" from Metropolis-Hastings—they work with unnormalized densities.

### Why This Matters for Bayesian Inference

In Bayesian inference, the posterior is:

$$
p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}
$$

The score of the posterior is:

$$
s(\theta) = \nabla_\theta \log p(\theta | \mathcal{D}) = \nabla_\theta \left[ \log p(\mathcal{D} | \theta) + \log p(\theta) \right]
$$

The intractable marginal likelihood $p(\mathcal{D})$ disappears from the gradient. We only need:

- **Gradient of log-likelihood**: $\nabla_\theta \log p(\mathcal{D} | \theta)$ — computable
- **Gradient of log-prior**: $\nabla_\theta \log p(\theta)$ — computable

## Score Functions for Common Distributions

### Univariate Gaussian

For $p(x) = \mathcal{N}(\mu, \sigma^2)$:

$$
\log p(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
$$

$$
s(x) = \frac{d}{dx} \log p(x) = -\frac{x - \mu}{\sigma^2}
$$

**Interpretation**: The score points toward the mean $\mu$, with strength inversely proportional to the variance. Far from the mean, the score has large magnitude; at the mean, $s(\mu) = 0$.

### Multivariate Gaussian

For $p(x) = \mathcal{N}(\mu, \Sigma)$ where $x, \mu \in \mathbb{R}^d$:

$$
\log p(x) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)
$$

$$
s(x) = -\Sigma^{-1}(x - \mu)
$$

**Interpretation**: The score points toward the mean, with the **precision matrix** $\Sigma^{-1}$ determining how the strength varies with direction. In directions of high variance (small eigenvalues of $\Sigma^{-1}$), the score is weaker.

### Gaussian Mixture Model

For a mixture $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$:

$$
s(x) = \frac{\sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) \cdot \left[ -\Sigma_k^{-1}(x - \mu_k) \right]}{\sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)}
$$

This is a **responsibility-weighted average** of component scores:

$$
s(x) = \sum_{k=1}^K r_k(x) \cdot s_k(x)
$$

where $r_k(x) = \frac{\pi_k \mathcal{N}(x | \mu_k, \Sigma_k)}{p(x)}$ is the responsibility of component $k$.

**Interpretation**: Near a particular mode, that component's score dominates. Between modes, the score is a weighted combination pointing toward the nearest or strongest mode.

### Exponential Distribution

For $p(x) = \lambda e^{-\lambda x}$ with $x \geq 0$:

$$
s(x) = -\lambda
$$

The score is constant everywhere—the log-density is linear.

### Laplace Distribution

For $p(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)$:

$$
s(x) = -\frac{\text{sign}(x - \mu)}{b}
$$

The score has constant magnitude but discontinuous sign at $\mu$.

### Log-Normal Distribution

For $\log X \sim \mathcal{N}(\mu, \sigma^2)$:

$$
s(x) = -\frac{\log x - \mu}{\sigma^2 x} - \frac{1}{x}
$$

## PyTorch Implementation

### Computing Scores with Autograd

PyTorch's automatic differentiation makes score computation straightforward:

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

def compute_score(log_prob_fn, x):
    """
    Compute the score function using automatic differentiation.
    
    Args:
        log_prob_fn: Function that computes log p(x)
        x: Point(s) at which to compute the score
        
    Returns:
        score: Gradient of log p(x) with respect to x
    """
    x = x.clone().requires_grad_(True)
    log_prob = log_prob_fn(x)
    
    # Handle batched inputs
    if log_prob.dim() == 0:
        log_prob.backward()
    else:
        log_prob.sum().backward()
    
    return x.grad.clone()


class ScoreFunction:
    """
    Score function computation for various distributions.
    """
    
    @staticmethod
    def gaussian_1d(x, mu=0.0, sigma=1.0):
        """
        Analytical score for 1D Gaussian.
        
        s(x) = -(x - mu) / sigma^2
        """
        return -(x - mu) / (sigma ** 2)
    
    @staticmethod
    def gaussian_nd(x, mu, precision):
        """
        Analytical score for multivariate Gaussian.
        
        Args:
            x: Points of shape (n, d) or (d,)
            mu: Mean of shape (d,)
            precision: Precision matrix (inverse covariance) of shape (d, d)
            
        Returns:
            score: Score vectors of same shape as x
        """
        diff = x - mu
        if diff.dim() == 1:
            return -precision @ diff
        else:
            return -diff @ precision.T
    
    @staticmethod
    def mixture_gaussian_1d(x, weights, means, stds):
        """
        Score for 1D Gaussian mixture.
        
        Args:
            x: Points of shape (n,) or scalar
            weights: Mixture weights (K,)
            means: Component means (K,)
            stds: Component standard deviations (K,)
            
        Returns:
            score: Score values
        """
        x = torch.atleast_1d(x)
        K = len(weights)
        
        # Compute component densities: (n, K)
        x_expanded = x.unsqueeze(-1)  # (n, 1)
        means_expanded = means.unsqueeze(0)  # (1, K)
        stds_expanded = stds.unsqueeze(0)  # (1, K)
        
        # Log densities of each component
        log_densities = dist.Normal(means_expanded, stds_expanded).log_prob(x_expanded)
        log_weighted = log_densities + torch.log(weights)
        
        # Responsibilities via softmax
        responsibilities = torch.softmax(log_weighted, dim=-1)  # (n, K)
        
        # Component scores: -(x - mu_k) / sigma_k^2
        component_scores = -(x_expanded - means_expanded) / (stds_expanded ** 2)  # (n, K)
        
        # Weighted average
        score = (responsibilities * component_scores).sum(dim=-1)
        
        return score
```

### Visualizing the Score Field

```python
def visualize_score_field_1d():
    """
    Visualize score function for 1D distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = torch.linspace(-4, 4, 200)
    
    # 1. Standard Gaussian
    ax = axes[0, 0]
    log_p = dist.Normal(0, 1).log_prob(x)
    score = ScoreFunction.gaussian_1d(x, mu=0, sigma=1)
    
    ax.plot(x.numpy(), torch.exp(log_p).numpy(), 'b-', linewidth=2, label='p(x)')
    ax.plot(x.numpy(), score.numpy(), 'r-', linewidth=2, label='s(x) = ∇log p(x)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('Standard Gaussian\ns(x) = -x', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 1)
    
    # 2. Gaussian with different variance
    ax = axes[0, 1]
    sigma = 2.0
    log_p = dist.Normal(0, sigma).log_prob(x)
    score = ScoreFunction.gaussian_1d(x, mu=0, sigma=sigma)
    
    ax.plot(x.numpy(), torch.exp(log_p).numpy(), 'b-', linewidth=2, label='p(x)')
    ax.plot(x.numpy(), score.numpy(), 'r-', linewidth=2, label='s(x)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title(f'Gaussian with σ={sigma}\ns(x) = -x/{sigma}² (weaker gradient)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 0.5)
    
    # 3. Bimodal mixture
    ax = axes[1, 0]
    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([-2.0, 2.0])
    stds = torch.tensor([0.7, 0.7])
    
    score = ScoreFunction.mixture_gaussian_1d(x, weights, means, stds)
    
    # Compute mixture density
    mixture = dist.MixtureSameFamily(
        dist.Categorical(weights),
        dist.Normal(means, stds)
    )
    p = torch.exp(mixture.log_prob(x))
    
    ax.plot(x.numpy(), p.numpy(), 'b-', linewidth=2, label='p(x)')
    ax.plot(x.numpy(), score.numpy(), 'r-', linewidth=2, label='s(x)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('Bimodal Mixture\nScore points toward nearest mode', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Asymmetric mixture
    ax = axes[1, 1]
    weights = torch.tensor([0.7, 0.3])
    means = torch.tensor([-1.5, 2.5])
    stds = torch.tensor([0.5, 1.0])
    
    score = ScoreFunction.mixture_gaussian_1d(x, weights, means, stds)
    
    mixture = dist.MixtureSameFamily(
        dist.Categorical(weights),
        dist.Normal(means, stds)
    )
    p = torch.exp(mixture.log_prob(x))
    
    ax.plot(x.numpy(), p.numpy(), 'b-', linewidth=2, label='p(x)')
    ax.plot(x.numpy(), score.numpy(), 'r-', linewidth=2, label='s(x)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('Asymmetric Mixture\nDominant mode has stronger pull', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('score_function_1d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: score_function_1d.png")


def visualize_score_field_2d():
    """
    Visualize 2D score vector field.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create grid
    x = torch.linspace(-3, 3, 20)
    y = torch.linspace(-3, 3, 20)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    # 1. Correlated Gaussian
    ax = axes[0]
    mu = torch.zeros(2)
    cov = torch.tensor([[1.0, 0.7], [0.7, 1.0]])
    precision = torch.linalg.inv(cov)
    
    scores = ScoreFunction.gaussian_nd(points, mu, precision)
    
    # Compute density for contours
    x_dense = torch.linspace(-3, 3, 100)
    y_dense = torch.linspace(-3, 3, 100)
    X_dense, Y_dense = torch.meshgrid(x_dense, y_dense, indexing='ij')
    points_dense = torch.stack([X_dense.flatten(), Y_dense.flatten()], dim=-1)
    
    mvn = dist.MultivariateNormal(mu, cov)
    Z = torch.exp(mvn.log_prob(points_dense)).reshape(100, 100)
    
    ax.contour(X_dense.numpy(), Y_dense.numpy(), Z.numpy(), levels=10, alpha=0.5, cmap='Blues')
    ax.quiver(X.numpy(), Y.numpy(), 
              scores[:, 0].reshape(20, 20).numpy(),
              scores[:, 1].reshape(20, 20).numpy(),
              color='red', alpha=0.7)
    ax.plot(0, 0, 'k*', markersize=15, label='Mode')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Correlated Gaussian\nScore field points toward mode', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Banana-shaped distribution
    ax = axes[1]
    
    def banana_log_prob(xy):
        """Log density of banana-shaped distribution."""
        x, y = xy[..., 0], xy[..., 1]
        return -0.5 * (x**2 + (y - x**2)**2)
    
    def banana_score(xy):
        """Analytical score for banana distribution."""
        x, y = xy[..., 0], xy[..., 1]
        s_x = -x - 2*x*(x**2 - y)
        s_y = -(y - x**2)
        return torch.stack([s_x, s_y], dim=-1)
    
    scores_banana = banana_score(points)
    
    # Compute density for contours
    Z_banana = torch.exp(banana_log_prob(points_dense)).reshape(100, 100)
    
    ax.contour(X_dense.numpy(), Y_dense.numpy(), Z_banana.numpy(), 
               levels=10, alpha=0.5, cmap='Greens')
    ax.quiver(X.numpy(), Y.numpy(),
              scores_banana[:, 0].reshape(20, 20).numpy(),
              scores_banana[:, 1].reshape(20, 20).numpy(),
              color='red', alpha=0.7)
    ax.plot(0, 0, 'k*', markersize=15, label='Mode')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Banana Distribution\nScore follows curved geometry', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('score_function_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: score_function_2d.png")


if __name__ == "__main__":
    visualize_score_field_1d()
    visualize_score_field_2d()
```

## Score Functions in Deep Learning

### Score Matching

In many applications, we don't know the target distribution $p(x)$ analytically—we only have samples. **Score matching** provides a way to learn the score function from data.

The naive objective would be to minimize:

$$
\mathcal{L}(\theta) = \mathbb{E}_{p(x)}\left[\frac{1}{2}\|s_\theta(x) - \nabla_x \log p(x)\|^2\right]
$$

But this requires knowing $\nabla_x \log p(x)$, which we don't have!

**Score matching theorem** (Hyvärinen, 2005): Under mild conditions, the objective above is equivalent (up to a constant) to:

$$
\mathcal{L}(\theta) = \mathbb{E}_{p(x)}\left[\frac{1}{2}\|s_\theta(x)\|^2 + \text{tr}(\nabla_x s_\theta(x))\right]
$$

This only requires samples from $p(x)$ and derivatives of the model $s_\theta$.

### Denoising Score Matching

An even simpler approach: add noise to data and learn to "denoise":

$$
\tilde{x} = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

The score of the noised conditional distribution $q(\tilde{x} | x) = \mathcal{N}(\tilde{x} | x, \sigma^2 I)$ is:

$$
\nabla_{\tilde{x}} \log q(\tilde{x} | x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

**Denoising score matching objective**:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x, \epsilon}\left[\left\|s_\theta(\tilde{x}, \sigma) + \frac{\epsilon}{\sigma}\right\|^2\right]
$$

This is equivalent to training a network to predict the noise—the foundation of diffusion models!

### Connection to Diffusion Models

The score function is the central object in score-based generative models:

| Classical MCMC | Diffusion Models |
|----------------|------------------|
| Known target $p(x)$ | Unknown data distribution $p_\text{data}(x)$ |
| Analytical score $\nabla_x \log p(x)$ | Learned score $s_\theta(x)$ |
| Sample via Langevin dynamics | Generate via reverse diffusion |
| Fixed score | Time-varying score $s_\theta(x, t)$ |

The deep insight: **diffusion models are Langevin MCMC with learned scores**.

## Practical Considerations

### Numerical Stability

When computing scores numerically, watch for:

**Overflow/Underflow**: Log-densities can become very negative in low-probability regions. Use log-sum-exp tricks for mixtures.

**Gradient Clipping**: In regions of very steep log-density, gradients can explode. Consider clipping:

```python
def clip_score(score, max_norm=10.0):
    """Clip score vectors to maximum norm."""
    norm = torch.norm(score, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    return score * scale
```

### Batch Computation

For efficiency, compute scores in batches:

```python
def batch_score(log_prob_fn, x_batch):
    """
    Compute scores for a batch of points.
    
    Args:
        log_prob_fn: Function mapping (batch,) -> (batch,) of log probs
        x_batch: Points of shape (batch, dim)
        
    Returns:
        scores: Gradients of shape (batch, dim)
    """
    x_batch = x_batch.clone().requires_grad_(True)
    log_probs = log_prob_fn(x_batch)
    
    # Compute gradient for each sample
    scores = torch.autograd.grad(
        outputs=log_probs.sum(),
        inputs=x_batch,
        create_graph=False
    )[0]
    
    return scores
```

## Summary

The score function $s(x) = \nabla_x \log p(x)$ is the fundamental quantity for gradient-based sampling:

| Property | Description |
|----------|-------------|
| Definition | Gradient of log-density |
| Direction | Points toward higher probability |
| Normalization | Independent of $Z$ |
| Zeros | Located at modes and saddle points |
| Computation | Via automatic differentiation |

**Key insight**: The score encodes the geometry of the probability distribution. Langevin dynamics and related methods use this geometry to guide samples toward the target distribution.

## Exercises

### Exercise 1: Score Computation

Derive the score function analytically for:

1. Student's t-distribution with $\nu$ degrees of freedom
2. Beta distribution $\text{Beta}(\alpha, \beta)$ on $[0, 1]$
3. Multivariate t-distribution

### Exercise 2: Score Visualization

Create a visualization of the score field for a mixture of three 2D Gaussians with different means and covariances. Show how the score field changes as you vary the mixture weights.

### Exercise 3: Score Matching Implementation

Implement the explicit score matching objective and verify it gives the same gradients as direct computation for a Gaussian target.

### Exercise 4: Numerical Gradients

Compare automatic differentiation scores with numerical finite-difference approximations:

$$
s(x) \approx \frac{\log p(x + h) - \log p(x - h)}{2h}
$$

For what values of $h$ do you get accurate results?

## References

1. Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *Journal of Machine Learning Research*, 6, 695-709.

2. Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7), 1661-1674.

3. Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*.

4. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.
