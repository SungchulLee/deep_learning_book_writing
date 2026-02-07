# Black-Box Variational Inference

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the motivation for gradient-based VI methods
2. Derive the score function gradient estimator (REINFORCE)
3. Implement variance reduction techniques for gradient estimation
4. Apply the reparameterization trick for continuous variables
5. Build black-box VI algorithms that work with any differentiable model

## Motivation: Beyond Conjugate Models

Traditional CAVI requires:

1. **Model-specific derivations**: Each new model needs custom update equations
2. **Conjugate priors**: Closed-form updates rely on conjugacy
3. **Analytical expectations**: Must be able to compute expected sufficient statistics

**Black-Box VI** removes these restrictions by using gradient-based optimization with Monte Carlo estimation, enabling VI for any differentiable probabilistic model.

## The Gradient Challenge

We want to maximize the ELBO:

$$
\text{ELBO}(q_\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D}, \theta) - \log q_\phi(\theta)]
$$

where $\phi$ are the variational parameters (e.g., mean and variance of a Gaussian $q$).

The challenge is computing the gradient:

$$
\nabla_\phi \text{ELBO}(q_\phi) = \nabla_\phi \mathbb{E}_{q_\phi(\theta)}[f(\theta)]
$$

where $f(\theta) = \log p(\mathcal{D}, \theta) - \log q_\phi(\theta)$.

**The problem**: The expectation is with respect to $q_\phi$, which depends on $\phi$!

## Score Function Gradient Estimator (REINFORCE)

### Derivation

Using the **log-derivative trick**:

$$
\nabla_\phi q_\phi(\theta) = q_\phi(\theta) \nabla_\phi \log q_\phi(\theta)
$$

We can rewrite the gradient:

$$
\begin{aligned}
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] &= \nabla_\phi \int q_\phi(\theta) f(\theta) \, d\theta \\
&= \int \nabla_\phi q_\phi(\theta) f(\theta) \, d\theta \\
&= \int q_\phi(\theta) \nabla_\phi \log q_\phi(\theta) f(\theta) \, d\theta \\
&= \mathbb{E}_{q_\phi}\left[f(\theta) \nabla_\phi \log q_\phi(\theta)\right]
\end{aligned}
$$

### Score Function Estimator

The **score function** is $s_\phi(\theta) = \nabla_\phi \log q_\phi(\theta)$.

The gradient estimator becomes:

$$
\boxed{\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_{s=1}^S f(\theta^{(s)}) \nabla_\phi \log q_\phi(\theta^{(s)})}
$$

where $\theta^{(s)} \sim q_\phi(\theta)$.

This is also known as the **REINFORCE** estimator (from reinforcement learning).

### Properties

**Advantages**:
- Works for **any** distribution $q_\phi$ (discrete or continuous)
- Only requires samples from $q_\phi$ and ability to evaluate $\nabla_\phi \log q_\phi$
- No model-specific derivations needed

**Disadvantages**:
- **High variance**: Can require many samples for accurate gradients
- **Slow convergence**: High variance leads to noisy optimization

## Variance Reduction Techniques

The score function estimator has high variance. Several techniques can help.

### Control Variates

A **control variate** is a function $c(\theta)$ with known expectation. We modify the estimator:

$$
\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_{s=1}^S \left(f(\theta^{(s)}) - c(\theta^{(s)})\right) \nabla_\phi \log q_\phi(\theta^{(s)}) + \mathbb{E}[c(\theta)] \mathbb{E}[\nabla_\phi \log q_\phi]
$$

Since $\mathbb{E}_{q_\phi}[\nabla_\phi \log q_\phi(\theta)] = 0$, we can use:

$$
\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_{s=1}^S \left(f(\theta^{(s)}) - c\right) \nabla_\phi \log q_\phi(\theta^{(s)})
$$

for any constant $c$ (the **baseline**).

### Optimal Baseline

The optimal constant baseline minimizes variance:

$$
c^* = \frac{\mathbb{E}[f(\theta) \|\nabla_\phi \log q_\phi(\theta)\|^2]}{\mathbb{E}[\|\nabla_\phi \log q_\phi(\theta)\|^2]}
$$

In practice, we estimate this from running averages.

### Rao-Blackwellization

When possible, analytically integrate out some variables to reduce variance:

$$
\mathbb{E}_{q(\theta_1, \theta_2)}[f(\theta_1, \theta_2)] = \mathbb{E}_{q(\theta_1)}[\mathbb{E}_{q(\theta_2|\theta_1)}[f(\theta_1, \theta_2)]]
$$

If the inner expectation can be computed analytically, variance is reduced.

## The Reparameterization Trick

For continuous distributions, the **reparameterization trick** provides a lower-variance alternative to the score function estimator.

### Key Idea

Instead of sampling $\theta \sim q_\phi(\theta)$ directly, we:

1. Sample from a **base distribution**: $\epsilon \sim p(\epsilon)$ (independent of $\phi$)
2. Apply a **deterministic transformation**: $\theta = g_\phi(\epsilon)$

This moves the stochasticity to $\epsilon$, making the gradient flow through $g_\phi$.

### For Gaussian $q$

If $q_\phi(\theta) = \mathcal{N}(\mu, \sigma^2)$ with $\phi = (\mu, \sigma)$:

$$
\theta = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

The gradient becomes:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]
$$

This can be estimated with:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] \approx \frac{1}{S} \sum_{s=1}^S \nabla_\phi f(\mu + \sigma \cdot \epsilon^{(s)})
$$

### General Reparameterization

For a distribution $q_\phi$ with CDF $F_\phi$:

$$
\theta = F_\phi^{-1}(u), \quad u \sim \text{Uniform}(0, 1)
$$

Common reparameterizations:

| Distribution | Base | Transformation |
|--------------|------|----------------|
| $\mathcal{N}(\mu, \sigma^2)$ | $\epsilon \sim \mathcal{N}(0,1)$ | $\theta = \mu + \sigma\epsilon$ |
| $\text{LogNormal}(\mu, \sigma^2)$ | $\epsilon \sim \mathcal{N}(0,1)$ | $\theta = \exp(\mu + \sigma\epsilon)$ |
| $\text{Gamma}(\alpha, \beta)$ | $\epsilon \sim \text{Gamma}(\alpha, 1)$ | $\theta = \epsilon / \beta$ |
| $\text{Beta}(\alpha, \beta)$ | $u \sim \text{Uniform}(0,1)$ | Inverse CDF (numerical) |

### Advantages of Reparameterization

- **Much lower variance** than score function estimator
- **Faster convergence** in optimization
- **Works with automatic differentiation** (just backprop through the transformation)

### Limitations

- Only works for **continuous** distributions
- Requires **differentiable** transformation
- Not all distributions have easy reparameterizations

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Callable, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class ScoreFunctionVI:
    """
    Black-Box VI using Score Function (REINFORCE) gradient estimator.
    
    Works for any distribution q_φ where we can:
    1. Sample θ ~ q_φ
    2. Compute ∇_φ log q_φ(θ)
    """
    
    def __init__(self, log_joint: Callable, dim: int):
        """
        Args:
            log_joint: Function that computes log p(D, θ)
            dim: Dimension of θ
        """
        self.log_joint = log_joint
        self.dim = dim
        
        # Variational parameters for Gaussian q(θ) = N(μ, diag(σ²))
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from q_φ(θ)."""
        q = dist.Normal(self.mu, self.sigma)
        return q.rsample((n_samples,))
    
    def log_q(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log q_φ(θ)."""
        q = dist.Normal(self.mu, self.sigma)
        return q.log_prob(theta).sum(dim=-1)
    
    def score_function_gradient(self, n_samples: int = 100,
                                baseline: float = 0.0) -> Tuple[torch.Tensor, float]:
        """
        Compute gradient using score function estimator with baseline.
        
        ∇_φ ELBO ≈ (1/S) Σ (f(θ⁽ˢ⁾) - b) ∇_φ log q_φ(θ⁽ˢ⁾)
        
        where f(θ) = log p(D,θ) - log q_φ(θ)
        """
        # Sample from q
        theta = self.sample(n_samples)  # [S, dim]
        
        # Compute f(θ) = log p(D,θ) - log q(θ)
        log_p = self.log_joint(theta)           # [S]
        log_q = self.log_q(theta)               # [S]
        f = log_p - log_q                       # [S]
        
        # ELBO estimate
        elbo = f.mean().item()
        
        # Score function gradient
        # Need to compute ∇_φ log q_φ(θ) for each sample
        # For Gaussian: ∇_μ log q = (θ - μ)/σ², ∇_log_σ log q = ((θ-μ)²/σ² - 1)
        
        # Zero gradients
        if self.mu.grad is not None:
            self.mu.grad.zero_()
        if self.log_sigma.grad is not None:
            self.log_sigma.grad.zero_()
        
        # Compute gradients sample by sample (inefficient but clear)
        grad_mu = torch.zeros_like(self.mu)
        grad_log_sigma = torch.zeros_like(self.log_sigma)
        
        for s in range(n_samples):
            # Score function components
            score_mu = (theta[s] - self.mu) / self.sigma**2
            score_log_sigma = ((theta[s] - self.mu)**2 / self.sigma**2 - 1)
            
            # Accumulate weighted by (f - baseline)
            weight = f[s].item() - baseline
            grad_mu += weight * score_mu
            grad_log_sigma += weight * score_log_sigma
        
        grad_mu /= n_samples
        grad_log_sigma /= n_samples
        
        return (grad_mu, grad_log_sigma), elbo
    
    def fit(self, n_iterations: int = 1000, n_samples: int = 100,
            lr: float = 0.01, use_baseline: bool = True,
            verbose: bool = True) -> Dict:
        """
        Optimize using score function gradients.
        """
        optimizer = torch.optim.Adam([self.mu, self.log_sigma], lr=lr)
        
        history = {'elbo': [], 'mu': [], 'sigma': []}
        baseline = 0.0
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # Compute gradient
            (grad_mu, grad_log_sigma), elbo = self.score_function_gradient(
                n_samples, baseline if use_baseline else 0.0
            )
            
            # Update baseline (running average of f)
            if use_baseline:
                baseline = 0.9 * baseline + 0.1 * elbo
            
            # Set gradients (negative because optimizer minimizes)
            self.mu.grad = -grad_mu
            self.log_sigma.grad = -grad_log_sigma
            
            optimizer.step()
            
            history['elbo'].append(elbo)
            history['mu'].append(self.mu.clone().detach())
            history['sigma'].append(self.sigma.clone().detach())
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iter {i+1:4d}: ELBO = {elbo:.4f}, "
                      f"μ = {self.mu.detach().numpy()}, "
                      f"σ = {self.sigma.detach().numpy()}")
        
        return history


class ReparameterizedVI:
    """
    Black-Box VI using Reparameterization Trick.
    
    Much lower variance than score function estimator for continuous q.
    """
    
    def __init__(self, log_joint: Callable, dim: int):
        """
        Args:
            log_joint: Function that computes log p(D, θ)
            dim: Dimension of θ
        """
        self.log_joint = log_joint
        self.dim = dim
        
        # Variational parameters
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def elbo(self, n_samples: int = 100) -> torch.Tensor:
        """
        Compute ELBO using reparameterization.
        
        θ = μ + σ ⊙ ε,  ε ~ N(0, I)
        
        ELBO = E_ε[log p(D, μ + σε) - log q(μ + σε)]
        """
        # Sample from base distribution
        epsilon = torch.randn(n_samples, self.dim)
        
        # Reparameterize
        theta = self.mu + self.sigma * epsilon
        
        # Compute log joint
        log_p = self.log_joint(theta)
        
        # Compute log q (analytical for Gaussian)
        log_q = dist.Normal(self.mu, self.sigma).log_prob(theta).sum(dim=-1)
        
        # ELBO
        return (log_p - log_q).mean()
    
    def fit(self, n_iterations: int = 1000, n_samples: int = 10,
            lr: float = 0.01, verbose: bool = True) -> Dict:
        """
        Optimize using reparameterization gradients.
        
        Note: Needs far fewer samples than score function!
        """
        optimizer = torch.optim.Adam([self.mu, self.log_sigma], lr=lr)
        
        history = {'elbo': [], 'mu': [], 'sigma': []}
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # Compute ELBO (gradients flow through reparameterization)
            elbo = self.elbo(n_samples)
            
            # Gradient ascent (minimize negative ELBO)
            loss = -elbo
            loss.backward()
            optimizer.step()
            
            history['elbo'].append(elbo.item())
            history['mu'].append(self.mu.clone().detach())
            history['sigma'].append(self.sigma.clone().detach())
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iter {i+1:4d}: ELBO = {elbo.item():.4f}, "
                      f"μ = {self.mu.detach().numpy()}, "
                      f"σ = {self.sigma.detach().numpy()}")
        
        return history


def compare_estimators():
    """Compare score function vs reparameterization estimators."""
    
    # Simple target: posterior for Gaussian mean
    # Prior: θ ~ N(0, 1)
    # Likelihood: x_i ~ N(θ, 1), with observed mean = 2.0
    
    n_data = 50
    data_mean = 2.0
    data_var = 1.0
    
    # Exact posterior: N(μ_n, σ_n²)
    posterior_precision = 1 + n_data / data_var
    exact_mean = n_data * data_mean / data_var / posterior_precision
    exact_std = 1 / np.sqrt(posterior_precision)
    
    print(f"Exact posterior: N({exact_mean:.4f}, {exact_std:.4f}²)")
    
    def log_joint(theta):
        """Log p(D, θ) = log p(D|θ) + log p(θ)"""
        # Prior
        log_prior = dist.Normal(0, 1).log_prob(theta).sum(dim=-1)
        
        # Likelihood (using sufficient statistics)
        log_lik = -0.5 * n_data * ((theta.squeeze() - data_mean)**2 + data_var)
        
        return log_prior + log_lik
    
    # Compare methods
    results = {}
    
    # Score function (needs many samples)
    print("\n--- Score Function Estimator ---")
    vi_score = ScoreFunctionVI(log_joint, dim=1)
    results['score'] = vi_score.fit(n_iterations=500, n_samples=100, 
                                    lr=0.01, verbose=True)
    
    # Reparameterization (needs few samples)
    print("\n--- Reparameterization Trick ---")
    vi_reparam = ReparameterizedVI(log_joint, dim=1)
    results['reparam'] = vi_reparam.fit(n_iterations=500, n_samples=10,
                                        lr=0.01, verbose=True)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ELBO comparison
    ax = axes[0, 0]
    ax.plot(results['score']['elbo'], alpha=0.7, label='Score Function')
    ax.plot(results['reparam']['elbo'], alpha=0.7, label='Reparameterization')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean convergence
    ax = axes[0, 1]
    mu_score = torch.stack(results['score']['mu']).squeeze().numpy()
    mu_reparam = torch.stack(results['reparam']['mu']).squeeze().numpy()
    ax.plot(mu_score, alpha=0.7, label='Score Function')
    ax.plot(mu_reparam, alpha=0.7, label='Reparameterization')
    ax.axhline(exact_mean, color='red', linestyle='--', label='Exact')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(b) Mean Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Variance in ELBO estimates
    ax = axes[1, 0]
    window = 50
    var_score = np.array([np.var(results['score']['elbo'][max(0,i-window):i+1]) 
                          for i in range(len(results['score']['elbo']))])
    var_reparam = np.array([np.var(results['reparam']['elbo'][max(0,i-window):i+1])
                            for i in range(len(results['reparam']['elbo']))])
    ax.plot(var_score, alpha=0.7, label='Score Function')
    ax.plot(var_reparam, alpha=0.7, label='Reparameterization')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(f'ELBO Variance (window={window})', fontsize=11)
    ax.set_title('(c) Gradient Variance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Final posteriors
    ax = axes[1, 1]
    theta_range = torch.linspace(exact_mean - 3*exact_std, 
                                  exact_mean + 3*exact_std, 200)
    
    # Exact
    exact_pdf = dist.Normal(exact_mean, exact_std).log_prob(theta_range).exp()
    ax.plot(theta_range.numpy(), exact_pdf.numpy(), 'k-', linewidth=2.5, 
            label='Exact')
    
    # Score function result
    mu_s = results['score']['mu'][-1].item()
    sigma_s = results['score']['sigma'][-1].item()
    score_pdf = dist.Normal(mu_s, sigma_s).log_prob(theta_range).exp()
    ax.plot(theta_range.numpy(), score_pdf.numpy(), '--', linewidth=2,
            label=f'Score (μ={mu_s:.3f}, σ={sigma_s:.3f})')
    
    # Reparam result
    mu_r = results['reparam']['mu'][-1].item()
    sigma_r = results['reparam']['sigma'][-1].item()
    reparam_pdf = dist.Normal(mu_r, sigma_r).log_prob(theta_range).exp()
    ax.plot(theta_range.numpy(), reparam_pdf.numpy(), ':', linewidth=2,
            label=f'Reparam (μ={mu_r:.3f}, σ={sigma_r:.3f})')
    
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Final Posteriors', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bbvi_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    compare_estimators()
```

## Automatic Differentiation VI (ADVI)

ADVI is a practical implementation of black-box VI that:

1. Transforms constrained parameters to unconstrained space
2. Uses diagonal Gaussian variational family
3. Applies reparameterization trick
4. Works automatically with any differentiable model

### Parameter Transformations

| Constraint | Transformation | Inverse |
|------------|---------------|---------|
| $\theta > 0$ | $\zeta = \log \theta$ | $\theta = \exp(\zeta)$ |
| $0 < \theta < 1$ | $\zeta = \text{logit}(\theta)$ | $\theta = \text{sigmoid}(\zeta)$ |
| $\theta \in \mathbb{R}^d$, $\|\theta\| = 1$ | Spherical coords | Unit vector |
| Simplex | Stick-breaking | Dirichlet |

### Jacobian Correction

When transforming $\theta = h(\zeta)$:

$$
\log p(\theta) = \log p(h(\zeta)) + \log \left|\det \frac{\partial h}{\partial \zeta}\right|
$$

## Summary

**Black-Box VI** enables variational inference for any differentiable model:

**Score Function Estimator**:
$$
\nabla_\phi \text{ELBO} \approx \frac{1}{S} \sum_s f(\theta^{(s)}) \nabla_\phi \log q_\phi(\theta^{(s)})
$$
- Works for any $q$ (discrete or continuous)
- High variance, needs many samples
- Use control variates/baselines to reduce variance

**Reparameterization Trick**:
$$
\theta = g_\phi(\epsilon), \quad \epsilon \sim p(\epsilon)
$$
- Much lower variance
- Only for continuous, reparameterizable distributions
- Enables automatic differentiation

## Exercises

### Exercise 1: Variance Comparison

Empirically compare the variance of score function vs reparameterization gradient estimates for different numbers of samples.

### Exercise 2: Control Variate Design

Implement and compare different control variate strategies for the score function estimator.

### Exercise 3: Non-Gaussian Variational Family

Implement black-box VI with a mixture of Gaussians variational family.

## References

1. Ranganath, R., Gerrish, S., & Blei, D. M. (2014). "Black Box Variational Inference."

2. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

3. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models."

4. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). "Automatic Differentiation Variational Inference."

5. Mohamed, S., Rosca, M., Figurnov, M., & Mnih, A. (2020). "Monte Carlo Gradient Estimation in Machine Learning."
