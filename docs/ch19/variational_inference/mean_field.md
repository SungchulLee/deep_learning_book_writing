# Mean-Field Variational Inference

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the mean-field assumption and its implications
2. Derive the optimal form of each variational factor
3. Recognize when mean-field approximation is appropriate
4. Quantify the limitations of ignoring posterior correlations
5. Implement mean-field VI for multivariate models

## The Mean-Field Assumption

The **mean-field approximation** is the most common simplifying assumption in variational inference. It assumes that the variational distribution fully factorizes across all latent variables.

### Definition

For a model with latent variables $\theta = (\theta_1, \theta_2, \ldots, \theta_K)$, the mean-field variational family is:

$$
\mathcal{Q}_{\text{MF}} = \left\{ q(\theta) : q(\theta) = \prod_{j=1}^K q_j(\theta_j) \right\}
$$

Each parameter $\theta_j$ has its own independent variational factor $q_j(\theta_j)$.

### Etymology: Why "Mean-Field"?

The term comes from statistical physics, where it describes approximations to complex many-body systems:

- Each particle interacts with the **average (mean) field** created by all other particles
- Individual fluctuations and correlations are ignored
- The approximation replaces complex interactions with simplified averages

In VI, each variable $\theta_j$ depends on the **expected values** of other variables, not their full distributions.

## Mathematical Consequences

### Independence Assumption

The mean-field assumption enforces:

$$
q(\theta_1, \theta_2, \ldots, \theta_K) = q_1(\theta_1) \times q_2(\theta_2) \times \cdots \times q_K(\theta_K)
$$

This implies:

$$
\text{Cov}_q(\theta_i, \theta_j) = 0 \quad \text{for all } i \neq j
$$

**Even if the true posterior has correlations, mean-field cannot capture them.**

### Visual Illustration

```
True Posterior p(θ₁,θ₂|D):            Mean-Field q(θ₁)q(θ₂):
┌─────────────────────┐              ┌─────────────────────┐
│         ╱           │              │                     │
│        ╱            │              │    ┌───────────┐    │
│       ╱   ╭──╮      │              │    │           │    │
│      ╱   │  │       │     →→→      │    │     ○     │    │
│     ╱    ╰──╯       │              │    │           │    │
│    ╱                │              │    └───────────┘    │
└─────────────────────┘              └─────────────────────┘
   Correlated ellipse                 Axis-aligned ellipse
   (tilted, captures                  (cannot capture
    covariance)                        covariance)
```

### Quantifying the Approximation Error

For a bivariate Gaussian with correlation $\rho$:

$$
p(\theta_1, \theta_2 | \mathcal{D}) = \mathcal{N}\left(\begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}\right)
$$

The best mean-field approximation is:

$$
q(\theta_1)q(\theta_2) = \mathcal{N}(\mu_1, \sigma_1^2) \times \mathcal{N}(\mu_2, \sigma_2^2)
$$

The KL divergence between them is:

$$
\text{KL}(q \| p) = -\frac{1}{2}\log(1 - \rho^2)
$$

This shows that **high correlation leads to poor approximation**:

| Correlation $\rho$ | KL Divergence |
|-------------------|---------------|
| 0.0 | 0.000 |
| 0.5 | 0.144 |
| 0.8 | 0.511 |
| 0.9 | 0.831 |
| 0.95 | 1.151 |
| 0.99 | 2.296 |

## Optimal Mean-Field Updates

### The CAVI Update Formula

For mean-field VI, there is a beautiful closed-form expression for the optimal factor $q_j^*(\theta_j)$:

$$
\boxed{q_j^*(\theta_j) \propto \exp\left\{ \mathbb{E}_{q_{-j}}[\log p(\theta, \mathcal{D})] \right\}}
$$

where $q_{-j} = \prod_{i \neq j} q_i(\theta_i)$ denotes all factors except $q_j$.

### Derivation

We want to maximize ELBO with respect to $q_j$ while holding all other factors fixed:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \sum_{k=1}^K \mathbb{E}_{q_k}[\log q_k(\theta_k)]
$$

Taking the functional derivative with respect to $q_j$ and setting to zero:

$$
\frac{\delta \text{ELBO}}{\delta q_j} = \mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)] - \log q_j(\theta_j) - 1 = 0
$$

Solving for $q_j$:

$$
\log q_j^*(\theta_j) = \mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)] + \text{const}
$$

Exponentiating and normalizing:

$$
q_j^*(\theta_j) = \frac{\exp\{\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\}}{\int \exp\{\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\} d\theta_j}
$$

### Key Insight

The optimal $q_j^*(\theta_j)$ depends on the **expected sufficient statistics** of all other variables under their current variational distributions. This creates an iterative dependency that we solve with coordinate ascent.

## Coordinate Ascent Variational Inference (CAVI)

### Algorithm

CAVI iteratively updates each variational factor while holding others fixed:

```
Algorithm: Coordinate Ascent Variational Inference (CAVI)
─────────────────────────────────────────────────────────
Input: Model p(D, θ), initial q₁⁽⁰⁾, ..., qₖ⁽⁰⁾
Output: Optimized variational factors q₁*, ..., qₖ*

1. Initialize variational factors q₁⁽⁰⁾, ..., qₖ⁽⁰⁾
2. Compute initial ELBO
3. Repeat until convergence:
   a. For j = 1, ..., K:
      i.  Compute expected sufficient statistics from q₋ⱼ
      ii. Update: qⱼ(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}
   b. Compute ELBO
   c. Check convergence (ELBO change < tolerance)
4. Return optimized factors
```

### Convergence Properties

**Theorem (CAVI Monotonicity)**: Each update of $q_j$ cannot decrease the ELBO.

*Proof sketch*: Each coordinate update finds the global optimum for that factor given all others. Since we're maximizing a concave functional over each factor, the ELBO can only increase or stay the same.

**Corollary**: CAVI converges to a local optimum of the ELBO.

## Example: Gaussian with Unknown Mean and Variance

### Model Specification

Consider the conjugate Normal-Gamma model:

$$
\begin{aligned}
\text{Prior on precision: } & \tau \sim \text{Gamma}(\alpha_0, \beta_0) \\
\text{Prior on mean: } & \mu | \tau \sim \mathcal{N}(\mu_0, (\lambda_0 \tau)^{-1}) \\
\text{Likelihood: } & x_i | \mu, \tau \sim \mathcal{N}(\mu, \tau^{-1})
\end{aligned}
$$

### Mean-Field Factorization

We assume:

$$
q(\mu, \tau) = q_\mu(\mu) \cdot q_\tau(\tau)
$$

### Deriving the Optimal Factors

**Joint log probability:**

$$
\log p(\mathbf{x}, \mu, \tau) = \log p(\tau) + \log p(\mu|\tau) + \sum_{i=1}^n \log p(x_i|\mu,\tau)
$$

**Optimal $q_\mu(\mu)$:**

Taking expectation over $q_\tau$:

$$
\log q_\mu^*(\mu) = \mathbb{E}_{q_\tau}[\log p(\mathbf{x}, \mu, \tau)] + \text{const}
$$

Collecting terms involving $\mu$:

$$
\log q_\mu^*(\mu) \propto -\frac{\mathbb{E}[\tau]}{2}\left[\lambda_0(\mu - \mu_0)^2 + \sum_{i=1}^n(x_i - \mu)^2\right]
$$

This is quadratic in $\mu$, so:

$$
q_\mu^*(\mu) = \mathcal{N}(\mu_n, \lambda_n^{-1})
$$

where:

$$
\begin{aligned}
\lambda_n &= (\lambda_0 + n)\mathbb{E}_{q_\tau}[\tau] \\
\mu_n &= \frac{\lambda_0 \mu_0 + n\bar{x}}{\lambda_0 + n}
\end{aligned}
$$

**Optimal $q_\tau(\tau)$:**

Similarly:

$$
q_\tau^*(\tau) = \text{Gamma}(\alpha_n, \beta_n)
$$

where:

$$
\begin{aligned}
\alpha_n &= \alpha_0 + \frac{n+1}{2} \\
\beta_n &= \beta_0 + \frac{1}{2}\left[\lambda_0(\mathbb{E}[\mu] - \mu_0)^2 + \sum_{i=1}^n(x_i - \mathbb{E}[\mu])^2 + n\text{Var}_{q_\mu}[\mu]\right]
\end{aligned}
$$

## PyTorch Implementation

```python
import torch
import torch.distributions as dist
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class MeanFieldGaussianCAVI:
    """
    Coordinate Ascent VI for Gaussian with unknown mean and precision.
    
    Model:
        τ ~ Gamma(α₀, β₀)           [precision prior]
        μ | τ ~ N(μ₀, (λ₀τ)⁻¹)      [conditional mean prior]
        xᵢ | μ,τ ~ N(μ, τ⁻¹)        [likelihood]
    
    Variational family:
        q(μ,τ) = q(μ)q(τ)
        q(μ) = N(μₙ, λₙ⁻¹)
        q(τ) = Gamma(αₙ, βₙ)
    """
    
    def __init__(self, alpha_0: float = 1.0, beta_0: float = 1.0,
                 mu_0: float = 0.0, lambda_0: float = 1.0):
        """
        Initialize with prior hyperparameters.
        """
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        
        # Variational parameters (initialized to prior)
        self.alpha_n = alpha_0
        self.beta_n = beta_0
        self.mu_n = mu_0
        self.lambda_n = lambda_0
    
    @property
    def E_tau(self) -> float:
        """Expected precision E_q[τ]"""
        return self.alpha_n / self.beta_n
    
    @property
    def E_log_tau(self) -> float:
        """Expected log precision E_q[log τ]"""
        return torch.digamma(torch.tensor(self.alpha_n)).item() - np.log(self.beta_n)
    
    @property
    def E_mu(self) -> float:
        """Expected mean E_q[μ]"""
        return self.mu_n
    
    @property
    def Var_mu(self) -> float:
        """Variance of mean Var_q[μ]"""
        return 1.0 / self.lambda_n
    
    def update_q_mu(self, data: torch.Tensor) -> None:
        """
        Update variational parameters for q(μ).
        
        q*(μ) ∝ exp{E_q(τ)[log p(x,μ,τ)]}
        """
        n = len(data)
        x_bar = data.mean().item()
        
        # Precision of variational Gaussian
        self.lambda_n = (self.lambda_0 + n) * self.E_tau
        
        # Mean of variational Gaussian
        self.mu_n = (self.lambda_0 * self.mu_0 + n * x_bar) / (self.lambda_0 + n)
    
    def update_q_tau(self, data: torch.Tensor) -> None:
        """
        Update variational parameters for q(τ).
        
        q*(τ) ∝ exp{E_q(μ)[log p(x,μ,τ)]}
        """
        n = len(data)
        
        # Shape parameter
        self.alpha_n = self.alpha_0 + (n + 1) / 2
        
        # Rate parameter
        # E[(μ - μ₀)²] = (E[μ] - μ₀)² + Var[μ]
        E_mu_minus_mu0_sq = (self.E_mu - self.mu_0)**2 + self.Var_mu
        
        # E[Σ(xᵢ - μ)²] = Σ(xᵢ - E[μ])² + n·Var[μ]
        E_sum_sq = torch.sum((data - self.E_mu)**2).item() + n * self.Var_mu
        
        self.beta_n = self.beta_0 + 0.5 * (self.lambda_0 * E_mu_minus_mu0_sq + E_sum_sq)
    
    def compute_elbo(self, data: torch.Tensor) -> float:
        """
        Compute the Evidence Lower Bound.
        
        ELBO = E_q[log p(x,μ,τ)] - E_q[log q(μ,τ)]
             = E_q[log p(x|μ,τ)] + E_q[log p(μ|τ)] + E_q[log p(τ)]
               - E_q[log q(μ)] - E_q[log q(τ)]
        """
        n = len(data)
        
        # E[log p(x|μ,τ)] - expected log-likelihood
        E_log_likelihood = (
            0.5 * n * (self.E_log_tau - np.log(2 * np.pi))
            - 0.5 * self.E_tau * (
                torch.sum((data - self.E_mu)**2).item() + n * self.Var_mu
            )
        )
        
        # E[log p(μ|τ)] - expected log prior on μ
        E_log_prior_mu = (
            0.5 * (np.log(self.lambda_0) + self.E_log_tau - np.log(2 * np.pi))
            - 0.5 * self.lambda_0 * self.E_tau * (
                (self.E_mu - self.mu_0)**2 + self.Var_mu
            )
        )
        
        # E[log p(τ)] - expected log prior on τ
        E_log_prior_tau = (
            self.alpha_0 * np.log(self.beta_0)
            - torch.lgamma(torch.tensor(self.alpha_0)).item()
            + (self.alpha_0 - 1) * self.E_log_tau
            - self.beta_0 * self.E_tau
        )
        
        # -E[log q(μ)] - entropy of q(μ)
        H_q_mu = 0.5 * (1 + np.log(2 * np.pi) - np.log(self.lambda_n))
        
        # -E[log q(τ)] - entropy of q(τ)
        H_q_tau = (
            self.alpha_n
            - np.log(self.beta_n)
            + torch.lgamma(torch.tensor(self.alpha_n)).item()
            + (1 - self.alpha_n) * torch.digamma(torch.tensor(self.alpha_n)).item()
        )
        
        elbo = E_log_likelihood + E_log_prior_mu + E_log_prior_tau + H_q_mu + H_q_tau
        
        return elbo
    
    def fit(self, data: torch.Tensor, max_iter: int = 100, 
            tol: float = 1e-6, verbose: bool = True) -> Dict:
        """
        Run CAVI algorithm until convergence.
        """
        history = {
            'elbo': [],
            'mu_n': [],
            'lambda_n': [],
            'alpha_n': [],
            'beta_n': [],
            'E_mu': [],
            'E_tau': []
        }
        
        elbo_prev = -float('inf')
        
        for iteration in range(max_iter):
            # Coordinate ascent updates
            self.update_q_mu(data)
            self.update_q_tau(data)
            
            # Compute ELBO
            elbo = self.compute_elbo(data)
            
            # Record history
            history['elbo'].append(elbo)
            history['mu_n'].append(self.mu_n)
            history['lambda_n'].append(self.lambda_n)
            history['alpha_n'].append(self.alpha_n)
            history['beta_n'].append(self.beta_n)
            history['E_mu'].append(self.E_mu)
            history['E_tau'].append(self.E_tau)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d}: ELBO = {elbo:.4f}, "
                      f"E[μ] = {self.E_mu:.4f}, E[τ] = {self.E_tau:.4f}")
            
            # Check convergence
            if abs(elbo - elbo_prev) < tol:
                if verbose:
                    print(f"\nConverged at iteration {iteration + 1}")
                break
            
            elbo_prev = elbo
        
        return history
    
    def get_posterior_distributions(self) -> Tuple[dist.Distribution, dist.Distribution]:
        """Return the variational distributions."""
        q_mu = dist.Normal(self.mu_n, 1.0 / np.sqrt(self.lambda_n))
        q_tau = dist.Gamma(self.alpha_n, self.beta_n)
        return q_mu, q_tau


def visualize_cavi_results(model: MeanFieldGaussianCAVI, 
                          history: Dict,
                          data: torch.Tensor,
                          true_mu: float = None,
                          true_tau: float = None):
    """Comprehensive visualization of CAVI results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: ELBO convergence
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean parameter convergence
    ax = axes[0, 1]
    ax.plot(history['E_mu'], 'b-', linewidth=2, label='E[μ]')
    if true_mu is not None:
        ax.axhline(true_mu, color='r', linestyle='--', linewidth=2, label='True μ')
    ax.axhline(data.mean().item(), color='g', linestyle=':', linewidth=2, 
               label='Sample mean')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(b) Mean Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Precision parameter convergence
    ax = axes[0, 2]
    ax.plot(history['E_tau'], 'b-', linewidth=2, label='E[τ]')
    if true_tau is not None:
        ax.axhline(true_tau, color='r', linestyle='--', linewidth=2, label='True τ')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('(c) Precision Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Posterior q(μ)
    ax = axes[1, 0]
    q_mu, _ = model.get_posterior_distributions()
    mu_range = torch.linspace(model.E_mu - 3/np.sqrt(model.lambda_n),
                              model.E_mu + 3/np.sqrt(model.lambda_n), 200)
    q_mu_pdf = torch.exp(q_mu.log_prob(mu_range))
    
    ax.plot(mu_range.numpy(), q_mu_pdf.numpy(), 'b-', linewidth=2.5, label='q(μ)')
    ax.fill_between(mu_range.numpy(), q_mu_pdf.numpy(), alpha=0.3, color='blue')
    if true_mu is not None:
        ax.axvline(true_mu, color='r', linestyle='--', linewidth=2, label='True μ')
    ax.axvline(model.E_mu, color='b', linestyle=':', linewidth=2, label='E[μ]')
    ax.set_xlabel('μ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Posterior q(μ)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Posterior q(τ)
    ax = axes[1, 1]
    _, q_tau = model.get_posterior_distributions()
    tau_range = torch.linspace(0.01, model.E_tau * 3, 200)
    q_tau_pdf = torch.exp(q_tau.log_prob(tau_range))
    
    ax.plot(tau_range.numpy(), q_tau_pdf.numpy(), 'g-', linewidth=2.5, label='q(τ)')
    ax.fill_between(tau_range.numpy(), q_tau_pdf.numpy(), alpha=0.3, color='green')
    if true_tau is not None:
        ax.axvline(true_tau, color='r', linestyle='--', linewidth=2, label='True τ')
    ax.axvline(model.E_tau, color='g', linestyle=':', linewidth=2, label='E[τ]')
    ax.set_xlabel('τ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(e) Posterior q(τ)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Data with posterior predictive
    ax = axes[1, 2]
    ax.hist(data.numpy(), bins=20, density=True, alpha=0.6, 
            color='gray', edgecolor='black', label='Data')
    
    # Posterior predictive (approximate by sampling)
    n_samples = 1000
    mu_samples = np.random.normal(model.mu_n, 1/np.sqrt(model.lambda_n), n_samples)
    tau_samples = np.random.gamma(model.alpha_n, 1/model.beta_n, n_samples)
    
    x_range = np.linspace(data.min().item() - 2, data.max().item() + 2, 200)
    pred_pdf = np.zeros_like(x_range)
    for mu_s, tau_s in zip(mu_samples, tau_samples):
        pred_pdf += np.exp(-0.5 * tau_s * (x_range - mu_s)**2) * np.sqrt(tau_s / (2*np.pi))
    pred_pdf /= n_samples
    
    ax.plot(x_range, pred_pdf, 'b-', linewidth=2.5, label='Posterior Predictive')
    if true_mu is not None:
        ax.axvline(true_mu, color='r', linestyle='--', linewidth=2, label='True μ')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(f) Posterior Predictive', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mean_field_cavi.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    true_mu = 3.0
    true_tau = 0.5  # precision = 1/variance
    n_samples = 100
    
    data = torch.randn(n_samples) / np.sqrt(true_tau) + true_mu
    
    print("=" * 60)
    print("Mean-Field CAVI for Gaussian Model")
    print("=" * 60)
    print(f"\nTrue μ = {true_mu}, True τ = {true_tau}")
    print(f"Sample mean = {data.mean().item():.4f}")
    print(f"Sample variance = {data.var().item():.4f}")
    
    # Run CAVI
    model = MeanFieldGaussianCAVI(
        alpha_0=1.0, beta_0=1.0,  # Gamma prior on τ
        mu_0=0.0, lambda_0=0.01   # Normal prior on μ
    )
    
    history = model.fit(data, max_iter=100, verbose=True)
    
    print(f"\nFinal estimates:")
    print(f"  E[μ] = {model.E_mu:.4f} (true: {true_mu})")
    print(f"  E[τ] = {model.E_tau:.4f} (true: {true_tau})")
    
    # Visualize
    visualize_cavi_results(model, history, data, true_mu, true_tau)
```

## Advantages and Limitations

### Advantages

1. **Tractability**: Closed-form updates for many models
2. **Scalability**: Independent factors scale to high dimensions
3. **Interpretability**: Each factor has clear meaning
4. **Convergence**: Guaranteed monotonic improvement in ELBO

### Limitations

1. **No correlations**: Cannot capture posterior dependencies
2. **Underestimates uncertainty**: Typically too confident
3. **Mode selection**: May miss modes in multimodal posteriors
4. **Local optima**: CAVI only finds local optima

## When to Use Mean-Field

**Good for:**

- Parameters with weak posterior correlations
- Large-scale problems where full covariance is infeasible
- Models with conjugate structure
- Quick approximate inference

**Avoid when:**

- Strong parameter correlations are expected
- Accurate uncertainty quantification is critical
- Small problems where MCMC is feasible
- Multimodal posteriors are likely

## Summary

The mean-field assumption:

$$
q(\theta) = \prod_{j=1}^K q_j(\theta_j)
$$

leads to the optimal update:

$$
q_j^*(\theta_j) \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\theta, \mathcal{D})]\}
$$

**Key trade-off**: Simplicity and tractability vs. inability to capture correlations.

## Exercises

### Exercise 1: Correlation Impact

Implement a simulation to show how mean-field approximation quality degrades as posterior correlation increases.

### Exercise 2: Multivariate Gaussian

Derive mean-field updates for a multivariate Gaussian with unknown mean vector and diagonal covariance.

### Exercise 3: CAVI for Linear Regression

Implement CAVI for Bayesian linear regression with unknown weights and noise variance.

## References

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians."

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 10.

3. Wainwright, M. J., & Jordan, M. I. (2008). "Graphical Models, Exponential Families, and Variational Inference."

4. Parisi, G. (1988). *Statistical Field Theory*. (Origin of mean-field terminology)

---

# Appendix: Coordinate Ascent Variational Inference (CAVI)

The following section provides detailed algorithmic and implementation coverage of CAVI, the classical optimization algorithm for mean-field VI.

---

# Coordinate Ascent Variational Inference

## Learning Objectives

By the end of this section, you will be able to:

1. Understand coordinate ascent as an optimization strategy for ELBO
2. Derive CAVI updates for exponential family models
3. Implement efficient CAVI algorithms with convergence monitoring
4. Analyze convergence properties and computational complexity
5. Apply CAVI to Gaussian Mixture Models

## Coordinate Ascent Strategy

Coordinate Ascent Variational Inference (CAVI) is the classical algorithm for optimizing the ELBO under mean-field assumptions. Instead of optimizing all variational parameters simultaneously, CAVI updates one factor at a time while holding others fixed.

### The CAVI Principle

For a mean-field factorization $q(\theta) = \prod_{j=1}^K q_j(\theta_j)$, CAVI iterates:

$$
\text{For } j = 1, \ldots, K: \quad q_j^{(t+1)}(\theta_j) \propto \exp\left\{\mathbb{E}_{q_{-j}^{(t)}}[\log p(\theta, \mathcal{D})]\right\}
$$

where $q_{-j}^{(t)}$ uses the most recent updates for factors $1, \ldots, j-1$ and previous iteration values for factors $j+1, \ldots, K$.

### Algorithm Structure

```
Algorithm: Coordinate Ascent Variational Inference (CAVI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: 
  - Model: p(D, θ) = p(D|θ)p(θ)
  - Factorization: q(θ) = ∏ⱼ qⱼ(θⱼ)
  - Tolerance: ε
  - Max iterations: T

Output: Optimized variational factors {qⱼ*}

1. Initialize: Set initial variational parameters {φⱼ⁽⁰⁾}
2. Compute: ELBO⁽⁰⁾

3. For t = 1, 2, ..., T:
   a. For j = 1, 2, ..., K:
      i.   Compute: E_{q₋ⱼ}[log p(θ, D)]
      ii.  Update: qⱼ⁽ᵗ⁾(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}
      iii. Extract: New variational parameters φⱼ⁽ᵗ⁾
   
   b. Compute: ELBO⁽ᵗ⁾
   
   c. If |ELBO⁽ᵗ⁾ - ELBO⁽ᵗ⁻¹⁾| < ε:
      Return {qⱼ⁽ᵗ⁾}

4. Return {qⱼ⁽ᵀ⁾} (may not have converged)
```

## Convergence Properties

### Monotonicity Theorem

**Theorem**: Each CAVI update cannot decrease the ELBO.

**Proof**: Consider updating factor $q_j$ while holding all others fixed. The ELBO can be written as:

$$
\text{ELBO}(q_1, \ldots, q_K) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \sum_{k=1}^K \mathbb{E}_{q_k}[\log q_k(\theta_k)]
$$

The terms involving $q_j$ are:

$$
\text{ELBO}_j = \mathbb{E}_{q_j}\left[\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\right] - \mathbb{E}_{q_j}[\log q_j(\theta_j)]
$$

This is the negative KL divergence (up to a constant):

$$
\text{ELBO}_j = -\text{KL}\left(q_j(\theta_j) \| \tilde{p}_j(\theta_j)\right) + \text{const}
$$

where $\tilde{p}_j(\theta_j) \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\}$.

Since KL divergence is minimized (equals zero) when $q_j = \tilde{p}_j$, the CAVI update maximizes $\text{ELBO}_j$. Therefore, each update can only increase or maintain the ELBO.

### Convergence to Local Optimum

**Corollary**: CAVI converges to a local optimum of the ELBO.

The ELBO is bounded above by $\log p(\mathcal{D})$. Combined with monotonicity, the sequence of ELBO values must converge. At convergence, no single-factor update can improve the ELBO, which is the definition of a coordinate-wise optimum.

### Rate of Convergence

CAVI typically exhibits linear convergence:

$$
\text{ELBO}^* - \text{ELBO}^{(t)} \leq C \cdot \rho^t
$$

where $\rho < 1$ is the convergence rate and depends on the problem structure.

## CAVI for Exponential Family Models

When both the complete conditional distributions and variational factors belong to exponential families, CAVI updates take a particularly elegant form.

### Exponential Family Background

A distribution is in the exponential family if it can be written as:

$$
p(x | \eta) = h(x) \exp\left\{\eta^\top T(x) - A(\eta)\right\}
$$

where:
- $\eta$ are the **natural parameters**
- $T(x)$ are the **sufficient statistics**
- $A(\eta)$ is the **log-partition function**
- $h(x)$ is the **base measure**

### CAVI Update in Natural Parameters

For exponential family models, the CAVI update for factor $q_j$ reduces to updating natural parameters:

$$
\eta_j^{(t+1)} = \mathbb{E}_{q_{-j}^{(t)}}\left[\eta_j(\theta_{-j}, \mathcal{D})\right]
$$

where $\eta_j(\theta_{-j}, \mathcal{D})$ are the natural parameters of the complete conditional $p(\theta_j | \theta_{-j}, \mathcal{D})$.

### Common Exponential Family Updates

| Distribution | Natural Parameters | Update Rule |
|--------------|-------------------|-------------|
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\eta_1 = \mu/\sigma^2$, $\eta_2 = -1/(2\sigma^2)$ | Mean of expectations |
| Gamma $\text{Ga}(\alpha, \beta)$ | $\eta_1 = \alpha - 1$, $\eta_2 = -\beta$ | Sum of expected statistics |
| Categorical $\text{Cat}(\pi)$ | $\eta_k = \log \pi_k$ | Log of expected probabilities |
| Dirichlet $\text{Dir}(\alpha)$ | $\eta_k = \alpha_k - 1$ | Sum of expected log-probabilities |

## Example: Gaussian Mixture Model

The Gaussian Mixture Model (GMM) is a canonical example for CAVI.

### Model Specification

$$
\begin{aligned}
\text{Mixing weights: } & \pi \sim \text{Dir}(\alpha_0) \\
\text{Component means: } & \mu_k \sim \mathcal{N}(\mu_0, \sigma_0^2) \quad k = 1, \ldots, K \\
\text{Cluster assignments: } & z_i \sim \text{Cat}(\pi) \quad i = 1, \ldots, N \\
\text{Observations: } & x_i | z_i, \{\mu_k\} \sim \mathcal{N}(\mu_{z_i}, \sigma^2)
\end{aligned}
$$

### Mean-Field Factorization

$$
q(\pi, \{\mu_k\}, \{z_i\}) = q(\pi) \prod_{k=1}^K q(\mu_k) \prod_{i=1}^N q(z_i)
$$

### CAVI Updates

**Update for $q(z_i)$ (cluster assignments):**

$$
q^*(z_i = k) \propto \exp\left\{\mathbb{E}[\log \pi_k] + \mathbb{E}\left[\log \mathcal{N}(x_i | \mu_k, \sigma^2)\right]\right\}
$$

Let $r_{ik} = q(z_i = k)$, then:

$$
r_{ik} \propto \exp\left\{\psi(\alpha_k) - \psi\left(\sum_j \alpha_j\right) - \frac{1}{2\sigma^2}\left[(x_i - \mathbb{E}[\mu_k])^2 + \text{Var}[\mu_k]\right]\right\}
$$

**Update for $q(\mu_k)$ (component means):**

$$
q^*(\mu_k) = \mathcal{N}(\mu_k | m_k, s_k^2)
$$

where:

$$
\begin{aligned}
s_k^2 &= \left(\frac{1}{\sigma_0^2} + \frac{N_k}{\sigma^2}\right)^{-1} \\
m_k &= s_k^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_i r_{ik} x_i}{\sigma^2}\right)
\end{aligned}
$$

and $N_k = \sum_i r_{ik}$ is the expected number of points in cluster $k$.

**Update for $q(\pi)$ (mixing weights):**

$$
q^*(\pi) = \text{Dir}(\alpha_1, \ldots, \alpha_K)
$$

where:

$$
\alpha_k = \alpha_0 + \sum_{i=1}^N r_{ik}
$$

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, Categorical
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class GMMCavi:
    """
    Coordinate Ascent VI for Gaussian Mixture Model.
    
    Model:
        π ~ Dir(α₀)
        μₖ ~ N(μ₀, σ₀²)
        zᵢ ~ Cat(π)
        xᵢ | zᵢ ~ N(μ_{zᵢ}, σ²)
    
    Variational family:
        q(π, μ, z) = q(π) ∏ₖ q(μₖ) ∏ᵢ q(zᵢ)
    """
    
    def __init__(self, n_components: int, 
                 alpha_0: float = 1.0,
                 mu_0: float = 0.0,
                 sigma_0: float = 10.0,
                 sigma: float = 1.0):
        """
        Initialize GMM-CAVI.
        
        Args:
            n_components: Number of mixture components K
            alpha_0: Dirichlet prior concentration
            mu_0: Prior mean for component means
            sigma_0: Prior std for component means
            sigma: Known observation noise std
        """
        self.K = n_components
        self.alpha_0 = alpha_0
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.sigma = sigma
        
        # Variational parameters (to be initialized with data)
        self.r = None       # N x K responsibility matrix
        self.alpha = None   # K Dirichlet parameters
        self.m = None       # K component means
        self.s2 = None      # K component variances
    
    def initialize(self, data: torch.Tensor, init_method: str = 'kmeans'):
        """
        Initialize variational parameters.
        """
        N = len(data)
        
        if init_method == 'random':
            # Random initialization
            self.r = torch.rand(N, self.K)
            self.r = self.r / self.r.sum(dim=1, keepdim=True)
        elif init_method == 'kmeans':
            # K-means++ style initialization
            indices = torch.randperm(N)[:self.K]
            centers = data[indices].clone()
            
            # Compute distances and assign
            dists = torch.cdist(data.unsqueeze(1), centers.unsqueeze(0).unsqueeze(2))
            self.r = F.softmax(-dists.squeeze() / self.sigma**2, dim=1)
        
        # Initialize other parameters based on responsibilities
        self._update_q_pi()
        self._update_q_mu(data)
    
    def _update_q_z(self, data: torch.Tensor) -> None:
        """
        Update q(zᵢ) for all data points.
        
        q*(zᵢ = k) ∝ exp{E[log πₖ] - (xᵢ - E[μₖ])²/(2σ²) - Var[μₖ]/(2σ²)}
        """
        N = len(data)
        
        # E[log πₖ] = ψ(αₖ) - ψ(Σⱼ αⱼ)
        E_log_pi = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        
        # Log responsibilities (unnormalized)
        log_r = torch.zeros(N, self.K)
        
        for k in range(self.K):
            # E[(xᵢ - μₖ)²] = (xᵢ - E[μₖ])² + Var[μₖ]
            E_sq_diff = (data - self.m[k])**2 + self.s2[k]
            log_r[:, k] = E_log_pi[k] - 0.5 / self.sigma**2 * E_sq_diff
        
        # Normalize (softmax)
        self.r = F.softmax(log_r, dim=1)
    
    def _update_q_mu(self, data: torch.Tensor) -> None:
        """
        Update q(μₖ) for all components.
        
        q*(μₖ) = N(mₖ, sₖ²)
        """
        N_k = self.r.sum(dim=0)  # Expected count per component
        
        # Posterior precision
        precision = 1/self.sigma_0**2 + N_k / self.sigma**2
        self.s2 = 1 / precision
        
        # Posterior mean
        weighted_sum = (self.r * data.unsqueeze(1)).sum(dim=0)
        self.m = self.s2 * (self.mu_0 / self.sigma_0**2 + weighted_sum / self.sigma**2)
    
    def _update_q_pi(self) -> None:
        """
        Update q(π).
        
        q*(π) = Dir(α₁, ..., αₖ)
        αₖ = α₀ + Σᵢ rᵢₖ
        """
        N_k = self.r.sum(dim=0)
        self.alpha = self.alpha_0 + N_k
    
    def compute_elbo(self, data: torch.Tensor) -> float:
        """
        Compute the Evidence Lower Bound.
        """
        N = len(data)
        N_k = self.r.sum(dim=0)
        
        # E[log p(x|z,μ)] - Expected log-likelihood
        E_log_lik = 0.0
        for k in range(self.K):
            E_sq_diff = (data - self.m[k])**2 + self.s2[k]
            E_log_lik += (self.r[:, k] * (
                -0.5 * np.log(2 * np.pi * self.sigma**2)
                - 0.5 / self.sigma**2 * E_sq_diff
            )).sum()
        
        # E[log p(z|π)] - Expected log prior on z
        E_log_pi = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        E_log_p_z = (self.r * E_log_pi).sum()
        
        # E[log p(π)] - Expected log prior on π
        E_log_p_pi = (
            torch.lgamma(torch.tensor(self.K * self.alpha_0))
            - self.K * torch.lgamma(torch.tensor(self.alpha_0))
            + (self.alpha_0 - 1) * E_log_pi.sum()
        )
        
        # E[log p(μ)] - Expected log prior on μ
        E_log_p_mu = 0.0
        for k in range(self.K):
            E_log_p_mu += (
                -0.5 * np.log(2 * np.pi * self.sigma_0**2)
                - 0.5 / self.sigma_0**2 * ((self.m[k] - self.mu_0)**2 + self.s2[k])
            )
        
        # -E[log q(z)] - Entropy of q(z)
        H_q_z = -(self.r * torch.log(self.r + 1e-10)).sum()
        
        # -E[log q(π)] - Entropy of q(π)
        H_q_pi = (
            torch.lgamma(self.alpha.sum())
            - torch.lgamma(self.alpha).sum()
            + (self.alpha - 1).dot(
                torch.digamma(self.alpha.sum()) - torch.digamma(self.alpha)
            )
        )
        
        # -E[log q(μ)] - Entropy of q(μ)
        H_q_mu = 0.5 * self.K * (1 + np.log(2 * np.pi)) + 0.5 * torch.log(self.s2).sum()
        
        elbo = (E_log_lik + E_log_p_z + E_log_p_pi + E_log_p_mu 
                + H_q_z + H_q_pi + H_q_mu)
        
        return elbo.item()
    
    def fit(self, data: torch.Tensor, max_iter: int = 100,
            tol: float = 1e-6, verbose: bool = True) -> Dict:
        """
        Run CAVI algorithm.
        """
        # Initialize
        self.initialize(data)
        
        history = {
            'elbo': [],
            'r': [],
            'm': [],
            'alpha': []
        }
        
        elbo_prev = -float('inf')
        
        for iteration in range(max_iter):
            # CAVI updates
            self._update_q_z(data)
            self._update_q_mu(data)
            self._update_q_pi()
            
            # Compute ELBO
            elbo = self.compute_elbo(data)
            
            # Record history
            history['elbo'].append(elbo)
            history['r'].append(self.r.clone())
            history['m'].append(self.m.clone())
            history['alpha'].append(self.alpha.clone())
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d}: ELBO = {elbo:.4f}, "
                      f"means = {self.m.numpy()}")
            
            # Check convergence
            if abs(elbo - elbo_prev) < tol:
                if verbose:
                    print(f"\nConverged at iteration {iteration + 1}")
                break
            
            elbo_prev = elbo
        
        return history
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments.
        """
        self._update_q_z(data)
        return self.r.argmax(dim=1)


def visualize_gmm_cavi(model: GMMCavi, history: Dict, data: torch.Tensor,
                       true_labels: torch.Tensor = None):
    """Visualize GMM CAVI results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: ELBO convergence
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Component means convergence
    ax = axes[0, 1]
    means_history = torch.stack(history['m'])
    for k in range(model.K):
        ax.plot(means_history[:, k], linewidth=2, label=f'μ_{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(b) Component Means', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final clustering
    ax = axes[0, 2]
    predictions = model.r.argmax(dim=1)
    colors = plt.cm.tab10(predictions.numpy() / model.K)
    ax.scatter(data.numpy(), np.zeros_like(data.numpy()), c=colors, alpha=0.6, s=50)
    for k in range(model.K):
        ax.axvline(model.m[k].item(), color=plt.cm.tab10(k / model.K), 
                   linestyle='--', linewidth=2, label=f'μ_{k+1}')
    ax.set_xlabel('x', fontsize=11)
    ax.set_title('(c) Final Clustering', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Responsibility evolution
    ax = axes[1, 0]
    r_history = torch.stack(history['r'])
    sample_idx = 0  # Track first data point
    for k in range(model.K):
        ax.plot(r_history[:, sample_idx, k], linewidth=2, label=f'r_{sample_idx+1},{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Responsibility', fontsize=11)
    ax.set_title(f'(d) Responsibilities (point {sample_idx+1})', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Dirichlet parameters
    ax = axes[1, 1]
    alpha_history = torch.stack(history['alpha'])
    for k in range(model.K):
        ax.plot(alpha_history[:, k], linewidth=2, label=f'α_{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Concentration', fontsize=11)
    ax.set_title('(e) Dirichlet Parameters', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final mixture density
    ax = axes[1, 2]
    x_range = torch.linspace(data.min() - 2, data.max() + 2, 500)
    
    E_pi = model.alpha / model.alpha.sum()
    mixture_pdf = torch.zeros_like(x_range)
    
    for k in range(model.K):
        component_pdf = E_pi[k] * torch.exp(
            -0.5 * (x_range - model.m[k])**2 / (model.sigma**2 + model.s2[k])
        ) / np.sqrt(2 * np.pi * (model.sigma**2 + model.s2[k]))
        mixture_pdf += component_pdf
        ax.plot(x_range.numpy(), component_pdf.numpy(), '--', 
                linewidth=1.5, alpha=0.7, label=f'Component {k+1}')
    
    ax.plot(x_range.numpy(), mixture_pdf.numpy(), 'b-', linewidth=2.5, 
            label='Mixture')
    ax.hist(data.numpy(), bins=30, density=True, alpha=0.4, color='gray')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(f) Fitted Mixture', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_cavi.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic mixture data
    n_samples = 300
    true_means = torch.tensor([-3.0, 0.0, 4.0])
    true_weights = torch.tensor([0.3, 0.4, 0.3])
    sigma = 1.0
    
    # Sample from mixture
    z_true = torch.multinomial(true_weights, n_samples, replacement=True)
    data = torch.randn(n_samples) * sigma + true_means[z_true]
    
    print("=" * 60)
    print("Gaussian Mixture Model CAVI")
    print("=" * 60)
    print(f"\nTrue means: {true_means.numpy()}")
    print(f"True weights: {true_weights.numpy()}")
    
    # Fit GMM with CAVI
    model = GMMCavi(n_components=3, alpha_0=1.0, sigma=sigma)
    history = model.fit(data, max_iter=100, verbose=True)
    
    print(f"\nFinal estimates:")
    print(f"  Means: {model.m.numpy()}")
    print(f"  Expected weights: {(model.alpha / model.alpha.sum()).numpy()}")
    
    # Visualize
    visualize_gmm_cavi(model, history, data)
```

## Computational Complexity

### Per-Iteration Cost

For a model with $K$ factors, $N$ data points:

| Component | Complexity |
|-----------|------------|
| Update $q(z_i)$ for all $i$ | $O(NK)$ |
| Update $q(\mu_k)$ for all $k$ | $O(NK)$ |
| Update $q(\pi)$ | $O(K)$ |
| Compute ELBO | $O(NK)$ |
| **Total per iteration** | $O(NK)$ |

### Comparison with EM

CAVI for GMM has the same complexity as the EM algorithm, but provides full posterior distributions rather than point estimates.

## Summary

CAVI provides a systematic approach to optimizing the ELBO:

**Algorithm**:
1. Initialize variational parameters
2. Update each factor in turn: $q_j^* \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\theta, \mathcal{D})]\}$
3. Monitor ELBO until convergence

**Properties**:
- Monotonic ELBO improvement
- Converges to local optimum
- Closed-form updates for exponential families
- Same complexity as EM for mixture models

## Exercises

### Exercise 1: CAVI for Factor Analysis

Derive and implement CAVI for a probabilistic factor analysis model.

### Exercise 2: Convergence Analysis

Empirically measure the convergence rate of CAVI for GMM as a function of cluster separation.

### Exercise 3: Initialization Sensitivity

Study how different initialization strategies affect the final ELBO and cluster quality.

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 10.

2. Blei, D. M., & Jordan, M. I. (2006). "Variational Inference for Dirichlet Process Mixtures."

3. Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). "Stochastic Variational Inference."

4. Wainwright, M. J., & Jordan, M. I. (2008). "Graphical Models, Exponential Families, and Variational Inference."
