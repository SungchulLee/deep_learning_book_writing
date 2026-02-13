# Prior, Likelihood, and Posterior

Bayesian inference revolves around three fundamental quantities: the **prior**, the **likelihood**, and the **posterior**. Understanding their roles, mathematical properties, and interplay is essential for applying Bayesian methods to machine learning. This section provides a rigorous treatment of each component and how they combine through Bayes' theorem.

---

## The Bayesian Trinity

### Bayes' Theorem: The Master Equation

For a parameter $\theta$ and observed data $\mathcal{D}$, Bayes' theorem states:

$$
\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\overbrace{p(\mathcal{D} \mid \theta)}^{\text{likelihood}} \cdot \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}
$$

Or in proportional form (often more useful computationally):

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \cdot p(\theta)
$$

Each component plays a distinct role:

| Component | Symbol | Role | Question Answered |
|-----------|--------|------|-------------------|
| Prior | $p(\theta)$ | Encodes beliefs before data | "What do I believe about $\theta$ before seeing any data?" |
| Likelihood | $p(\mathcal{D} \mid \theta)$ | Connects parameters to data | "How probable is my data if $\theta$ were true?" |
| Posterior | $p(\theta \mid \mathcal{D})$ | Updated beliefs after data | "What should I believe about $\theta$ after seeing data?" |
| Evidence | $p(\mathcal{D})$ | Normalizing constant | "How probable is my data across all possible $\theta$?" |

---

## The Prior Distribution

### Definition and Interpretation

The **prior distribution** $p(\theta)$ represents our beliefs about the parameter $\theta$ **before** observing any data. It encodes:

- Domain knowledge from experts
- Results from previous experiments
- Physical constraints on parameters
- Deliberate ignorance (uninformative priors)

### Types of Priors

**1. Informative Priors**

Encode genuine prior knowledge:

$$
\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)
$$

where $\mu_0$ and $\sigma_0^2$ reflect prior beliefs about location and uncertainty.

*Example*: For human body temperature (°C), an informative prior might be $\theta \sim \mathcal{N}(37, 0.5^2)$.

**2. Weakly Informative Priors**

Provide mild regularization without strong assumptions:

$$
\theta \sim \mathcal{N}(0, 10^2)
$$

This says "parameters are probably not enormous" without specifying much else.

**3. Uninformative (Diffuse) Priors**

Attempt to "let the data speak":

$$
p(\theta) \propto 1 \quad \text{(improper uniform)}
$$

**Warning**: Improper priors (those that don't integrate to 1) require careful handling—they can lead to improper posteriors.

**4. Jeffreys Priors**

Invariant under reparameterization. For a parameter $\theta$ with Fisher information $I(\theta)$:

$$
p_J(\theta) \propto \sqrt{I(\theta)} = \sqrt{-\mathbb{E}\left[\frac{\partial^2 \log p(\mathcal{D} \mid \theta)}{\partial \theta^2}\right]}
$$

*Example*: For Bernoulli likelihood with parameter $\theta \in (0,1)$:

$$
p_J(\theta) \propto \theta^{-1/2}(1-\theta)^{-1/2} = \text{Beta}(1/2, 1/2)
$$

**5. Maximum Entropy Priors**

Encode only specified constraints, assuming nothing else:

$$
p(\theta) = \arg\max_q H(q) \quad \text{subject to constraints}
$$

where $H(q) = -\int q(\theta) \log q(\theta) \, d\theta$.

### Prior Elicitation

Translating expert knowledge into mathematical priors:

| Expert Statement | Mathematical Prior |
|------------------|-------------------|
| "Probably between 0 and 10" | $\theta \sim \text{Uniform}(0, 10)$ |
| "Most likely around 5, rarely above 8" | $\theta \sim \mathcal{N}(5, 1.5^2)$ truncated |
| "Could be anywhere, but extreme values unlikely" | $\theta \sim \text{Cauchy}(0, 2.5)$ |
| "Should be positive, typically small" | $\theta \sim \text{Exponential}(\lambda)$ |
| "Probably sparse (many zeros)" | $\theta \sim \text{Laplace}(0, b)$ |

### Prior Predictive Distribution

The prior predictive is what the prior implies about observable data **before** seeing any:

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) \, d\theta
$$

This is useful for **prior predictive checks**: if $p(\mathcal{D})$ assigns negligible probability to reasonable data configurations, the prior may be poorly calibrated.

---

## The Likelihood Function

### Definition

The **likelihood function** $\mathcal{L}(\theta; \mathcal{D}) = p(\mathcal{D} \mid \theta)$ measures how well parameter value $\theta$ explains the observed data $\mathcal{D}$.

**Critical distinction**: 

- As a function of $\mathcal{D}$ (with $\theta$ fixed): This is a probability distribution
- As a function of $\theta$ (with $\mathcal{D}$ fixed): This is the likelihood—**not** a probability distribution over $\theta$

$$
\int p(\mathcal{D} \mid \theta) \, d\mathcal{D} = 1 \quad \text{(integrates over data)}
$$

$$
\int p(\mathcal{D} \mid \theta) \, d\theta \neq 1 \quad \text{(does NOT integrate over } \theta \text{)}
$$

### Log-Likelihood

For numerical stability and analytical convenience, we typically work with the **log-likelihood**:

$$
\ell(\theta; \mathcal{D}) = \log p(\mathcal{D} \mid \theta)
$$

For independent observations $\mathcal{D} = \{x_1, \ldots, x_n\}$:

$$
\ell(\theta; \mathcal{D}) = \sum_{i=1}^{n} \log p(x_i \mid \theta)
$$

### Likelihood Principle

The **likelihood principle** states that all information about $\theta$ contained in the data is captured by the likelihood function. Two datasets yielding the same likelihood function (up to a constant) contain identical information about $\theta$.

### Common Likelihood Functions

**Bernoulli/Binomial** (binary outcomes):

$$
p(\mathcal{D} \mid \theta) = \theta^k (1-\theta)^{n-k}
$$

where $k$ = number of successes in $n$ trials.

**Gaussian** (continuous measurements with known variance):

$$
p(\mathcal{D} \mid \mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

**Poisson** (count data):

$$
p(\mathcal{D} \mid \lambda) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}
$$

**Categorical/Multinomial** (multi-class):

$$
p(\mathcal{D} \mid \boldsymbol{\theta}) = \prod_{k=1}^{K} \theta_k^{n_k}
$$

where $n_k$ = count of category $k$.

### Sufficient Statistics

A statistic $T(\mathcal{D})$ is **sufficient** for $\theta$ if the likelihood factors as:

$$
p(\mathcal{D} \mid \theta) = g(T(\mathcal{D}), \theta) \cdot h(\mathcal{D})
$$

The posterior depends on $\mathcal{D}$ only through $T(\mathcal{D})$:

$$
p(\theta \mid \mathcal{D}) = p(\theta \mid T(\mathcal{D}))
$$

| Distribution | Sufficient Statistic |
|--------------|---------------------|
| Bernoulli | $\sum_i x_i$ (number of successes) |
| Gaussian (known $\sigma^2$) | $\bar{x} = \frac{1}{n}\sum_i x_i$ (sample mean) |
| Gaussian (unknown $\mu, \sigma^2$) | $(\bar{x}, \sum_i (x_i - \bar{x})^2)$ |
| Poisson | $\sum_i x_i$ (total count) |
| Exponential | $\sum_i x_i$ (total waiting time) |

---

## The Posterior Distribution

### Definition

The **posterior distribution** $p(\theta \mid \mathcal{D})$ represents our updated beliefs about $\theta$ after observing data $\mathcal{D}$:

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}
$$

The posterior synthesizes prior knowledge and data evidence in an optimal way.

### Computing the Posterior

**Analytical (closed-form)**: Possible for conjugate prior-likelihood pairs (see next chapter).

**Grid approximation**: Discretize $\theta$, compute unnormalized posterior on grid, normalize.

**Monte Carlo methods**: MCMC (Metropolis-Hastings, HMC, Gibbs) to sample from posterior.

**Variational inference**: Approximate posterior with tractable distribution family.

### Posterior Summaries

Once we have the posterior, we extract useful summaries:

**Point Estimates**:

| Estimate | Definition | Property |
|----------|------------|----------|
| Posterior Mean | $\mathbb{E}[\theta \mid \mathcal{D}] = \int \theta \, p(\theta \mid \mathcal{D}) \, d\theta$ | Minimizes squared error loss |
| Posterior Median | $\theta^*$ such that $P(\theta \leq \theta^* \mid \mathcal{D}) = 0.5$ | Minimizes absolute error loss |
| Posterior Mode (MAP) | $\arg\max_\theta p(\theta \mid \mathcal{D})$ | Minimizes 0-1 loss |

**Uncertainty Quantification**:

- **Posterior variance**: $\text{Var}[\theta \mid \mathcal{D}]$
- **Credible intervals**: Region $C$ such that $P(\theta \in C \mid \mathcal{D}) = 1 - \alpha$

**Credible Interval Types**:

1. **Equal-tailed**: $P(\theta < a \mid \mathcal{D}) = P(\theta > b \mid \mathcal{D}) = \alpha/2$
2. **Highest Posterior Density (HPD)**: Shortest interval containing $1-\alpha$ probability mass

### Posterior vs Confidence Intervals

| Aspect | Bayesian Credible Interval | Frequentist Confidence Interval |
|--------|---------------------------|--------------------------------|
| Interpretation | "95% probability $\theta$ is in interval" | "95% of such intervals contain true $\theta$" |
| Fixed quantity | Interval | Parameter $\theta$ |
| Random quantity | Parameter $\theta$ | Interval bounds |
| Prior dependence | Yes | No |
| Single-sample validity | Yes | Only as coverage guarantee |

---

## The Evidence (Marginal Likelihood)

### Definition

The **evidence** or **marginal likelihood** $p(\mathcal{D})$ is the normalizing constant:

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

It represents the probability of the data averaged over all possible parameter values, weighted by the prior.

### Role in Bayesian Inference

**For parameter inference**: Often ignored (we work with unnormalized posteriors).

**For model comparison**: Critical! The evidence measures how well a model (prior + likelihood) predicts the observed data.

### Computing the Evidence

The evidence integral is often intractable. Estimation methods include:

| Method | Description | Use Case |
|--------|-------------|----------|
| Analytical | Closed-form for conjugate models | Simple models |
| Harmonic mean | $\hat{p}(\mathcal{D}) = \left[\frac{1}{S}\sum_{s=1}^S \frac{1}{p(\mathcal{D} \mid \theta^{(s)})}\right]^{-1}$ | **Avoid!** High variance |
| Importance sampling | Weighted samples from proposal | Moderate dimensions |
| Nested sampling | Sequential compression | Model comparison |
| Thermodynamic integration | Path from prior to posterior | Rigorous but expensive |
| Variational bound (ELBO) | $\log p(\mathcal{D}) \geq \text{ELBO}$ | Approximate, scalable |

---

## The Mechanics of Updating

### Prior-to-Posterior Flow

Visualizing how beliefs transform:

```
Prior p(θ)          ×    Likelihood p(D|θ)     ∝    Posterior p(θ|D)
     │                         │                         │
     ▼                         ▼                         ▼
[Broad/uncertain]    ×    [Peaked at MLE]     =    [Refined beliefs]
```

### Precision-Weighted Averaging (Gaussian Case)

For Gaussian prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$ and Gaussian likelihood with known variance $\sigma^2$:

$$
p(\theta \mid \mathcal{D}) = \mathcal{N}(\mu_n, \sigma_n^2)
$$

where:

$$
\mu_n = \frac{\frac{1}{\sigma_0^2}\mu_0 + \frac{n}{\sigma^2}\bar{x}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} = \frac{\tau_0 \mu_0 + \tau_{\text{data}} \bar{x}}{\tau_0 + \tau_{\text{data}}}
$$

$$
\sigma_n^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} = \frac{1}{\tau_0 + \tau_{\text{data}}}
$$

Here $\tau = 1/\sigma^2$ is the **precision** (inverse variance).

**Key insight**: The posterior mean is a **precision-weighted average** of prior mean and data mean. More precise (lower variance) sources get more weight.

### Asymptotic Behavior

**Bernstein-von Mises Theorem**: Under regularity conditions, as $n \to \infty$:

$$
p(\theta \mid \mathcal{D}_n) \xrightarrow{d} \mathcal{N}\left(\hat{\theta}_{\text{MLE}}, \frac{1}{n I(\theta_0)}\right)
$$

where $I(\theta_0)$ is the Fisher information at the true parameter.

**Implications**:

1. The posterior concentrates around the true parameter
2. The prior is "washed out" by sufficient data
3. Bayesian and frequentist inference agree asymptotically
4. Posterior variance shrinks as $O(1/n)$

---

## Python Implementation

```python
"""
Prior, Likelihood, and Posterior: Core Bayesian Machinery

This module implements the fundamental components of Bayesian inference
with visualizations and practical demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Core Classes
# =============================================================================

@dataclass
class BayesianModel:
    """
    Encapsulates a Bayesian model with prior and likelihood.
    
    Attributes
    ----------
    prior : callable
        Prior density p(θ)
    likelihood : callable
        Likelihood function p(D|θ)
    theta_range : tuple
        Valid range for parameter θ
    """
    prior: Callable[[np.ndarray], np.ndarray]
    likelihood: Callable[[np.ndarray, np.ndarray], np.ndarray]
    theta_range: Tuple[float, float]
    
    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """Log prior density."""
        return np.log(self.prior(theta) + 1e-300)
    
    def log_likelihood(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Log likelihood."""
        return np.log(self.likelihood(theta, data) + 1e-300)
    
    def unnormalized_posterior(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Unnormalized posterior: likelihood × prior."""
        return self.likelihood(theta, data) * self.prior(theta)
    
    def log_unnormalized_posterior(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Log unnormalized posterior."""
        return self.log_likelihood(theta, data) + self.log_prior(theta)
    
    def compute_evidence(self, data: np.ndarray, n_points: int = 1000) -> float:
        """
        Compute evidence p(D) via numerical integration.
        
        Parameters
        ----------
        data : array
            Observed data
        n_points : int
            Number of quadrature points
        
        Returns
        -------
        float
            Evidence (marginal likelihood)
        """
        def integrand(theta):
            return self.unnormalized_posterior(np.array([theta]), data)[0]
        
        evidence, _ = quad(integrand, self.theta_range[0], self.theta_range[1])
        return evidence
    
    def posterior_grid(
        self, 
        data: np.ndarray, 
        n_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalized posterior on a grid.
        
        Returns
        -------
        theta_grid : array
            Grid of parameter values
        posterior : array
            Normalized posterior density at each grid point
        """
        theta_grid = np.linspace(self.theta_range[0], self.theta_range[1], n_points)
        unnorm_post = self.unnormalized_posterior(theta_grid, data)
        
        # Normalize via trapezoidal integration
        evidence = np.trapz(unnorm_post, theta_grid)
        posterior = unnorm_post / evidence
        
        return theta_grid, posterior


# =============================================================================
# Posterior Summaries
# =============================================================================

def posterior_mean(theta_grid: np.ndarray, posterior: np.ndarray) -> float:
    """Compute posterior mean E[θ|D]."""
    return np.trapz(theta_grid * posterior, theta_grid)


def posterior_variance(theta_grid: np.ndarray, posterior: np.ndarray) -> float:
    """Compute posterior variance Var[θ|D]."""
    mean = posterior_mean(theta_grid, posterior)
    return np.trapz((theta_grid - mean)**2 * posterior, theta_grid)


def posterior_mode(theta_grid: np.ndarray, posterior: np.ndarray) -> float:
    """Compute posterior mode (MAP estimate)."""
    return theta_grid[np.argmax(posterior)]


def credible_interval(
    theta_grid: np.ndarray, 
    posterior: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute equal-tailed credible interval.
    
    Parameters
    ----------
    alpha : float
        Significance level (default 0.05 for 95% CI)
    
    Returns
    -------
    tuple
        (lower, upper) bounds of credible interval
    """
    # Compute CDF
    cdf = np.cumsum(posterior) * (theta_grid[1] - theta_grid[0])
    cdf = cdf / cdf[-1]  # Ensure ends at 1
    
    # Find quantiles
    lower_idx = np.searchsorted(cdf, alpha / 2)
    upper_idx = np.searchsorted(cdf, 1 - alpha / 2)
    
    return theta_grid[lower_idx], theta_grid[min(upper_idx, len(theta_grid)-1)]


def hpd_interval(
    theta_grid: np.ndarray, 
    posterior: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute Highest Posterior Density (HPD) interval.
    
    The shortest interval containing (1-alpha) probability mass.
    """
    # Sort by density (descending)
    sorted_indices = np.argsort(posterior)[::-1]
    sorted_posterior = posterior[sorted_indices]
    sorted_theta = theta_grid[sorted_indices]
    
    # Accumulate until we reach (1-alpha) mass
    cumsum = np.cumsum(sorted_posterior) * (theta_grid[1] - theta_grid[0])
    cutoff_idx = np.searchsorted(cumsum, 1 - alpha)
    
    # HPD region is all theta with density above threshold
    hpd_theta = sorted_theta[:cutoff_idx+1]
    
    return hpd_theta.min(), hpd_theta.max()


# =============================================================================
# Example: Binomial-Beta Model
# =============================================================================

def create_beta_binomial_model(alpha_prior: float, beta_prior: float) -> BayesianModel:
    """
    Create a Beta-Binomial Bayesian model.
    
    Prior: θ ~ Beta(α, β)
    Likelihood: k | θ ~ Binomial(n, θ)
    """
    def prior(theta):
        return stats.beta.pdf(theta, alpha_prior, beta_prior)
    
    def likelihood(theta, data):
        k, n = data  # k successes in n trials
        return stats.binom.pmf(k, n, theta)
    
    return BayesianModel(
        prior=prior,
        likelihood=likelihood,
        theta_range=(0.001, 0.999)
    )


# =============================================================================
# Example: Gaussian Model (Known Variance)
# =============================================================================

def create_gaussian_model(
    prior_mean: float, 
    prior_var: float, 
    known_var: float
) -> BayesianModel:
    """
    Create a Gaussian Bayesian model with known variance.
    
    Prior: μ ~ N(μ₀, σ₀²)
    Likelihood: x | μ ~ N(μ, σ²) where σ² is known
    """
    def prior(mu):
        return stats.norm.pdf(mu, prior_mean, np.sqrt(prior_var))
    
    def likelihood(mu, data):
        # Product of Gaussian likelihoods
        result = np.ones_like(mu, dtype=float)
        for x in data:
            result *= stats.norm.pdf(x, mu, np.sqrt(known_var))
        return result
    
    # Set range based on prior ± 4 standard deviations plus data range
    return BayesianModel(
        prior=prior,
        likelihood=likelihood,
        theta_range=(prior_mean - 5*np.sqrt(prior_var), 
                     prior_mean + 5*np.sqrt(prior_var))
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_bayesian_update(
    model: BayesianModel,
    data: np.ndarray,
    true_theta: Optional[float] = None,
    title: str = "Bayesian Update"
):
    """
    Visualize prior, likelihood, and posterior.
    """
    theta_grid = np.linspace(model.theta_range[0], model.theta_range[1], 1000)
    
    # Compute components
    prior_vals = model.prior(theta_grid)
    likelihood_vals = model.likelihood(theta_grid, data)
    _, posterior_vals = model.posterior_grid(data)
    
    # Normalize for visualization
    prior_vals = prior_vals / np.max(prior_vals)
    likelihood_vals = likelihood_vals / np.max(likelihood_vals)
    posterior_vals = posterior_vals / np.max(posterior_vals)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(theta_grid, prior_vals, alpha=0.3, color='blue', label='Prior')
    ax.plot(theta_grid, prior_vals, 'b-', linewidth=2)
    
    ax.fill_between(theta_grid, likelihood_vals, alpha=0.3, color='green', label='Likelihood')
    ax.plot(theta_grid, likelihood_vals, 'g-', linewidth=2)
    
    ax.fill_between(theta_grid, posterior_vals, alpha=0.3, color='red', label='Posterior')
    ax.plot(theta_grid, posterior_vals, 'r-', linewidth=2)
    
    if true_theta is not None:
        ax.axvline(true_theta, color='black', linestyle='--', linewidth=2, 
                   label=f'True θ = {true_theta}')
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_posterior_summaries(
    theta_grid: np.ndarray,
    posterior: np.ndarray,
    true_theta: Optional[float] = None
):
    """
    Visualize posterior with point estimates and credible intervals.
    """
    mean = posterior_mean(theta_grid, posterior)
    mode = posterior_mode(theta_grid, posterior)
    var = posterior_variance(theta_grid, posterior)
    ci = credible_interval(theta_grid, posterior, 0.05)
    hpd = hpd_interval(theta_grid, posterior, 0.05)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Posterior
    ax.fill_between(theta_grid, posterior, alpha=0.4, color='steelblue')
    ax.plot(theta_grid, posterior, 'b-', linewidth=2, label='Posterior')
    
    # Point estimates
    ax.axvline(mean, color='red', linestyle='-', linewidth=2, 
               label=f'Mean = {mean:.3f}')
    ax.axvline(mode, color='orange', linestyle='--', linewidth=2, 
               label=f'Mode (MAP) = {mode:.3f}')
    
    # Credible interval
    ax.axvspan(ci[0], ci[1], alpha=0.2, color='green', 
               label=f'95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
    
    # True value
    if true_theta is not None:
        ax.axvline(true_theta, color='black', linestyle=':', linewidth=2,
                   label=f'True θ = {true_theta}')
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title(f'Posterior Summary (σ = {np.sqrt(var):.3f})', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Demonstration: Effect of Prior and Data
# =============================================================================

def demonstrate_prior_data_tradeoff():
    """
    Show how the posterior balances prior and data.
    """
    np.random.seed(42)
    
    # True parameter
    true_theta = 0.7
    
    # Generate data
    n_obs = 10
    data = np.random.binomial(1, true_theta, n_obs)
    k = data.sum()  # successes
    
    print(f"Data: {k} successes in {n_obs} trials")
    print(f"MLE: {k/n_obs:.3f}")
    print()
    
    # Different priors
    priors = [
        ("Uniform (α=1, β=1)", 1, 1),
        ("Skeptical (α=2, β=8)", 2, 8),  # Prior believes low θ
        ("Confident (α=8, β=2)", 8, 2),  # Prior believes high θ
        ("Strong (α=20, β=20)", 20, 20),  # Strong prior at 0.5
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, (name, alpha, beta) in zip(axes.flat, priors):
        model = create_beta_binomial_model(alpha, beta)
        theta_grid, posterior = model.posterior_grid(np.array([k, n_obs]))
        
        # Plot prior
        prior_vals = stats.beta.pdf(theta_grid, alpha, beta)
        prior_vals = prior_vals / np.max(prior_vals)
        ax.plot(theta_grid, prior_vals, 'b--', linewidth=2, label='Prior', alpha=0.7)
        
        # Plot posterior
        posterior_norm = posterior / np.max(posterior)
        ax.fill_between(theta_grid, posterior_norm, alpha=0.4, color='red')
        ax.plot(theta_grid, posterior_norm, 'r-', linewidth=2, label='Posterior')
        
        # True value and MLE
        ax.axvline(true_theta, color='black', linestyle=':', linewidth=2, label=f'True θ')
        ax.axvline(k/n_obs, color='green', linestyle='--', linewidth=2, label='MLE')
        
        # Posterior mean
        post_mean = posterior_mean(theta_grid, posterior)
        ax.axvline(post_mean, color='red', linestyle=':', linewidth=1.5)
        
        ax.set_title(f'{name}\nPosterior mean = {post_mean:.3f}', fontsize=11)
        ax.set_xlabel('θ')
        ax.set_ylabel('Density (normalized)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Prior-Data Tradeoff ({k}/{n_obs} successes, true θ = {true_theta})', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('prior_data_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()


def demonstrate_data_overwhelming_prior():
    """
    Show how increasing data overwhelms the prior.
    """
    np.random.seed(42)
    
    true_theta = 0.7
    
    # Strong prior at wrong location
    prior_alpha, prior_beta = 10, 40  # Prior mean = 0.2
    
    sample_sizes = [1, 5, 20, 100, 500]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for ax, n in zip(axes, sample_sizes):
        # Generate data
        data = np.random.binomial(1, true_theta, n)
        k = data.sum()
        
        model = create_beta_binomial_model(prior_alpha, prior_beta)
        theta_grid, posterior = model.posterior_grid(np.array([k, n]))
        
        # Prior
        prior_vals = stats.beta.pdf(theta_grid, prior_alpha, prior_beta)
        ax.plot(theta_grid, prior_vals / np.max(prior_vals), 'b--', 
                linewidth=2, label='Prior', alpha=0.7)
        
        # Posterior
        ax.fill_between(theta_grid, posterior / np.max(posterior), 
                        alpha=0.4, color='red')
        ax.plot(theta_grid, posterior / np.max(posterior), 'r-', 
                linewidth=2, label='Posterior')
        
        # True value
        ax.axvline(true_theta, color='black', linestyle=':', linewidth=2)
        
        post_mean = posterior_mean(theta_grid, posterior)
        ax.set_title(f'n = {n}\nPost. mean = {post_mean:.3f}')
        ax.set_xlabel('θ')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    
    plt.suptitle(f'Data Overwhelming Prior (Prior mean = 0.2, True θ = {true_theta})', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('data_overwhelming_prior.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Prior: Beta(10, 40) with mean 0.2")
    print(f"True θ: {true_theta}")
    print("\nAs n increases, posterior mean approaches true value:")
    for n in sample_sizes:
        data = np.random.binomial(1, true_theta, n)
        k = data.sum()
        model = create_beta_binomial_model(prior_alpha, prior_beta)
        theta_grid, posterior = model.posterior_grid(np.array([k, n]))
        print(f"  n = {n:4d}: posterior mean = {posterior_mean(theta_grid, posterior):.4f}")


# =============================================================================
# Main Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEMONSTRATION: PRIOR, LIKELIHOOD, AND POSTERIOR")
    print("=" * 60)
    
    print("\n1. Basic Bayesian Update (Beta-Binomial)")
    print("-" * 40)
    
    # Simple example
    model = create_beta_binomial_model(alpha_prior=2, beta_prior=2)
    data = np.array([7, 10])  # 7 successes in 10 trials
    
    theta_grid, posterior = model.posterior_grid(data)
    
    print(f"Prior: Beta(2, 2)")
    print(f"Data: 7 successes in 10 trials")
    print(f"Posterior: Beta(9, 5)")  # Conjugate update
    print(f"\nPosterior summaries:")
    print(f"  Mean: {posterior_mean(theta_grid, posterior):.4f}")
    print(f"  Mode: {posterior_mode(theta_grid, posterior):.4f}")
    print(f"  Std:  {np.sqrt(posterior_variance(theta_grid, posterior)):.4f}")
    print(f"  95% CI: {credible_interval(theta_grid, posterior)}")
    
    # Evidence
    evidence = model.compute_evidence(data)
    print(f"  Evidence p(D): {evidence:.6f}")
    
    print("\n2. Effect of Different Priors")
    print("-" * 40)
    demonstrate_prior_data_tradeoff()
    print("See: prior_data_tradeoff.png")
    
    print("\n3. Data Overwhelming Prior")
    print("-" * 40)
    demonstrate_data_overwhelming_prior()
    print("See: data_overwhelming_prior.png")
```

---

## Summary

| Component | Definition | Role |
|-----------|------------|------|
| **Prior** $p(\theta)$ | Belief before data | Encodes domain knowledge |
| **Likelihood** $p(\mathcal{D} \mid \theta)$ | Data probability given parameter | Connects model to observations |
| **Posterior** $p(\theta \mid \mathcal{D})$ | Belief after data | Optimal synthesis of prior + data |
| **Evidence** $p(\mathcal{D})$ | Marginal data probability | Normalization; model comparison |

### Key Insights

1. **Posterior ∝ Likelihood × Prior**: The fundamental update equation
2. **Sufficient statistics**: Data summarization without information loss
3. **Precision-weighted averaging**: More precise sources get more weight
4. **Asymptotic dominance**: With enough data, the posterior converges to truth regardless of (reasonable) prior
5. **Credible intervals**: Direct probability statements about parameters

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Probability as belief | Ch13: Probability Belief | Philosophical foundation |
| Conjugate priors | Ch13: Conjugate Priors | Analytical tractability |
| Beta-Binomial | Ch13: Bernoulli-Beta | Detailed conjugate example |
| Gaussian inference | Ch13: Gaussian Models | Continuous parameter estimation |
| Evidence computation | Ch13: Model Evidence | Marginal likelihood methods |
| MCMC | Ch13: MCMC | Posterior sampling |
| Variational inference | Ch13: VI | Approximate posteriors |

### Key References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapters 1-2.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapters 3, 5.
- Robert, C. P. (2007). *The Bayesian Choice* (2nd ed.). Springer.

---

# Appendix: Continuous Bayesian Inference

The following section extends the discrete examples above to continuous parameter spaces, developing the Beta-Binomial and Normal-Normal models with full posterior analysis.

---

# Continuous Bayesian Inference

## Overview

This module extends Bayesian inference to continuous parameter spaces, where we work with probability density functions instead of discrete probabilities. We develop the Beta-Binomial conjugate model for coin flipping, examine the effect of different priors, and introduce the Normal-Normal model for inference on continuous data.

---

## 1. Bayes' Theorem for Continuous Parameters

### 1.1 The Continuous Formulation

When the parameter $\theta$ takes values in a continuous space, Bayes' theorem becomes:

$$
\boxed{p(\theta|D) = \frac{p(D|\theta) \, p(\theta)}{p(D)}}
$$

where:

| Term | Name | Description |
|------|------|-------------|
| $p(\theta\|D)$ | Posterior density | Probability density of $\theta$ given data |
| $p(D\|\theta)$ | Likelihood function | Probability of data given parameter value |
| $p(\theta)$ | Prior density | Initial belief about parameter |
| $p(D)$ | Evidence | Marginal likelihood (normalizing constant) |

### 1.2 The Evidence Integral

The evidence is computed via integration over the parameter space:

$$
p(D) = \int p(D|\theta) \, p(\theta) \, d\theta
$$

In practice, we often work with the **proportional form**:

$$
p(\theta|D) \propto p(D|\theta) \, p(\theta)
$$

and normalize afterward by ensuring $\int p(\theta|D) \, d\theta = 1$.

### 1.3 Grid Approximation

For numerical computation, we can discretize the continuous parameter space:

```python
import numpy as np

def posterior_continuous(theta_grid, prior, likelihood, normalize=True):
    """
    Compute posterior distribution on a grid.
    
    Parameters
    ----------
    theta_grid : array-like
        Grid of parameter values
    prior : array-like
        Prior density evaluated at theta_grid
    likelihood : array-like
        Likelihood evaluated at theta_grid
    
    Returns
    -------
    posterior : numpy array
        Posterior density at theta_grid
    """
    theta_grid = np.asarray(theta_grid)
    prior = np.asarray(prior)
    likelihood = np.asarray(likelihood)
    
    # Unnormalized posterior
    posterior = prior * likelihood
    
    if normalize:
        # Numerical integration (trapezoidal rule)
        evidence = np.trapz(posterior, theta_grid)
        posterior = posterior / evidence
    
    return posterior
```

---

## 2. The Beta-Binomial Model

### 2.1 Problem Setting

We wish to estimate the probability $\theta$ of heads for a coin based on observed flips. This is the canonical example of Bayesian inference with a continuous parameter.

### 2.2 The Beta Prior

The **Beta distribution** is the natural prior for a probability parameter $\theta \in [0, 1]$:

$$
p(\theta) = \text{Beta}(\theta | \alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

**Properties of Beta$(\alpha, \beta)$:**

| Property | Formula |
|----------|---------|
| Mean | $\displaystyle\frac{\alpha}{\alpha + \beta}$ |
| Mode | $\displaystyle\frac{\alpha - 1}{\alpha + \beta - 2}$ (for $\alpha, \beta > 1$) |
| Variance | $\displaystyle\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

**Interpretations of hyperparameters:**

- $\alpha = \beta = 1$: **Uniform prior** (all values equally likely)
- $\alpha = \beta = 0.5$: **Jeffreys prior** (non-informative, invariant under reparametrization)
- $\alpha = \beta > 1$: Concentrates probability near $\theta = 0.5$
- $\alpha > \beta$: Prior belief that heads is more likely
- $\alpha + \beta$: **Concentration** — larger values indicate stronger prior conviction

### 2.3 The Binomial Likelihood

Given $n$ coin flips with $k$ heads, the likelihood is:

$$
p(k | n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

As a function of $\theta$ (ignoring the constant):

$$
\mathcal{L}(\theta) \propto \theta^k (1-\theta)^{n-k}
$$

### 2.4 Conjugate Posterior

The Beta prior is **conjugate** to the Binomial likelihood, meaning the posterior is also a Beta distribution:

$$
\boxed{p(\theta | k, n) = \text{Beta}(\theta | \alpha + k, \beta + n - k)}
$$

**Conjugate update rule:**

$$
\alpha_{\text{post}} = \alpha_{\text{prior}} + k \quad \text{(add number of heads)}
$$

$$
\beta_{\text{post}} = \beta_{\text{prior}} + (n - k) \quad \text{(add number of tails)}
$$

The hyperparameters can be interpreted as **pseudo-counts**: $\alpha - 1$ prior heads and $\beta - 1$ prior tails.

### 2.5 Implementation

```python
from scipy import stats

def beta_binomial_inference(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """
    Bayesian inference for coin flip probability using Beta-Binomial model.
    
    Parameters
    ----------
    n_heads : int
        Number of observed heads
    n_tails : int
        Number of observed tails
    prior_alpha, prior_beta : float
        Parameters of Beta prior
    
    Returns
    -------
    posterior_dist : scipy.stats.beta
        Posterior Beta distribution
    """
    # Conjugate update
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    
    # Create distributions
    prior_dist = stats.beta(prior_alpha, prior_beta)
    posterior_dist = stats.beta(post_alpha, post_beta)
    
    # Statistics
    prior_mean = prior_dist.mean()
    post_mean = posterior_dist.mean()
    post_mode = (post_alpha - 1) / (post_alpha + post_beta - 2)
    
    # Maximum Likelihood Estimate
    mle = n_heads / (n_heads + n_tails)
    
    return posterior_dist
```

### 2.6 Example: 15 Heads, 5 Tails with Uniform Prior

Starting with Beta$(1, 1)$ (uniform prior):

$$
\text{Posterior} = \text{Beta}(1 + 15, 1 + 5) = \text{Beta}(16, 6)
$$

| Estimate | Value |
|----------|-------|
| MLE | $15/20 = 0.750$ |
| Posterior Mean | $16/22 \approx 0.727$ |
| Posterior Mode | $15/20 = 0.750$ |

The posterior mean is **shrunk toward 0.5** compared to the MLE due to the uniform prior.

---

## 3. Effect of Different Priors

### 3.1 Prior Sensitivity Analysis

With small sample sizes, the choice of prior significantly affects the posterior. Consider observing 7 heads in 10 flips:

| Prior | $(\alpha, \beta)$ | Posterior Mean | Diff from MLE |
|-------|-------------------|----------------|---------------|
| Uniform | $(1, 1)$ | 0.667 | 0.033 |
| Jeffreys | $(0.5, 0.5)$ | 0.682 | 0.018 |
| Weak Fair | $(2, 2)$ | 0.643 | 0.057 |
| Strong Fair | $(10, 10)$ | 0.567 | 0.133 |
| Skeptical | $(2, 8)$ | 0.450 | 0.250 |

The MLE is $7/10 = 0.70$ in all cases.

### 3.2 Prior-Data Conflict

When prior beliefs strongly conflict with observed data:

- **Small samples**: Prior dominates, posterior reflects prior beliefs
- **Large samples**: Likelihood dominates, posteriors converge regardless of prior

This is formalized by the **Bernstein-von Mises theorem**: as $n \to \infty$, the posterior concentrates around the true parameter value regardless of the prior (under regularity conditions).

### 3.3 Implementation

```python
def compare_priors(n_heads, n_tails):
    """Compare posteriors under different priors."""
    
    priors = {
        'Uniform': (1, 1),
        'Jeffreys': (0.5, 0.5),
        'Weak Fair': (2, 2),
        'Strong Fair': (10, 10),
        'Skeptical': (2, 8),
    }
    
    mle = n_heads / (n_heads + n_tails)
    
    for name, (alpha, beta) in priors.items():
        post_alpha = alpha + n_heads
        post_beta = beta + n_tails
        post_mean = post_alpha / (post_alpha + post_beta)
        print(f"{name}: posterior mean = {post_mean:.4f}")
```

---

## 4. Sequential Updating

### 4.1 Online Bayesian Learning

A key property of Bayesian inference is that **sequential updating equals batch updating**. Processing observations one at a time yields the same posterior as processing all at once.

For Beta-Binomial:

$$
\text{Beta}(\alpha, \beta) \xrightarrow{\text{observe H}} \text{Beta}(\alpha + 1, \beta) \xrightarrow{\text{observe T}} \text{Beta}(\alpha + 1, \beta + 1)
$$

This is equivalent to:

$$
\text{Beta}(\alpha, \beta) \xrightarrow{\text{observe 1H, 1T}} \text{Beta}(\alpha + 1, \beta + 1)
$$

### 4.2 Implementation

```python
def sequential_beta_binomial(flip_sequence, prior_alpha=1, prior_beta=1):
    """
    Sequential Bayesian updating with Beta-Binomial model.
    
    Parameters
    ----------
    flip_sequence : list of str
        Sequence of 'H' and 'T' observations
    prior_alpha, prior_beta : float
        Initial prior parameters
    
    Returns
    -------
    alpha_history, beta_history : lists
        Parameter evolution over updates
    """
    alpha_history = [prior_alpha]
    beta_history = [prior_beta]
    
    current_alpha = prior_alpha
    current_beta = prior_beta
    
    for flip in flip_sequence:
        if flip == 'H':
            current_alpha += 1
        else:
            current_beta += 1
        
        alpha_history.append(current_alpha)
        beta_history.append(current_beta)
    
    return alpha_history, beta_history
```

### 4.3 Convergence Behavior

As data accumulates:

1. **Posterior concentrates**: Variance decreases as $O(1/n)$
2. **Prior washes out**: Posterior mean converges to true parameter
3. **Uncertainty quantified**: Credible intervals narrow with more data

---

## 5. Normal-Normal Model

### 5.1 Problem Setting

We wish to estimate the mean $\mu$ of a Normal distribution with **known variance** $\sigma^2$, given $n$ observations $x_1, \ldots, x_n$.

### 5.2 Conjugate Structure

**Prior:** $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**Likelihood:** $x_i | \mu \sim \mathcal{N}(\mu, \sigma^2)$ independently

**Posterior:** $\mu | x_1, \ldots, x_n \sim \mathcal{N}(\mu_n, \sigma_n^2)$

### 5.3 Posterior Parameters

The posterior parameters have elegant closed-form expressions:

$$
\boxed{\mu_n = \frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar{x}}{\sigma^2 + n\sigma_0^2}}
$$

$$
\boxed{\sigma_n^2 = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}}
$$

where $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ is the sample mean.

### 5.4 Precision-Weighted Form

Defining **precision** as inverse variance ($\tau = 1/\sigma^2$):

$$
\tau_n = \tau_0 + n\tau_{\text{data}}
$$

$$
\mu_n = \frac{\tau_0 \mu_0 + n\tau_{\text{data}} \bar{x}}{\tau_n}
$$

The posterior mean is a **precision-weighted average** of prior mean and sample mean.

### 5.5 Implementation

```python
def normal_normal_inference(data, prior_mean, prior_std, known_std):
    """
    Bayesian inference for Normal mean with known variance.
    
    Parameters
    ----------
    data : array-like
        Observed data points
    prior_mean : float
        Mean of prior distribution
    prior_std : float
        Standard deviation of prior
    known_std : float
        Known standard deviation of data
    
    Returns
    -------
    posterior_dist : scipy.stats.norm
        Posterior Normal distribution
    """
    data = np.asarray(data)
    n = len(data)
    sample_mean = np.mean(data)
    
    # Precisions
    prior_precision = 1 / prior_std**2
    data_precision = n / known_std**2
    posterior_precision = prior_precision + data_precision
    
    # Posterior parameters
    posterior_mean = (prior_precision * prior_mean + 
                      data_precision * sample_mean) / posterior_precision
    posterior_std = np.sqrt(1 / posterior_precision)
    
    return stats.norm(posterior_mean, posterior_std)
```

### 5.6 Example

Given:
- Prior: $\mathcal{N}(0, 5^2)$
- Known data std: $\sigma = 2$
- Observed: $n = 20$ samples with $\bar{x} = 4.8$

Compute weights:
- Prior precision: $\tau_0 = 1/25 = 0.04$
- Data precision: $n\tau = 20/4 = 5$
- Total precision: $\tau_n = 5.04$

Posterior:
- Mean: $\mu_n = \frac{0.04 \times 0 + 5 \times 4.8}{5.04} \approx 4.76$
- Std: $\sigma_n = \sqrt{1/5.04} \approx 0.445$

With 20 observations, the data dominates the prior (weight $\approx 99.2\%$).

---

## 6. Key Takeaways

1. **Continuous parameters** require probability densities rather than discrete probabilities. The posterior is a density function over the parameter space.

2. **Conjugate priors** yield posteriors in the same distributional family, enabling closed-form updates:
   - Beta-Binomial: $\text{Beta}(\alpha, \beta) \to \text{Beta}(\alpha + k, \beta + n - k)$
   - Normal-Normal: $\mathcal{N}(\mu_0, \sigma_0^2) \to \mathcal{N}(\mu_n, \sigma_n^2)$

3. **Prior sensitivity** matters with small samples. Strong priors can dominate the posterior. With large samples, the likelihood dominates.

4. **Sequential updating** is equivalent to batch updating — a fundamental property enabling online learning.

5. **Precision weighting** provides intuition: the posterior mean is a weighted average where weights are proportional to precision (inverse variance).

---

## 7. Exercises

### Exercise 1: Strong vs Weak Priors
Compare Beta$(1,1)$ vs Beta$(100,100)$ priors when observing 5 heads in 10 flips. Quantify how much the strong prior influences the posterior.

### Exercise 2: Prior Sensitivity
For the medical test example from Module 1, reformulate it with a continuous test accuracy parameter using a Beta prior. How does this change the inference?

### Exercise 3: Convergence
Generate a long sequence of coin flips from a fair coin. Plot how the posterior mean and 95% credible interval converge to the true parameter for different priors.

### Exercise 4: Normal-Inverse-Gamma
Research the Normal-Inverse-Gamma conjugate family for the case where both mean and variance are unknown. Implement inference for this model.

### Exercise 5: Real Data Application
Find a dataset with binary outcomes (e.g., customer conversion, email click-through rates). Use the Beta-Binomial model to estimate the success rate and compute 95% credible intervals.

---

## References

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapters 2–3
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, Chapter 3
- Hoff, P. *A First Course in Bayesian Statistical Methods*
