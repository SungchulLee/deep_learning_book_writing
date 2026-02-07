# Variational Inference Framework

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the computational challenges of exact Bayesian inference
2. Formulate variational inference as an optimization problem
3. Explain the relationship between VI and KL divergence
4. Compare variational inference with other approximate inference methods
5. Implement basic VI for simple probabilistic models

## The Challenge of Bayesian Inference

Bayesian inference seeks to compute the posterior distribution over parameters given observed data. For a model with parameters $\theta$ and observed data $\mathcal{D}$, Bayes' theorem gives:

$$
p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}
$$

where:

- $p(\theta | \mathcal{D})$ is the **posterior distribution** (what we want)
- $p(\mathcal{D} | \theta)$ is the **likelihood** (probability of data given parameters)
- $p(\theta)$ is the **prior distribution** (our initial beliefs)
- $p(\mathcal{D})$ is the **marginal likelihood** or **evidence**

The fundamental computational challenge lies in the marginal likelihood:

$$
p(\mathcal{D}) = \int p(\mathcal{D} | \theta) p(\theta) \, d\theta
$$

### Why Is This Integral Intractable?

The marginal likelihood integral becomes computationally intractable when:

1. **High dimensionality**: The parameter space $\theta$ may have hundreds or millions of dimensions
2. **Complex likelihood functions**: Non-conjugate models lack closed-form solutions
3. **Latent variable models**: Additional integration over latent variables compounds difficulty
4. **Multimodality**: The integrand may have complex structure

!!! example "Mixture Models: A Classic Intractable Case"
    Consider a Gaussian Mixture Model with $K$ components. Even this relatively simple model has:
    
    - $K$ mixture weights $\pi_1, \ldots, \pi_K$
    - $K$ mean vectors $\mu_1, \ldots, \mu_K$
    - $K$ covariance matrices $\Sigma_1, \ldots, \Sigma_K$
    - $N$ latent cluster assignments $z_1, \ldots, z_N$
    
    The posterior over all these quantities has no closed-form solution.

## Variational Inference: Optimization Instead of Integration

Variational inference (VI) reframes the intractable integration problem as a tractable optimization problem. The key insight is:

!!! info "The VI Principle"
    Instead of computing the exact posterior $p(\theta | \mathcal{D})$, we:
    
    1. Choose a family of simpler distributions $\mathcal{Q} = \{q(\theta)\}$
    2. Find the member $q^*(\theta) \in \mathcal{Q}$ that is "closest" to $p(\theta | \mathcal{D})$
    3. Use $q^*(\theta)$ as an approximation to the true posterior

### The Variational Family

The **variational family** $\mathcal{Q}$ is a set of distributions over the latent variables that we optimize over. Common choices include:

| Family | Description | Flexibility |
|--------|-------------|-------------|
| Mean-field | Fully factorized: $q(\theta) = \prod_j q_j(\theta_j)$ | Low |
| Structured | Partial factorization respecting some dependencies | Medium |
| Normalizing flows | Transformed base distribution | High |
| Neural networks | Amortized inference with encoders | High |

### Measuring "Closeness": KL Divergence

We measure the "distance" between distributions using the **Kullback-Leibler (KL) divergence**:

$$
\text{KL}(q(\theta) \| p(\theta | \mathcal{D})) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | \mathcal{D})} \, d\theta
$$

The KL divergence has important properties:

- **Non-negative**: $\text{KL}(q \| p) \geq 0$
- **Zero iff equal**: $\text{KL}(q \| p) = 0 \Leftrightarrow q = p$ almost everywhere
- **Asymmetric**: $\text{KL}(q \| p) \neq \text{KL}(p \| q)$

### The Variational Objective

The VI objective is to minimize the KL divergence:

$$
q^*(\theta) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\theta) \| p(\theta | \mathcal{D}))
$$

However, this objective still contains the intractable posterior! The key mathematical trick of VI is to rewrite this in a computable form.

## Mathematical Framework

### From KL Divergence to ELBO

Starting from the KL divergence definition:

$$
\begin{aligned}
\text{KL}(q \| p) &= \mathbb{E}_q\left[\log \frac{q(\theta)}{p(\theta | \mathcal{D})}\right] \\
&= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\theta | \mathcal{D})]
\end{aligned}
$$

Applying Bayes' rule to the posterior:

$$
\log p(\theta | \mathcal{D}) = \log p(\mathcal{D} | \theta) + \log p(\theta) - \log p(\mathcal{D})
$$

Substituting:

$$
\begin{aligned}
\text{KL}(q \| p) &= \mathbb{E}_q[\log q(\theta)] - \mathbb{E}_q[\log p(\mathcal{D} | \theta)] - \mathbb{E}_q[\log p(\theta)] + \log p(\mathcal{D}) \\
&= -\underbrace{\left(\mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]\right)}_{\text{ELBO}(q)} + \log p(\mathcal{D})
\end{aligned}
$$

Rearranging gives the fundamental identity:

$$
\boxed{\log p(\mathcal{D}) = \text{ELBO}(q) + \text{KL}(q \| p)}
$$

### The Evidence Lower Bound

Since KL divergence is non-negative:

$$
\log p(\mathcal{D}) \geq \text{ELBO}(q)
$$

This is why we call it the **Evidence Lower Bound (ELBO)**. The ELBO is defined as:

$$
\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \mathbb{E}_q[\log q(\theta)]
$$

!!! success "Key Insight"
    **Maximizing ELBO is equivalent to minimizing KL divergence** since:
    
    $$\max_q \text{ELBO}(q) \Leftrightarrow \min_q \text{KL}(q \| p)$$
    
    And crucially, **ELBO can be computed without knowing $p(\mathcal{D})$**!

## Forward vs Reverse KL Divergence

The choice of KL divergence direction has important implications for the approximation behavior.

### Forward KL: $\text{KL}(q \| p)$ (Used in Standard VI)

$$
\text{KL}(q \| p) = \int q(\theta) \log \frac{q(\theta)}{p(\theta | \mathcal{D})} \, d\theta
$$

**Properties:**

- **Mean-seeking**: $q$ tends to cover all modes of $p$
- **Zero-avoiding**: $q(\theta) > 0$ wherever $p(\theta | \mathcal{D}) > 0$
- **May overestimate uncertainty**: $q$ spreads mass to cover $p$

### Reverse KL: $\text{KL}(p \| q)$

$$
\text{KL}(p \| q) = \int p(\theta | \mathcal{D}) \log \frac{p(\theta | \mathcal{D})}{q(\theta)} \, d\theta
$$

**Properties:**

- **Mode-seeking**: $q$ tends to focus on one mode of $p$
- **Zero-forcing**: Forces $q(\theta) = 0$ where $p(\theta | \mathcal{D}) \approx 0$
- **May underestimate uncertainty**: $q$ concentrates on dominant mode

### Visual Comparison

```
True Posterior (Bimodal):           Forward KL Result:        Reverse KL Result:
       ╱╲    ╱╲                          ╱────╲                      ╱╲
      ╱  ╲  ╱  ╲                        ╱      ╲                    ╱  ╲
     ╱    ╲╱    ╲                      ╱        ╲                  ╱    ╲
    ╱            ╲                    ╱          ╲                ╱      ╲
───╱              ╲───            ───╱            ╲───          ─╱        ╲────
   Mode 1   Mode 2                 Covers both modes           Focuses on one
```

## PyTorch Implementation

### Simple VI for Gaussian Mean Estimation

```python
import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt

class SimpleVI:
    """
    Variational inference for estimating the mean of a Gaussian
    with known variance.
    
    Model:
        Prior: θ ~ N(μ₀, σ₀²)
        Likelihood: x_i | θ ~ N(θ, σ²)
        
    Variational family:
        q(θ) = N(m, s²)  where m, s are variational parameters
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_std: float = 1.0,
                 likelihood_std: float = 1.0):
        self.mu_0 = prior_mean
        self.sigma_0 = prior_std
        self.sigma = likelihood_std
        
        # Variational parameters (to be optimized)
        self.m = torch.tensor([0.0], requires_grad=True)
        self.log_s = torch.tensor([0.0], requires_grad=True)
    
    @property
    def s(self):
        """Ensure positive standard deviation."""
        return torch.exp(self.log_s)
    
    def compute_elbo(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the Evidence Lower Bound.
        
        ELBO = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]
             = E_q[log p(D|θ)] - KL(q(θ) || p(θ))
        
        For Gaussian q and p, we have closed-form expressions.
        """
        n = len(data)
        
        # E_q[log p(D|θ)] - Expected log-likelihood
        # For Gaussian likelihood with known variance:
        # E_q[log p(x|θ)] = -n/2 log(2πσ²) - 1/(2σ²) Σᵢ E_q[(xᵢ - θ)²]
        #                 = -n/2 log(2πσ²) - 1/(2σ²) [Σᵢ(xᵢ - m)² + n·s²]
        
        expected_log_likelihood = (
            -0.5 * n * torch.log(2 * torch.pi * self.sigma**2)
            - 0.5 / self.sigma**2 * (
                torch.sum((data - self.m)**2) + n * self.s**2
            )
        )
        
        # KL(q(θ) || p(θ)) - KL divergence from prior
        # For Gaussians: KL(N(m,s²) || N(μ₀,σ₀²))
        #              = log(σ₀/s) + (s² + (m-μ₀)²)/(2σ₀²) - 1/2
        
        kl_divergence = (
            torch.log(self.sigma_0 / self.s)
            + (self.s**2 + (self.m - self.mu_0)**2) / (2 * self.sigma_0**2)
            - 0.5
        )
        
        elbo = expected_log_likelihood - kl_divergence
        return elbo
    
    def fit(self, data: torch.Tensor, n_iterations: int = 1000,
            learning_rate: float = 0.01, verbose: bool = True):
        """
        Optimize variational parameters using gradient ascent on ELBO.
        """
        optimizer = torch.optim.Adam([self.m, self.log_s], lr=learning_rate)
        
        history = {'elbo': [], 'm': [], 's': []}
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            elbo = self.compute_elbo(data)
            loss = -elbo  # Minimize negative ELBO = maximize ELBO
            
            loss.backward()
            optimizer.step()
            
            # Record history
            history['elbo'].append(elbo.item())
            history['m'].append(self.m.item())
            history['s'].append(self.s.item())
            
            if verbose and (i + 1) % 200 == 0:
                print(f"Iter {i+1:4d}: ELBO = {elbo.item():.4f}, "
                      f"m = {self.m.item():.4f}, s = {self.s.item():.4f}")
        
        return history
    
    def get_posterior(self):
        """Return the approximate posterior distribution."""
        return dist.Normal(self.m.detach(), self.s.detach())
    
    def exact_posterior(self, data: torch.Tensor):
        """
        Compute the exact posterior (available for this conjugate model).
        
        Posterior: θ | D ~ N(μₙ, σₙ²)
        where:
            precision_n = 1/σ₀² + n/σ²
            μₙ = (μ₀/σ₀² + Σxᵢ/σ²) / precision_n
            σₙ² = 1 / precision_n
        """
        n = len(data)
        precision_0 = 1 / self.sigma_0**2
        precision_data = n / self.sigma**2
        precision_n = precision_0 + precision_data
        
        mu_n = (self.mu_0 * precision_0 + data.sum() * precision_data / n) / precision_n
        sigma_n = 1 / torch.sqrt(torch.tensor(precision_n))
        
        return dist.Normal(mu_n, sigma_n)


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate synthetic data
    true_mean = 2.5
    n_samples = 50
    data = torch.randn(n_samples) + true_mean
    
    print("=" * 60)
    print("Variational Inference for Gaussian Mean")
    print("=" * 60)
    print(f"\nTrue mean: {true_mean}")
    print(f"Sample mean: {data.mean().item():.4f}")
    print(f"Sample size: {n_samples}")
    
    # Fit VI model
    vi = SimpleVI(prior_mean=0.0, prior_std=2.0, likelihood_std=1.0)
    history = vi.fit(data, n_iterations=1000)
    
    # Compare with exact posterior
    exact = vi.exact_posterior(data)
    approx = vi.get_posterior()
    
    print(f"\nExact posterior:  N({exact.mean.item():.4f}, {exact.stddev.item():.4f}²)")
    print(f"VI approximation: N({approx.mean.item():.4f}, {approx.stddev.item():.4f}²)")
    print(f"\nDifference in mean: {abs(exact.mean.item() - approx.mean.item()):.6f}")
    print(f"Difference in std:  {abs(exact.stddev.item() - approx.stddev.item()):.6f}")
```

### Visualization

```python
def visualize_vi_results(vi, data, history):
    """Visualize VI optimization and results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: ELBO convergence
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Parameter trajectories
    ax = axes[0, 1]
    ax.plot(history['m'], history['s'], 'b-', alpha=0.5, linewidth=1)
    ax.plot(history['m'][0], history['s'][0], 'go', markersize=10, label='Start')
    ax.plot(history['m'][-1], history['s'][-1], 'ro', markersize=10, label='End')
    
    exact = vi.exact_posterior(data)
    ax.plot(exact.mean.item(), exact.stddev.item(), 'k*', 
            markersize=15, label='Exact')
    
    ax.set_xlabel('Mean (m)', fontsize=11)
    ax.set_ylabel('Std Dev (s)', fontsize=11)
    ax.set_title('(b) Parameter Trajectory', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Posterior comparison
    ax = axes[1, 0]
    theta_range = torch.linspace(-1, 5, 500)
    
    exact_pdf = torch.exp(exact.log_prob(theta_range))
    approx = vi.get_posterior()
    approx_pdf = torch.exp(approx.log_prob(theta_range))
    
    ax.plot(theta_range.numpy(), exact_pdf.numpy(), 'b-', 
            linewidth=2.5, label='Exact Posterior')
    ax.plot(theta_range.numpy(), approx_pdf.numpy(), 'r--', 
            linewidth=2.5, label='VI Approximation')
    ax.axvline(data.mean().item(), color='green', linestyle=':', 
               linewidth=2, label='Sample Mean')
    
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(c) Posterior Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Data histogram
    ax = axes[1, 1]
    ax.hist(data.numpy(), bins=15, density=True, alpha=0.6, 
            color='gray', edgecolor='black', label='Data')
    ax.axvline(approx.mean.item(), color='red', linestyle='--', 
               linewidth=2, label='Posterior Mean')
    ax.set_xlabel('Data Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Data Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vi_framework_example.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## VI vs Other Inference Methods

### Comparison with MCMC

| Aspect | Variational Inference | MCMC |
|--------|----------------------|------|
| **Output** | Approximate distribution $q(\theta)$ | Samples from $p(\theta\|\mathcal{D})$ |
| **Accuracy** | Biased but fast | Asymptotically exact |
| **Speed** | Fast (optimization) | Slow (sampling) |
| **Scalability** | Scales well to large datasets | Limited by serial sampling |
| **Uncertainty** | May underestimate | Accurate in limit |
| **Convergence** | Easy to diagnose (ELBO) | Complex diagnostics |
| **Parallelization** | Naturally parallel | Difficult |

### When to Use VI

**Choose VI when:**

- Dataset is large (millions of observations)
- Speed is critical (real-time applications)
- Approximate posterior is sufficient
- Model has many latent variables
- Need to scale to complex models

**Choose MCMC when:**

- Accurate uncertainty quantification is critical
- Dataset is small to moderate
- Gold-standard inference is required
- Need to detect multimodality

## Advantages and Limitations

### Advantages of VI

1. **Scalability**: Works with stochastic optimization for large datasets
2. **Speed**: Much faster than MCMC for many problems
3. **Determinism**: No sampling noise in the approximation
4. **Model comparison**: ELBO provides lower bound on log evidence
5. **Convergence monitoring**: ELBO provides clear optimization target

### Limitations of VI

1. **Approximation bias**: $q(\theta) \neq p(\theta | \mathcal{D})$ in general
2. **Family restriction**: Quality limited by variational family choice
3. **Uncertainty underestimation**: Particularly with mean-field assumption
4. **Local optima**: Optimization may not find global optimum
5. **Mode covering vs mode seeking**: Forward KL may miss modes

## Summary

Variational inference transforms intractable Bayesian inference into tractable optimization:

$$
\text{Intractable: } p(\theta | \mathcal{D}) = \frac{p(\mathcal{D}|\theta)p(\theta)}{\int p(\mathcal{D}|\theta)p(\theta)d\theta}
$$

$$
\text{Tractable: } q^*(\theta) = \arg\max_{q \in \mathcal{Q}} \text{ELBO}(q)
$$

**Key relationships:**

- $\log p(\mathcal{D}) = \text{ELBO}(q) + \text{KL}(q \| p)$
- $\max_q \text{ELBO}(q) \Leftrightarrow \min_q \text{KL}(q \| p)$
- ELBO can be computed without knowing $p(\mathcal{D})$

## Exercises

### Exercise 1: KL Divergence Properties

Prove that KL divergence is non-negative using Jensen's inequality.

### Exercise 2: ELBO for Bernoulli Model

Derive the ELBO for a Beta-Bernoulli model with:
- Prior: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$
- Likelihood: $x_i | \theta \sim \text{Bernoulli}(\theta)$

### Exercise 3: Compare Forward and Reverse KL

Implement both forward and reverse KL optimization for approximating a mixture of two Gaussians. Visualize the different behaviors.

## References

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians." *Journal of the American Statistical Association*.

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 10.

3. Murphy, K. P. (2022). *Probabilistic Machine Learning: Advanced Topics*, Chapters on VI.

4. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). "An Introduction to Variational Methods for Graphical Models." *Machine Learning*.

5. Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). "Stochastic Variational Inference." *Journal of Machine Learning Research*.
