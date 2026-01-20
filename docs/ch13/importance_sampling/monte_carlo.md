# Monte Carlo Integration

## Overview

Monte Carlo integration provides a fundamental framework for approximating integrals through random sampling. This technique forms the mathematical foundation for importance sampling and serves as a cornerstone of modern Bayesian inference and computational statistics.

## The Integration Problem

### Motivation

In Bayesian inference and statistical learning, we frequently encounter integrals of the form:

$$
I = \int_{\mathcal{X}} f(x) \, dx
$$

or more commonly, expectations under a probability distribution:

$$
\mathbb{E}_p[h(X)] = \int h(x) \, p(x) \, dx
$$

These integrals are often analytically intractable due to:

- **High dimensionality**: The curse of dimensionality renders grid-based methods infeasible
- **Complex integrands**: No closed-form antiderivative exists
- **Unknown normalizing constants**: Posterior distributions are known only up to proportionality

### Classical Numerical Integration

Traditional quadrature methods discretize the domain:

$$
\int_a^b f(x) \, dx \approx \sum_{i=1}^n w_i \, f(x_i)
$$

where $x_i$ are deterministic grid points and $w_i$ are quadrature weights.

**Limitations of Deterministic Methods:**

| Method | Grid Points | Convergence Rate | Dimension Scaling |
|--------|-------------|------------------|-------------------|
| Trapezoidal | $n$ | $O(n^{-2})$ | $O(n^d)$ points needed |
| Simpson's | $n$ | $O(n^{-4})$ | $O(n^d)$ points needed |
| Gaussian Quadrature | $n$ | $O(n^{-2k})$ | $O(n^d)$ points needed |

For a $d$-dimensional problem with $n$ points per dimension, the total computation scales as $O(n^d)$—the **curse of dimensionality**.

## Monte Carlo Estimation

### The Basic Principle

Monte Carlo estimation leverages the Law of Large Numbers. If $X_1, X_2, \ldots, X_n \stackrel{\text{i.i.d.}}{\sim} p(x)$, then:

$$
\hat{I}_{\text{MC}} = \frac{1}{n} \sum_{i=1}^n h(X_i) \xrightarrow{a.s.} \mathbb{E}_p[h(X)] = I
$$

### Derivation from First Principles

Starting with the expectation:

$$
I = \mathbb{E}_p[h(X)] = \int h(x) \, p(x) \, dx
$$

The Monte Carlo estimator is:

$$
\hat{I}_{\text{MC}} = \frac{1}{n} \sum_{i=1}^n h(X_i), \quad X_i \sim p(x)
$$

**Unbiasedness:**

$$
\mathbb{E}[\hat{I}_{\text{MC}}] = \mathbb{E}\left[\frac{1}{n} \sum_{i=1}^n h(X_i)\right] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[h(X_i)] = \frac{1}{n} \cdot n \cdot I = I
$$

**Variance:**

$$
\text{Var}(\hat{I}_{\text{MC}}) = \text{Var}\left(\frac{1}{n} \sum_{i=1}^n h(X_i)\right) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(h(X_i)) = \frac{\sigma^2_h}{n}
$$

where $\sigma^2_h = \text{Var}_p(h(X)) = \mathbb{E}_p[h^2(X)] - I^2$.

### Central Limit Theorem

For sufficiently large $n$, the Central Limit Theorem provides:

$$
\sqrt{n}(\hat{I}_{\text{MC}} - I) \xrightarrow{d} \mathcal{N}(0, \sigma^2_h)
$$

This yields the standard error:

$$
\text{SE}(\hat{I}_{\text{MC}}) = \frac{\sigma_h}{\sqrt{n}}
$$

**Key insight**: The convergence rate is $O(n^{-1/2})$, **independent of dimension**.

## Dimension-Independent Convergence

### The Monte Carlo Advantage

| Method | Convergence Rate | 10D Problem | 100D Problem |
|--------|------------------|-------------|--------------|
| Grid (Simpson) | $O(n^{-4/d})$ | $O(n^{-0.4})$ | $O(n^{-0.04})$ |
| Monte Carlo | $O(n^{-1/2})$ | $O(n^{-0.5})$ | $O(n^{-0.5})$ |

Monte Carlo maintains its $O(n^{-1/2})$ convergence regardless of dimensionality, making it the method of choice for high-dimensional problems.

### Comparison Example

To achieve precision $\epsilon$ in $d$ dimensions:

**Grid Methods:**
$$
n_{\text{grid}} \propto \epsilon^{-d/k}
$$
where $k$ is the method order (e.g., $k=2$ for trapezoidal).

**Monte Carlo:**
$$
n_{\text{MC}} \propto \epsilon^{-2}
$$

For $d = 20$ and $k = 2$: grid needs $\epsilon^{-10}$ points while MC needs only $\epsilon^{-2}$.

## Estimating the Variance

Since $\sigma^2_h = \text{Var}_p(h(X))$ is typically unknown, we estimate it:

$$
\hat{\sigma}^2_h = \frac{1}{n-1} \sum_{i=1}^n (h(X_i) - \hat{I}_{\text{MC}})^2
$$

This yields the estimated standard error:

$$
\widehat{\text{SE}} = \frac{\hat{\sigma}_h}{\sqrt{n}}
$$

### Confidence Intervals

An approximate $(1-\alpha)$ confidence interval for $I$:

$$
\hat{I}_{\text{MC}} \pm z_{1-\alpha/2} \cdot \frac{\hat{\sigma}_h}{\sqrt{n}}
$$

For 95% confidence ($\alpha = 0.05$): $z_{0.975} \approx 1.96$.

## PyTorch Implementation

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def monte_carlo_estimate(h_function, sampler, n_samples, return_diagnostics=False):
    """
    Basic Monte Carlo integration.
    
    Parameters
    ----------
    h_function : callable
        Function h(x) to integrate
    sampler : callable
        Function that returns n samples from distribution p
    n_samples : int
        Number of samples
    return_diagnostics : bool
        Whether to return diagnostic information
        
    Returns
    -------
    estimate : torch.Tensor
        Monte Carlo estimate of E_p[h(X)]
    se : torch.Tensor
        Estimated standard error
    diagnostics : dict, optional
        Additional diagnostic information
    
    Mathematical Foundation
    -----------------------
    Î = (1/n) Σᵢ h(Xᵢ),  Xᵢ ~ p(x)
    
    SE(Î) = σ_h / √n, where σ²_h = Var_p(h(X))
    """
    # Draw samples from the distribution
    samples = sampler(n_samples)
    
    # Evaluate function at sample points
    h_values = h_function(samples)
    
    # Monte Carlo estimate: (1/n) Σ h(xᵢ)
    estimate = torch.mean(h_values)
    
    # Estimate variance: (1/(n-1)) Σ (h(xᵢ) - Î)²
    variance = torch.var(h_values, unbiased=True)
    
    # Standard error: σ̂_h / √n
    se = torch.sqrt(variance / n_samples)
    
    if return_diagnostics:
        diagnostics = {
            'samples': samples,
            'h_values': h_values,
            'variance': variance,
            'n_samples': n_samples
        }
        return estimate, se, diagnostics
    
    return estimate, se


def convergence_analysis(h_function, sampler, true_value, 
                        sample_sizes, n_replications=100):
    """
    Analyze Monte Carlo convergence behavior.
    
    Verifies that:
    1. Estimate converges to true value (LLN)
    2. Standard error scales as O(1/√n)
    3. Distribution approaches normal (CLT)
    """
    results = {
        'sample_sizes': sample_sizes,
        'mean_estimates': [],
        'std_estimates': [],
        'theoretical_se': []
    }
    
    # Get variance estimate from large sample
    _, _, diag = monte_carlo_estimate(
        h_function, sampler, 10000, return_diagnostics=True
    )
    sigma_h = torch.sqrt(diag['variance'])
    
    for n in sample_sizes:
        estimates = []
        for _ in range(n_replications):
            est, _ = monte_carlo_estimate(h_function, sampler, n)
            estimates.append(est.item())
        
        estimates = torch.tensor(estimates)
        results['mean_estimates'].append(torch.mean(estimates).item())
        results['std_estimates'].append(torch.std(estimates).item())
        results['theoretical_se'].append((sigma_h / torch.sqrt(torch.tensor(n))).item())
    
    return results


# Example: E[X²] where X ~ N(0, 1)
# True value: E[X²] = Var(X) + E[X]² = 1 + 0 = 1

# Define distribution and function
normal_dist = dist.Normal(0, 1)
h = lambda x: x**2
sampler = lambda n: normal_dist.sample((n,))

# Single estimate with diagnostics
estimate, se, diagnostics = monte_carlo_estimate(
    h, sampler, n_samples=10000, return_diagnostics=True
)

print("Monte Carlo Estimation: E[X²] where X ~ N(0,1)")
print("=" * 50)
print(f"True value: 1.000000")
print(f"MC estimate: {estimate.item():.6f}")
print(f"Standard error: {se.item():.6f}")
print(f"95% CI: [{estimate.item() - 1.96*se.item():.6f}, "
      f"{estimate.item() + 1.96*se.item():.6f}]")

# Convergence analysis
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
results = convergence_analysis(h, sampler, true_value=1.0, 
                              sample_sizes=sample_sizes)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Convergence of estimate
ax = axes[0]
ax.plot(results['sample_sizes'], results['mean_estimates'], 
        'bo-', linewidth=2, markersize=8, label='Mean MC Estimate')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, 
           label='True Value')
ax.fill_between(results['sample_sizes'],
               [m - 2*s for m, s in zip(results['mean_estimates'], 
                                        results['std_estimates'])],
               [m + 2*s for m, s in zip(results['mean_estimates'], 
                                        results['std_estimates'])],
               alpha=0.3, color='blue', label='±2 Std Dev')
ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Estimate', fontsize=12)
ax.set_title('Monte Carlo Convergence', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Standard error scaling
ax = axes[1]
ax.plot(results['sample_sizes'], results['std_estimates'], 
        'go-', linewidth=2, markersize=8, label='Empirical Std Dev')
ax.plot(results['sample_sizes'], results['theoretical_se'], 
        'r--', linewidth=2, label='Theoretical SE = σ/√n')
ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Standard Error', fontsize=12)
ax.set_title('O(1/√n) Convergence Rate', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Limitations of Naive Monte Carlo

### When Sampling from $p$ is Difficult

The basic Monte Carlo method requires sampling from $p(x)$. This becomes problematic when:

1. **Posterior distributions**: $p(\theta|y) = \frac{p(y|\theta)p(\theta)}{p(y)}$ where $p(y)$ is intractable
2. **Complex distributions**: No standard sampling algorithm exists
3. **Rare events**: Events of interest have very low probability under $p$

### When Variance is High

Even when sampling is possible, naive MC can be inefficient:

**Example: Tail Probability**

Estimate $\mathbb{P}(X > 4)$ where $X \sim \mathcal{N}(0, 1)$.

True value: $\Phi(-4) \approx 3.2 \times 10^{-5}$

Using indicator function $h(x) = \mathbb{1}(x > 4)$:
- Variance: $\sigma^2_h = p(1-p) \approx 3.2 \times 10^{-5}$
- Standard error with $n = 10^6$: SE $\approx 5.7 \times 10^{-6}$
- Coefficient of variation: CV $= \frac{\text{SE}}{p} \approx 18\%$

To achieve CV $= 1\%$, we need $n \approx 3 \times 10^{10}$ samples!

### The Solution: Importance Sampling

These limitations motivate importance sampling:

1. **Sample from an alternative distribution** $q(x)$ that is easy to sample from
2. **Correct for the distribution change** using importance weights
3. **Reduce variance** by choosing $q$ strategically

This is the topic of the next section.

## Key Takeaways

!!! success "Monte Carlo Strengths"
    - **Dimension-independent convergence**: $O(n^{-1/2})$ regardless of $d$
    - **Unbiased estimation**: $\mathbb{E}[\hat{I}] = I$
    - **Easy uncertainty quantification**: CLT provides confidence intervals
    - **Trivially parallelizable**: Samples are i.i.d.

!!! warning "Monte Carlo Limitations"
    - **Requires sampling from target**: Not always possible
    - **Can have high variance**: Especially for rare events or heavy tails
    - **$O(n^{-1/2})$ convergence is slow**: Need 100× samples for 10× precision

!!! info "Connection to Importance Sampling"
    Importance sampling addresses these limitations by:
    
    - Allowing integration under distributions we can't sample from
    - Potentially reducing variance through strategic proposal choice
    - Enabling estimation of normalizing constants

## Exercises

### Exercise 1: Convergence Verification
Verify the $O(n^{-1/2})$ convergence rate empirically. Estimate $\mathbb{E}[e^X]$ where $X \sim \mathcal{N}(0, 1)$ (true value: $e^{1/2} \approx 1.6487$) for sample sizes $n = 100, 400, 1600, 6400$. Confirm that the standard error decreases by half when sample size quadruples.

### Exercise 2: High-Dimensional Integration
Estimate the volume of a unit hypersphere in $d = 10$ dimensions using Monte Carlo. Compare with the analytical formula $V_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}$. Discuss why grid-based methods would be impractical.

### Exercise 3: Variance Decomposition
For $X \sim \mathcal{N}(\mu, \sigma^2)$ and $h(X) = X^2$:

a) Derive $\text{Var}_p(h(X))$ analytically
b) Verify your result with Monte Carlo
c) How does the variance change with $\mu$ and $\sigma$?

## References

1. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. Chapter 3: Monte Carlo Integration.

2. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. Available at: https://artowen.su.domains/mc/

3. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. Chapter 2: Monte Carlo Quadrature.

4. Caflisch, R. E. (1998). "Monte Carlo and quasi-Monte Carlo methods." *Acta Numerica*, 7, 1-49.
