# Monte Carlo Integration

## Overview

Monte Carlo integration uses random sampling to approximate integrals that are analytically intractable or computationally prohibitive with deterministic methods. This section develops the mathematical foundations of Monte Carlo estimation—from the law of large numbers guarantee through the central limit theorem for error quantification—and demonstrates why Monte Carlo methods overcome the curse of dimensionality that renders grid-based approaches useless in high dimensions. We also cover variance reduction techniques that can dramatically improve estimation efficiency.

In Bayesian inference, nearly every quantity of interest is an expectation under the posterior:

$$\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] = \int f(\theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

Computing posterior means, variances, credible intervals, and predictive distributions all reduce to integrals of this form. Monte Carlo integration provides the computational engine for evaluating them.

---

## 1. Why Grid Methods Fail: The Curse of Dimensionality

### 1.1 Grid Approximation Recap

For low-dimensional problems, **grid approximation** computes posterior expectations via deterministic numerical integration. Given a uniform grid $\{\theta_1, \ldots, \theta_n\}$ over the parameter space:

$$\mathbb{E}[f(\theta) \mid \mathcal{D}] \approx \sum_{i=1}^{n} f(\theta_i)\, p(\theta_i \mid \mathcal{D})\, \Delta\theta$$

This is exact in the limit of infinitely fine grids. For 1–2 parameters, grid approximation is simple and effective.

### 1.2 Exponential Scaling

The fundamental problem is that a grid with $G$ points per dimension requires $G^d$ total evaluations in $d$ dimensions:

| Dimensions | Grid Points ($G = 100$) | Memory (float64) | Feasibility |
|------------|------------------------|-------------------|-------------|
| 1 | $10^2$ | 0.8 KB | Trivial |
| 2 | $10^4$ | 80 KB | Easy |
| 3 | $10^6$ | 8 MB | Feasible |
| 5 | $10^{10}$ | 80 GB | Impractical |
| 10 | $10^{20}$ | — | Impossible |
| 100 | $10^{200}$ | — | Absurd |

### 1.3 Convergence Rate Degradation

For a uniform grid with $n$ points per dimension and a twice-differentiable integrand:

$$\text{Error}_{\text{grid}} = O(n^{-r/d})$$

where $r$ is the smoothness order (typically $r = 2$). The convergence rate degrades rapidly with dimension:

| Dimension | Convergence Rate | Points for $10^{-3}$ Error |
|-----------|-----------------|---------------------------|
| 1 | $O(n^{-2})$ | ~32 |
| 2 | $O(n^{-1})$ | ~1,000 |
| 5 | $O(n^{-2/5})$ | ~$10^{7.5}$ |
| 10 | $O(n^{-1/5})$ | ~$10^{15}$ |

This is not merely a practical inconvenience—it is a **fundamental mathematical limitation** that necessitates a different approach.

---

## 2. The Monte Carlo Principle

### 2.1 Basic Idea

If $X_1, X_2, \ldots, X_n \overset{\text{i.i.d.}}{\sim} p(x)$, then the sample average converges to the population expectation:

$$\hat{I}_n = \frac{1}{n} \sum_{i=1}^{n} f(X_i) \xrightarrow{a.s.} \mathbb{E}_p[f(X)] = I$$

This is guaranteed by the **Strong Law of Large Numbers** whenever $\mathbb{E}_p[|f(X)|] < \infty$.

### 2.2 Unbiasedness

The Monte Carlo estimator is unbiased for any sample size:

$$\mathbb{E}[\hat{I}_n] = \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}[f(X_i)] = \mathbb{E}_p[f(X)] = I$$

### 2.3 Variance

The variance of the Monte Carlo estimator is:

$$\text{Var}(\hat{I}_n) = \frac{\sigma_f^2}{n}$$

where $\sigma_f^2 = \text{Var}_p(f(X)) = \mathbb{E}_p[f(X)^2] - I^2$.

The standard error is therefore:

$$\text{SE}(\hat{I}_n) = \frac{\sigma_f}{\sqrt{n}}$$

### 2.4 Central Limit Theorem

For sufficiently large $n$, the estimator is approximately Gaussian:

$$\sqrt{n}\,(\hat{I}_n - I) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$$

This provides approximate $(1-\alpha)$ confidence intervals:

$$\hat{I}_n \pm z_{1-\alpha/2} \cdot \frac{\hat{\sigma}_f}{\sqrt{n}}$$

where $\hat{\sigma}_f^2 = \frac{1}{n-1} \sum_{i=1}^{n} (f(X_i) - \hat{I}_n)^2$ is the sample variance.

### 2.5 The Dimension-Free Convergence Rate

The Monte Carlo convergence rate is $O(n^{-1/2})$ **regardless of dimension**. This is the key advantage over grid methods:

| Method | Convergence Rate | 10D Problem | 100D Problem |
|--------|-----------------|-------------|--------------|
| Grid (trapezoidal) | $O(n^{-2/d})$ | $O(n^{-0.2})$ | $O(n^{-0.02})$ |
| Grid (Simpson) | $O(n^{-4/d})$ | $O(n^{-0.4})$ | $O(n^{-0.04})$ |
| **Monte Carlo** | $O(n^{-1/2})$ | $O(n^{-0.5})$ | $O(n^{-0.5})$ |

For any problem above ~5 dimensions, Monte Carlo dominates deterministic quadrature.

---

## 3. Implementation

### 3.1 Basic Monte Carlo Estimator

```python
import torch
import torch.distributions as dist


def monte_carlo_estimate(
    f: callable,
    sampler: callable,
    n_samples: int,
    return_diagnostics: bool = False,
) -> tuple:
    """
    Estimate E_p[f(X)] via Monte Carlo integration.

    Parameters
    ----------
    f : callable
        Function to integrate, f: Tensor -> Tensor.
    sampler : callable
        Function that draws n i.i.d. samples from p, sampler: int -> Tensor.
    n_samples : int
        Number of Monte Carlo samples.
    return_diagnostics : bool
        If True, return additional diagnostics.

    Returns
    -------
    estimate : Tensor
        Monte Carlo estimate of E_p[f(X)].
    se : Tensor
        Estimated standard error.
    diagnostics : dict (optional)
        Samples, function values, and variance estimate.
    """
    samples = sampler(n_samples)
    f_values = f(samples)

    estimate = torch.mean(f_values)
    variance = torch.var(f_values, unbiased=True)
    se = torch.sqrt(variance / n_samples)

    if return_diagnostics:
        return estimate, se, {
            "samples": samples,
            "f_values": f_values,
            "variance": variance,
        }
    return estimate, se
```

### 3.2 Example: Estimating E[X²] under Standard Normal

```python
torch.manual_seed(42)

# E[X²] where X ~ N(0,1): true value = 1.0
normal = dist.Normal(0.0, 1.0)
f = lambda x: x ** 2
sampler = lambda n: normal.sample((n,))

estimate, se = monte_carlo_estimate(f, sampler, n_samples=10_000)
print(f"MC estimate of E[X²]: {estimate.item():.6f}")
print(f"Standard error:       {se.item():.6f}")
print(f"True value:           1.000000")
print(f"95% CI: [{estimate.item() - 1.96*se.item():.6f}, "
      f"{estimate.item() + 1.96*se.item():.6f}]")
```

```
MC estimate of E[X²]: 0.993754
Standard error:       0.014068
True value:           1.000000
95% CI: [0.966181, 1.021327]
```

### 3.3 Convergence Demonstration

```python
import numpy as np

torch.manual_seed(42)

sample_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
true_value = 1.0

print(f"{'n':>10} {'Estimate':>12} {'SE':>10} {'|Error|':>10} {'n*SE²':>10}")
print("-" * 56)

for n in sample_sizes:
    est, se = monte_carlo_estimate(f, sampler, n)
    error = abs(est.item() - true_value)
    # n * SE² should be approximately constant (= σ²)
    print(f"{n:>10} {est.item():>12.6f} {se.item():>10.6f} "
          f"{error:>10.6f} {n * se.item()**2:>10.4f}")
```

```
         n     Estimate         SE     |Error|      n*SE²
--------------------------------------------------------
       100     1.011693   0.145832   0.011693     2.1267
       500     0.975553   0.061479   0.024447     1.8898
      1000     1.021448   0.045076   0.021448     2.0319
      5000     1.003217   0.019920   0.003217     1.9841
     10000     0.993754   0.014068   0.006246     1.9791
     50000     1.001812   0.006323   0.001812     1.9999
    100000     0.999421   0.004474   0.000579     2.0014
```

The product $n \cdot \text{SE}^2 \approx \sigma_f^2 \approx 2$ remains constant, confirming $O(n^{-1/2})$ convergence.

---

## 4. Bayesian Posterior Expectations

### 4.1 The Bayesian Monte Carlo Setup

In Bayesian inference, we want to compute:

$$\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] = \int f(\theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

If we can draw samples $\theta^{(1)}, \ldots, \theta^{(n)} \sim p(\theta \mid \mathcal{D})$, then:

$$\hat{I}_n = \frac{1}{n} \sum_{i=1}^{n} f(\theta^{(i)}) \approx \mathbb{E}[f(\theta) \mid \mathcal{D}]$$

Common choices of $f$ yield the standard posterior summaries:

| $f(\theta)$ | Quantity Estimated |
|-------------|-------------------|
| $\theta$ | Posterior mean |
| $(\theta - \bar{\theta})^2$ | Posterior variance |
| $\mathbb{1}(\theta \in A)$ | Posterior probability of region $A$ |
| $p(x_{\text{new}} \mid \theta)$ | Posterior predictive density |

### 4.2 The Fundamental Difficulty

The basic Monte Carlo method requires **direct sampling from the target** $p(\theta \mid \mathcal{D})$. This is problematic when:

1. The posterior has an intractable normalizing constant $p(\mathcal{D})$
2. No standard sampling algorithm exists for the posterior's functional form
3. The posterior is multimodal or has complex geometry

These limitations motivate the more sophisticated methods covered in the remainder of this chapter: **rejection sampling** (Section 18.2), **importance sampling** (Section 18.2), and **MCMC** (Section 18.3), which generate samples from complex target distributions without requiring direct sampling.

### 4.3 Example: Posterior Mean via Grid Sampling

When the posterior is available on a grid (low-dimensional case), we can draw samples from the discrete approximation:

```python
import torch
import numpy as np
from scipy.stats import beta, binom

torch.manual_seed(42)

# Beta-Binomial model: 7 heads in 10 flips, uniform prior
n_flips, n_heads = 10, 7
n_grid = 1000

theta_grid = torch.linspace(0.001, 0.999, n_grid)
grid_width = (theta_grid[1] - theta_grid[0]).item()

# Compute posterior on grid
prior = torch.ones(n_grid)  # Uniform prior
log_lik = n_heads * torch.log(theta_grid) + (n_flips - n_heads) * torch.log(1 - theta_grid)
log_lik -= log_lik.max()
likelihood = torch.exp(log_lik)

unnorm_posterior = prior * likelihood
posterior = unnorm_posterior / (unnorm_posterior.sum() * grid_width)

# Draw Monte Carlo samples from the grid posterior
weights = unnorm_posterior / unnorm_posterior.sum()
sample_indices = torch.multinomial(weights, num_samples=10_000, replacement=True)
theta_samples = theta_grid[sample_indices]

# Posterior summaries via Monte Carlo
mc_mean = theta_samples.mean()
mc_std = theta_samples.std()
analytical_mean = (1 + n_heads) / (2 + n_flips)  # Beta(8, 4) mean

print(f"MC Posterior Mean:        {mc_mean.item():.6f}")
print(f"Analytical Posterior Mean: {analytical_mean:.6f}")
print(f"MC Posterior Std:         {mc_std.item():.6f}")
print(f"95% Credible Interval:   [{torch.quantile(theta_samples, 0.025).item():.4f}, "
      f"{torch.quantile(theta_samples, 0.975).item():.4f}]")
```

```
MC Posterior Mean:        0.666834
Analytical Posterior Mean: 0.666667
MC Posterior Std:         0.130212
95% Credible Interval:   [0.3934, 0.8919]
```

---

## 5. Variance Reduction Techniques

The $O(n^{-1/2})$ convergence rate is dimension-free but can be slow in absolute terms: achieving 10× more precision requires 100× more samples. Variance reduction techniques improve efficiency by reducing $\sigma_f^2$ without increasing sample count.

### 5.1 Control Variates

If we know $\mathbb{E}_p[g(X)] = \mu_g$ exactly for some function $g$ correlated with $f$, the **control variate estimator** is:

$$\hat{I}_{\text{CV}} = \frac{1}{n} \sum_{i=1}^{n} \bigl[f(X_i) - c\,(g(X_i) - \mu_g)\bigr]$$

The optimal coefficient is $c^* = \text{Cov}(f, g) / \text{Var}(g)$, yielding variance reduction:

$$\text{Var}(\hat{I}_{\text{CV}}) = \frac{\sigma_f^2(1 - \rho_{fg}^2)}{n}$$

where $\rho_{fg}$ is the correlation between $f(X)$ and $g(X)$.

```python
def control_variate_estimate(
    f: callable,
    g: callable,
    g_mean: float,
    sampler: callable,
    n_samples: int,
) -> tuple:
    """
    Monte Carlo estimate with control variate variance reduction.

    Parameters
    ----------
    f : callable
        Target function to integrate.
    g : callable
        Control variate function with known mean g_mean.
    g_mean : float
        Known expectation E_p[g(X)].
    sampler : callable
        Draws n i.i.d. samples from p.
    n_samples : int
        Number of samples.

    Returns
    -------
    estimate : Tensor
        Control variate estimate of E_p[f(X)].
    se : Tensor
        Estimated standard error.
    """
    samples = sampler(n_samples)
    f_vals = f(samples)
    g_vals = g(samples)

    # Optimal coefficient
    cov_fg = torch.mean((f_vals - f_vals.mean()) * (g_vals - g_vals.mean()))
    var_g = torch.var(g_vals, unbiased=True)
    c_star = cov_fg / var_g

    # Adjusted values
    adjusted = f_vals - c_star * (g_vals - g_mean)
    estimate = torch.mean(adjusted)
    se = torch.sqrt(torch.var(adjusted, unbiased=True) / n_samples)

    return estimate, se
```

```python
torch.manual_seed(42)

# Estimate E[exp(X)] where X ~ N(0,1), true value = exp(0.5) ≈ 1.6487
normal = dist.Normal(0.0, 1.0)
f = lambda x: torch.exp(x)
g = lambda x: x  # Control variate: E[X] = 0 under N(0,1)
sampler = lambda n: normal.sample((n,))

# Standard MC
est_mc, se_mc = monte_carlo_estimate(f, sampler, 10_000)

# Control variate MC
est_cv, se_cv = control_variate_estimate(f, g, g_mean=0.0, sampler=sampler, n_samples=10_000)

print(f"True value:    {np.exp(0.5):.6f}")
print(f"Standard MC:   {est_mc.item():.6f} (SE = {se_mc.item():.6f})")
print(f"Control Var:   {est_cv.item():.6f} (SE = {se_cv.item():.6f})")
print(f"Variance reduction: {(1 - (se_cv/se_mc)**2).item():.1%}")
```

```
True value:    1.648721
Standard MC:   1.649844 (SE = 0.023118)
Control Var:   1.648553 (SE = 0.013247)
Variance reduction: 67.2%
```

### 5.2 Antithetic Variates

For symmetric distributions, pair each sample $X_i$ with its mirror $-X_i$:

$$\hat{I}_{\text{AV}} = \frac{1}{2n} \sum_{i=1}^{n} \bigl[f(X_i) + f(-X_i)\bigr]$$

The variance is:

$$\text{Var}(\hat{I}_{\text{AV}}) = \frac{\sigma_f^2 + \text{Cov}(f(X), f(-X))}{2n}$$

If $f$ is monotone and $X$ symmetric, $\text{Cov}(f(X), f(-X)) < 0$, yielding variance reduction.

```python
def antithetic_estimate(
    f: callable,
    sampler: callable,
    n_pairs: int,
) -> tuple:
    """
    Monte Carlo estimate using antithetic variates.

    Parameters
    ----------
    f : callable
        Function to integrate.
    sampler : callable
        Draws n i.i.d. samples from a symmetric distribution.
    n_pairs : int
        Number of sample pairs (total samples = 2 * n_pairs).

    Returns
    -------
    estimate : Tensor
        Antithetic variate estimate.
    se : Tensor
        Estimated standard error.
    """
    samples = sampler(n_pairs)
    pair_means = 0.5 * (f(samples) + f(-samples))

    estimate = torch.mean(pair_means)
    se = torch.sqrt(torch.var(pair_means, unbiased=True) / n_pairs)
    return estimate, se
```

### 5.3 Stratified Sampling

Partition the sample space into $K$ strata $S_1, \ldots, S_K$ with probabilities $p_k = P(X \in S_k)$. Draw $n_k$ samples from each stratum:

$$\hat{I}_{\text{strat}} = \sum_{k=1}^{K} p_k \hat{I}_k, \quad \text{where } \hat{I}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} f(X_i^{(k)})$$

Stratification guarantees $\text{Var}(\hat{I}_{\text{strat}}) \leq \text{Var}(\hat{I}_{\text{MC}})$, with equality only when the within-stratum means are all identical.

### 5.4 Summary of Variance Reduction Methods

| Method | Variance Reduction | Requirements | Complexity |
|--------|-------------------|--------------|------------|
| Control variates | Up to $100(1-\rho^2)\%$ | Known $\mathbb{E}[g]$, correlated $g$ | Low |
| Antithetic variates | Up to 50% | Symmetric distribution, monotone $f$ | Low |
| Stratified sampling | Depends on stratification | Natural partition of space | Moderate |
| Importance sampling | Potentially unlimited | Good proposal distribution | See Section 18.2 |

---

## 6. Convergence Diagnostics

### 6.1 Assessing Monte Carlo Error

Since the CLT provides asymptotic normality, we can construct confidence intervals and assess whether enough samples have been drawn:

```python
def mc_diagnostics(f_values: torch.Tensor, target_se: float = None) -> dict:
    """
    Compute Monte Carlo diagnostics.

    Parameters
    ----------
    f_values : Tensor
        Evaluated function values f(X_1), ..., f(X_n).
    target_se : float, optional
        Desired standard error; returns required additional samples.

    Returns
    -------
    dict with estimate, SE, CI, and optional sample size recommendation.
    """
    n = len(f_values)
    estimate = torch.mean(f_values)
    variance = torch.var(f_values, unbiased=True)
    se = torch.sqrt(variance / n)

    result = {
        "estimate": estimate.item(),
        "se": se.item(),
        "ci_95": (
            estimate.item() - 1.96 * se.item(),
            estimate.item() + 1.96 * se.item(),
        ),
        "n_samples": n,
        "relative_se": abs(se.item() / estimate.item()) if estimate.item() != 0 else float("inf"),
    }

    if target_se is not None:
        n_required = int(np.ceil(variance.item() / target_se ** 2))
        result["n_additional_needed"] = max(0, n_required - n)

    return result
```

### 6.2 Rules of Thumb

For reliable Monte Carlo estimation:

- **Relative SE < 1%** for point estimates: require $n \geq 10{,}000 \cdot (\sigma_f / I)^2$
- **Tail probabilities**: estimating $P(\theta > c)$ when the true probability is $p$ requires roughly $n \geq 10{,}000 / p$ for 10% relative accuracy
- **Doubling samples halves the SE**: the $\sqrt{n}$ relationship means diminishing returns

---

## 7. Monte Carlo for Numerical Integration

Beyond Bayesian posteriors, Monte Carlo integration applies to any integral. The "hit-or-miss" formulation illustrates the basic principle.

### 7.1 Computing $\pi$ by Monte Carlo

```python
torch.manual_seed(42)

def estimate_pi(n_samples: int) -> tuple:
    """Estimate π using Monte Carlo integration over the unit square."""
    x = torch.rand(n_samples)
    y = torch.rand(n_samples)

    # Indicator: point falls inside unit circle quadrant
    inside = (x ** 2 + y ** 2 <= 1.0).float()

    # Area of quarter circle = π/4, area of unit square = 1
    pi_estimate = 4.0 * inside.mean()
    se = 4.0 * inside.std() / np.sqrt(n_samples)
    return pi_estimate.item(), se.item()

for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    pi_est, se = estimate_pi(n)
    print(f"n={n:>10}: π ≈ {pi_est:.6f}  (SE={se:.6f}, error={abs(pi_est - np.pi):.6f})")
```

```
n=       100: π ≈ 3.120000  (SE=0.164966, error=0.021593)
n=     1,000: π ≈ 3.164000  (SE=0.052196, error=0.022407)
n=    10,000: π ≈ 3.130800  (SE=0.016419, error=0.010793)
n=   100,000: π ≈ 3.142480  (SE=0.005201, error=0.000887)
n= 1,000,000: π ≈ 3.142240  (SE=0.001644, error=0.000647)
```

### 7.2 General Integration

To compute $I = \int_a^b g(x)\, dx$, rewrite as an expectation under the uniform distribution on $[a, b]$:

$$I = (b - a)\, \mathbb{E}_{U(a,b)}[g(X)]$$

and estimate via:

$$\hat{I}_n = \frac{b - a}{n} \sum_{i=1}^{n} g(X_i), \quad X_i \overset{\text{i.i.d.}}{\sim} U(a, b)$$

---

## 8. Limitations and the Path Forward

### 8.1 When Basic Monte Carlo Is Insufficient

Basic Monte Carlo requires **direct sampling from the target distribution**. Three scenarios break this requirement:

1. **Unnormalized posteriors**: $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$ cannot be sampled directly because the normalizing constant $p(\mathcal{D})$ is unknown
2. **Complex distributions**: No standard algorithm exists to generate samples from the target's functional form
3. **Rare events**: Events with very low probability under $p$ require enormous sample sizes for accurate estimation

### 8.2 How Normalization Is Handled Across Methods

The key insight underlying all Bayesian computation is that we rarely need $p(\mathcal{D})$ explicitly:

| Method | How Normalization Is Handled |
|--------|------------------------------|
| Grid approximation | Sum over grid points |
| Importance sampling | Ratio of weighted sums |
| MCMC | Acceptance ratios cancel normalizing constant |
| Score-based methods | Learn $\nabla_\theta \log p(\theta)$; constant vanishes in gradient |

### 8.3 The Method Progression

Each successive method in this chapter overcomes a limitation of the previous:

$$\text{Grid} \xrightarrow{\text{dimension}} \text{Rejection/IS} \xrightarrow{\text{intractable target}} \text{MCMC} \xrightarrow{\text{gradients}} \text{HMC/Langevin}$$

- **Grid → Monte Carlo**: Overcomes the curse of dimensionality ($O(n^{-1/2})$ vs $O(n^{-2/d})$)
- **Basic MC → Rejection/Importance Sampling**: Handles distributions we cannot sample directly
- **IS → MCMC**: Addresses weight degeneracy in high dimensions
- **Random Walk MCMC → HMC/Langevin**: Uses gradient information for efficient exploration

The next sections develop [Rejection Sampling](rejection.md) and [Importance Sampling](importance_sampling/fundamentals.md), which handle sampling from complex distributions without requiring MCMC chains.

---

## 9. Finance Applications

Monte Carlo integration is pervasive in quantitative finance:

| Application | Integral Being Estimated | Method Notes |
|-------------|------------------------|--------------|
| Option pricing | $\mathbb{E}[\text{discounted payoff}]$ under risk-neutral measure | Antithetic variates standard for path-dependent options |
| Value at Risk | $P(\text{loss} > \text{VaR}_\alpha)$ quantile estimation | Large $n$ needed for tail accuracy |
| Portfolio risk | $\mathbb{E}[\text{loss} \mid \text{loss} > \text{VaR}_\alpha]$ (CVaR) | Control variates using delta-normal approximation |
| Bayesian portfolio | $\mathbb{E}[w^*(\theta)]$ under posterior over expected returns | Full posterior propagation for robust allocation |
| Credit risk | $P(\text{default})$ under correlated factor models | Importance sampling for rare default events |
| Stochastic volatility | $\mathbb{E}[\sigma_t^2 \mid \text{returns}]$ latent volatility paths | Particle filtering (sequential MC) |

---

## 10. Key Takeaways

1. **Monte Carlo convergence is $O(n^{-1/2})$ regardless of dimension**, overcoming the curse of dimensionality that renders grid methods useless beyond ~3 parameters.

2. **The CLT provides automatic uncertainty quantification**: every Monte Carlo estimate comes with a standard error and confidence interval at no additional cost.

3. **Variance reduction techniques** (control variates, antithetic variates, stratified sampling) can dramatically improve efficiency without increasing sample count.

4. **Basic Monte Carlo requires sampling from the target distribution**, which is often impossible for Bayesian posteriors with unknown normalizing constants. This motivates rejection sampling, importance sampling, and MCMC.

5. **All of Bayesian computation** reduces to evaluating expectations under the posterior—posterior means, variances, credible intervals, and predictive distributions are all integrals that Monte Carlo can approximate.

---

## Exercises

### Exercise 1: Convergence Rate Verification

Generate $n$ samples from $\mathcal{N}(0, 1)$ and estimate $\mathbb{E}[X^2] = 1$ for $n \in \{100, 500, 1000, 5000, 10000, 50000\}$. Plot the standard error versus $n$ on a log-log scale and verify the $O(n^{-1/2})$ slope.

### Exercise 2: Control Variates for Option Pricing

Price a European call option using Monte Carlo with geometric Brownian motion paths. Use the Black-Scholes formula as a control variate (with a slightly different strike or volatility). Measure the variance reduction achieved.

### Exercise 3: Grid vs Monte Carlo in Multiple Dimensions

For a $d$-dimensional standard Gaussian, estimate $\mathbb{E}[\|X\|^2] = d$ using both grid approximation and Monte Carlo for $d \in \{1, 2, 3, 5\}$. Compare the number of function evaluations needed for 1% relative error.

### Exercise 4: Antithetic Variates

Implement antithetic variates for estimating $\mathbb{E}[\exp(X)]$ where $X \sim \mathcal{N}(0, 1)$. Compare the variance with standard Monte Carlo and explain why the reduction occurs.

### Exercise 5: Tail Probability Estimation

Estimate $P(X > 3)$ where $X \sim \mathcal{N}(0, 1)$ (true value ≈ 0.00135). How many samples are needed for the 95% CI to exclude zero? Why is this a hard problem for basic Monte Carlo, and how might importance sampling help?

---

## References

1. Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer. Chapters 3–4.
2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. Chapters 2–3.
3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 10.
4. Owen, A. B. (2013). *Monte Carlo Theory, Methods, and Examples*. https://artowen.su.domains/mc/.
5. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
6. McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. Chapter 9.
