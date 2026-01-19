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
