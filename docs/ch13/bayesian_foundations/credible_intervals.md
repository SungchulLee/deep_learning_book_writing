# Credible Intervals

## Overview

Credible intervals provide Bayesian uncertainty quantification for parameter estimates. This module develops equal-tailed and Highest Posterior Density (HPD) intervals, and clarifies the fundamental distinction between Bayesian credible intervals and frequentist confidence intervals.

---

## 1. Credible Intervals: Definition and Interpretation

### 1.1 Definition

A **$(1-\alpha) \times 100\%$ credible interval** $[L, U]$ for parameter $\theta$ satisfies:

$$
\boxed{P(L \leq \theta \leq U \mid D) = 1 - \alpha}
$$

### 1.2 Interpretation

Given the observed data $D$, there is a $(1-\alpha) \times 100\%$ probability that the true parameter lies within the interval.

**Example:** A 95% credible interval $[0.55, 0.82]$ means:

> "Given our data and prior, there is a 95% probability that $\theta$ is between 0.55 and 0.82."

This is a direct probability statement about the parameter — the quantity we actually care about.

---

## 2. Types of Credible Intervals

### 2.1 Equal-Tailed Interval

The **equal-tailed interval** places equal probability mass ($\alpha/2$) in each tail:

$$
[L, U] = [q_{\alpha/2}, q_{1-\alpha/2}]
$$

where $q_p$ is the $p$-th quantile of the posterior distribution.

**Properties:**
- Easy to compute from quantiles
- Symmetric in probability (not necessarily in width)
- May not be the shortest interval

**Implementation:**

```python
def compute_equal_tailed_interval(posterior_dist, alpha=0.05):
    """
    Compute equal-tailed credible interval.
    
    Parameters
    ----------
    posterior_dist : scipy.stats distribution
        Posterior distribution object
    alpha : float
        Significance level (0.05 for 95% interval)
    
    Returns
    -------
    interval : tuple
        (lower_bound, upper_bound)
    """
    lower = posterior_dist.ppf(alpha / 2)
    upper = posterior_dist.ppf(1 - alpha / 2)
    return (lower, upper)
```

### 2.2 Highest Posterior Density (HPD) Interval

The **HPD interval** is the shortest interval containing $(1-\alpha) \times 100\%$ of the posterior probability mass.

**Defining property:** Every point inside the HPD interval has higher posterior density than every point outside.

$$
\{θ : p(θ|D) \geq k\}
$$

where $k$ is chosen such that $P(\theta \in \text{HPD} \mid D) = 1 - \alpha$.

**Properties:**
- Shortest possible interval for given coverage
- Optimal for unimodal posteriors
- More complex to compute
- For symmetric distributions: HPD = Equal-tailed

**Implementation (from samples):**

```python
import numpy as np

def compute_hpd_interval(samples, alpha=0.05):
    """
    Compute HPD interval from posterior samples.
    
    Parameters
    ----------
    samples : array-like
        Samples from posterior distribution
    alpha : float
        Significance level
    
    Returns
    -------
    interval : tuple
        (lower_bound, upper_bound)
    """
    samples_sorted = np.sort(samples)
    n = len(samples)
    
    # Number of samples to include
    n_included = int(np.ceil((1 - alpha) * n))
    
    # Find all possible intervals of this size
    n_intervals = n - n_included + 1
    interval_widths = samples_sorted[n_included-1:] - samples_sorted[:n_intervals]
    
    # Choose interval with minimum width
    min_idx = np.argmin(interval_widths)
    hpd_lower = samples_sorted[min_idx]
    hpd_upper = samples_sorted[min_idx + n_included - 1]
    
    return (hpd_lower, hpd_upper)
```

### 2.3 Comparison

| Property | Equal-Tailed | HPD |
|----------|--------------|-----|
| Computation | Simple (quantiles) | More complex |
| Optimality | Not guaranteed shortest | Shortest interval |
| Symmetric posteriors | Identical | Identical |
| Skewed posteriors | Longer | Shorter |
| Interpretation | Equal tail probabilities | Highest density region |

---

## 3. Example: Beta Posterior

### 3.1 Setup

Consider a Beta posterior from coin flipping:

- Data: 15 heads, 5 tails
- Prior: Beta$(1, 1)$ (uniform)
- Posterior: Beta$(16, 6)$

### 3.2 Computing Intervals

```python
from scipy import stats

# Posterior distribution
posterior = stats.beta(16, 6)

# 95% Equal-tailed interval
et_lower = posterior.ppf(0.025)  # 0.5765
et_upper = posterior.ppf(0.975)  # 0.8875
# Width: 0.311

# 95% HPD interval (from 100,000 samples)
samples = posterior.rvs(100000)
hpd_lower, hpd_upper = compute_hpd_interval(samples, alpha=0.05)
# Approximately [0.571, 0.885]
# Width: 0.314
```

### 3.3 Results

| Interval Type | Lower | Upper | Width |
|---------------|-------|-------|-------|
| Equal-Tailed | 0.577 | 0.888 | 0.311 |
| HPD | 0.571 | 0.885 | 0.314 |

For this mildly skewed Beta$(16, 6)$ posterior, both intervals are similar. The difference becomes more pronounced with highly skewed posteriors.

---

## 4. Credible Intervals vs Confidence Intervals

### 4.1 Fundamental Distinction

| Aspect | Credible Interval (Bayesian) | Confidence Interval (Frequentist) |
|--------|------------------------------|-----------------------------------|
| **Probability statement** | About the parameter | About the procedure |
| **Data** | Fixed (observed) | Random (hypothetical repetitions) |
| **Parameter** | Random (has distribution) | Fixed (unknown constant) |
| **Interpretation** | "95% probability θ is here" | "95% of such intervals contain θ" |

### 4.2 Interpretation Comparison

**95% Credible Interval $[0.55, 0.82]$:**

> "Given our observed data, there is a 95% probability that the true parameter $\theta$ lies between 0.55 and 0.82."

**95% Confidence Interval $[0.55, 0.82]$:**

> "If we were to repeat this experiment many times and compute an interval each time, 95% of those intervals would contain the true parameter $\theta$."

### 4.3 The Frequentist Caveat

A confidence interval does **not** mean there's a 95% probability the parameter is in the interval. In frequentist statistics, the parameter is a fixed (but unknown) constant — either it's in the interval or it isn't. The probability statement is about the random interval, not the fixed parameter.

---

## 5. Simulation Study

### 5.1 Coverage Comparison

We can verify interval properties through simulation:

```python
import numpy as np
from scipy import stats

def coverage_simulation(true_p=0.7, n_trials=20, n_experiments=1000):
    """Compare coverage of credible and confidence intervals."""
    
    np.random.seed(42)
    
    credible_coverage = 0
    confidence_coverage = 0
    
    for _ in range(n_experiments):
        # Generate data
        successes = np.random.binomial(n_trials, true_p)
        
        # Bayesian 95% credible interval (uniform prior)
        post = stats.beta(1 + successes, 1 + n_trials - successes)
        cred_lower, cred_upper = post.ppf(0.025), post.ppf(0.975)
        
        if cred_lower <= true_p <= cred_upper:
            credible_coverage += 1
        
        # Frequentist 95% confidence interval (Wald)
        p_hat = successes / n_trials
        se = np.sqrt(p_hat * (1 - p_hat) / n_trials)
        conf_lower = max(0, p_hat - 1.96 * se)
        conf_upper = min(1, p_hat + 1.96 * se)
        
        if conf_lower <= true_p <= conf_upper:
            confidence_coverage += 1
    
    return {
        'credible': credible_coverage / n_experiments,
        'confidence': confidence_coverage / n_experiments
    }
```

### 5.2 Typical Results

For $p_{\text{true}} = 0.7$, $n = 20$, over 1000 experiments:

| Interval Type | Observed Coverage |
|---------------|-------------------|
| 95% Credible | ~94-96% |
| 95% Confidence (Wald) | ~90-94% |

The Wald confidence interval often has **under-coverage** for small samples or extreme probabilities. The Bayesian credible interval tends to have better calibration.

---

## 6. Practical Considerations

### 6.1 When to Use Each Interval Type

| Situation | Recommended Interval |
|-----------|---------------------|
| Symmetric posterior | Either (they're equivalent) |
| Skewed posterior | HPD (shorter) |
| Quick computation | Equal-tailed |
| Optimal decision-making | HPD |
| Communication to non-statisticians | Credible (more intuitive) |

### 6.2 Reporting Credible Intervals

Standard practice is to report:

1. **Point estimate** (posterior mean or MAP)
2. **95% credible interval** (equal-tailed or HPD)
3. **Posterior distribution** (when possible)

**Example:**

> "The estimated success probability is 0.73 (95% CI: 0.58–0.89)"

### 6.3 Multivariate Credible Regions

For multiple parameters, we compute **credible regions** instead of intervals:

$$
P(\theta \in \mathcal{R} \mid D) = 1 - \alpha
$$

For 2D parameters, this produces **credible ellipses** (for Gaussian posteriors) or more complex regions.

---

## 7. Key Takeaways

1. **Credible intervals** make direct probability statements about parameters — the intuitive interpretation most people want.

2. **Equal-tailed intervals** are easy to compute using quantiles; **HPD intervals** are the shortest possible intervals.

3. **For symmetric posteriors**, equal-tailed and HPD intervals coincide. For skewed posteriors, HPD is shorter.

4. **Credible ≠ Confidence intervals**: They answer different questions and have different interpretations.
   - Credible: "95% probability θ is here" (probability about parameter)
   - Confidence: "95% of such procedures capture θ" (probability about procedure)

5. In practice, Bayesian credible intervals often have better **coverage properties** than frequentist confidence intervals, especially for small samples.

---

## 8. Exercises

### Exercise 1: Skewed Posterior
Compute both equal-tailed and HPD intervals for a Gamma$(2, 1)$ posterior. Verify that HPD is shorter.

### Exercise 2: Symmetric Posteriors
Show analytically (or by simulation) that for symmetric distributions (e.g., Normal), equal-tailed and HPD intervals are identical.

### Exercise 3: Multivariate Regions
Implement 95% credible ellipses for a 2D Normal posterior. Visualize how the region changes with different covariance structures.

### Exercise 4: Coverage Study
Compare the actual coverage of credible vs confidence intervals for small sample sizes ($n = 5, 10, 20$) and extreme true probabilities ($p = 0.05, 0.5, 0.95$).

---

## References

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 2
- Kruschke, J. *Doing Bayesian Data Analysis* (2nd ed.), Chapter 12
- Hoff, P. *A First Course in Bayesian Statistical Methods*, Chapter 3
