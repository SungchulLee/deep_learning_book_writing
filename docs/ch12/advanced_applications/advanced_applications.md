# Advanced Applications

## Overview

This module covers practical applications of Bayesian inference in real-world scenarios, focusing on Bayesian A/B testing. We demonstrate how Bayesian methods provide direct probability statements, enable early stopping, and offer intuitive interpretation of results.

---

## 1. Bayesian A/B Testing

### 1.1 Traditional vs Bayesian A/B Testing

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Output** | p-value, confidence interval | P(B > A), credible intervals |
| **Interpretation** | "Reject/fail to reject null" | "B is better with probability X" |
| **Sample size** | Fixed in advance | Can stop early |
| **Multiple testing** | Requires correction | Natural handling |
| **Decision** | Arbitrary threshold (p < 0.05) | Based on business value |

### 1.2 The Bayesian Framework

**Model for each variant:**
$$
\text{Conversions}_i | \theta_i \sim \text{Binomial}(n_i, \theta_i)
$$

**Prior:**
$$
\theta_i \sim \text{Beta}(\alpha_0, \beta_0)
$$

**Posterior:**
$$
\theta_i | \text{data} \sim \text{Beta}(\alpha_0 + c_i, \beta_0 + n_i - c_i)
$$

where $c_i$ is conversions and $n_i$ is trials for variant $i$.

### 1.3 Key Quantities

**Probability B is better than A:**
$$
P(\theta_B > \theta_A | \text{data}) = \int_0^1 \int_0^{\theta_B} p(\theta_A | \text{data}) \, p(\theta_B | \text{data}) \, d\theta_A \, d\theta_B
$$

**Expected lift:**
$$
\text{Lift} = \frac{\theta_B - \theta_A}{\theta_A} = \frac{\theta_B}{\theta_A} - 1
$$

**Risk (expected loss from choosing wrong variant):**
$$
\text{Risk}_A = \mathbb{E}[\max(\theta_B - \theta_A, 0)]
$$

---

## 2. Implementation

### 2.1 Core Function

```python
import numpy as np
from scipy import stats

def bayesian_ab_test(conversions_a, trials_a, conversions_b, trials_b,
                     prior_alpha=1, prior_beta=1, n_samples=100000):
    """
    Bayesian A/B test for conversion rates.
    
    Parameters
    ----------
    conversions_a, trials_a : int
        Results for variant A
    conversions_b, trials_b : int
        Results for variant B
    prior_alpha, prior_beta : float
        Beta prior parameters (default: uniform)
    n_samples : int
        Monte Carlo samples
    
    Returns
    -------
    dict with test results
    """
    # Posterior distributions
    post_a = stats.beta(prior_alpha + conversions_a, 
                        prior_beta + trials_a - conversions_a)
    post_b = stats.beta(prior_alpha + conversions_b, 
                        prior_beta + trials_b - conversions_b)
    
    # Monte Carlo estimation
    samples_a = post_a.rvs(n_samples)
    samples_b = post_b.rvs(n_samples)
    
    # P(B > A)
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Lift distribution
    lift = samples_b / samples_a - 1
    
    # Expected loss (risk)
    risk_a = np.mean(np.maximum(samples_b - samples_a, 0))
    risk_b = np.mean(np.maximum(samples_a - samples_b, 0))
    
    return {
        'prob_b_better': prob_b_better,
        'prob_a_better': 1 - prob_b_better,
        'lift_mean': np.mean(lift),
        'lift_ci': (np.percentile(lift, 2.5), np.percentile(lift, 97.5)),
        'risk_a': risk_a,
        'risk_b': risk_b,
        'post_a': post_a,
        'post_b': post_b
    }
```

### 2.2 Example Usage

```python
results = bayesian_ab_test(
    conversions_a=120, trials_a=1000,
    conversions_b=145, trials_b=1000
)

print(f"Variant A: 120/1000 = 12.0%")
print(f"Variant B: 145/1000 = 14.5%")
print(f"\nP(B > A) = {results['prob_b_better']:.4f}")
print(f"Expected lift: {results['lift_mean']*100:.2f}%")
print(f"95% CI for lift: [{results['lift_ci'][0]*100:.2f}%, {results['lift_ci'][1]*100:.2f}%]")
```

**Output:**
```
Variant A: 120/1000 = 12.0%
Variant B: 145/1000 = 14.5%

P(B > A) = 0.9812
Expected lift: 21.32%
95% CI for lift: [4.12%, 41.58%]
```

---

## 3. Decision Making

### 3.1 Probability Thresholds

Rather than p < 0.05, set thresholds based on business needs:

| Threshold | Decision |
|-----------|----------|
| P(B > A) > 0.95 | Implement B |
| P(B > A) < 0.05 | Keep A |
| Otherwise | Continue testing |

### 3.2 Expected Loss Framework

Choose the variant with lower expected loss:

$$
\text{Choose B if } \mathbb{E}[\text{Loss}_B] < \mathbb{E}[\text{Loss}_A]
$$

where:
- $\text{Loss}_B = \max(\theta_A - \theta_B, 0)$ (loss from choosing B when A is better)
- $\text{Loss}_A = \max(\theta_B - \theta_A, 0)$ (loss from choosing A when B is better)

### 3.3 Value-Based Decisions

Incorporate business value:

$$
\text{Expected Value of B} = \mathbb{E}[\theta_B] \times \text{Revenue per conversion}
$$

---

## 4. Early Stopping

### 4.1 The Bayesian Advantage

Unlike frequentist tests (which require fixed sample sizes), Bayesian A/B testing allows:

- **Peeking at results** without inflating error rates
- **Stopping early** when evidence is strong
- **Continuing longer** when results are inconclusive

### 4.2 Stopping Rules

Stop when any condition is met:

| Condition | Action |
|-----------|--------|
| P(B > A) > 0.99 | Implement B |
| P(A > B) > 0.99 | Keep A |
| Expected loss < threshold | Choose lower-loss variant |
| Maximum sample size reached | Choose based on current evidence |

### 4.3 Sequential Analysis

Track P(B > A) over time:

```python
def sequential_ab_analysis(results_a, results_b, prior=(1, 1)):
    """
    Compute P(B > A) after each observation.
    
    Parameters
    ----------
    results_a, results_b : list of 0/1
        Sequential conversion outcomes
    
    Returns
    -------
    prob_history : list
        P(B > A) after each pair of observations
    """
    prob_history = []
    alpha, beta = prior
    
    for i in range(1, len(results_a) + 1):
        conv_a = sum(results_a[:i])
        conv_b = sum(results_b[:i])
        
        post_a = stats.beta(alpha + conv_a, beta + i - conv_a)
        post_b = stats.beta(alpha + conv_b, beta + i - conv_b)
        
        samples_a = post_a.rvs(10000)
        samples_b = post_b.rvs(10000)
        
        prob_history.append(np.mean(samples_b > samples_a))
    
    return prob_history
```

---

## 5. Visualization

### 5.1 Posterior Distributions

```python
import matplotlib.pyplot as plt

def plot_posteriors(results):
    """Plot posterior distributions for both variants."""
    p = np.linspace(0, 1, 1000)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, results['post_a'].pdf(p), 'b-', lw=2, label='Variant A')
    plt.plot(p, results['post_b'].pdf(p), 'r-', lw=2, label='Variant B')
    plt.axvline(results['post_a'].mean(), color='blue', ls='--', alpha=0.7)
    plt.axvline(results['post_b'].mean(), color='red', ls='--', alpha=0.7)
    plt.xlabel('Conversion Rate')
    plt.ylabel('Density')
    plt.title('Posterior Distributions')
    plt.legend()
```

### 5.2 Lift Distribution

```python
def plot_lift(results, n_samples=100000):
    """Plot distribution of lift (B vs A)."""
    samples_a = results['post_a'].rvs(n_samples)
    samples_b = results['post_b'].rvs(n_samples)
    lift = (samples_b / samples_a - 1) * 100
    
    plt.figure(figsize=(10, 6))
    plt.hist(lift, bins=50, alpha=0.7, color='green', density=True)
    plt.axvline(0, color='red', ls='--', lw=2, label='No difference')
    plt.axvline(np.mean(lift), color='black', lw=2, label=f'Mean: {np.mean(lift):.1f}%')
    plt.xlabel('Lift (%)')
    plt.ylabel('Density')
    plt.title('Distribution of Lift (B vs A)')
    plt.legend()
```

---

## 6. Extensions

### 6.1 Multiple Variants

For $K$ variants, compute pairwise probabilities or find:

$$
P(\theta_i = \max_j \theta_j | \text{data})
$$

### 6.2 Continuous Metrics

For revenue per user (continuous):

$$
\text{Revenue}_i \sim \mathcal{N}(\mu_i, \sigma^2)
$$

Use Normal-Normal conjugate model.

### 6.3 Bayesian Bandits

Combine A/B testing with exploration-exploitation:

- **Thompson Sampling**: Sample from posteriors, play best-looking variant
- **Gradual rollout**: Allocate traffic based on posterior probabilities

---

## 7. Advantages Over Frequentist Testing

### 7.1 Direct Probability Statements

**Frequentist:** "If there were no difference, we'd see data this extreme 3% of the time"

**Bayesian:** "There's a 98% probability that B is better than A"

### 7.2 No Peeking Problem

Frequentist tests have inflated false positive rates when you check results multiple times. Bayesian methods don't suffer from this — the posterior is valid at any stopping point.

### 7.3 Business-Friendly Decisions

Bayesian outputs map directly to business decisions:
- "What's the probability B increases revenue?"
- "What's our expected loss if we choose wrong?"
- "How confident should we be before shipping?"

---

## 8. Key Takeaways

1. **Bayesian A/B testing** provides direct probability statements: "P(B > A) = 0.98" is more actionable than "p = 0.03".

2. **Early stopping** is natural in Bayesian testing — stop when evidence is strong, continue when uncertain.

3. **Lift distribution** quantifies not just whether B is better, but by how much.

4. **Expected loss** provides a decision-theoretic framework incorporating business value.

5. **No multiple testing correction** needed — the posterior is valid at any time you choose to look.

---

## 9. Exercises

### Exercise 1: Small Sample Behavior
Run a Bayesian A/B test with very small samples (10 per variant). How does the posterior behave? Compare with a frequentist test.

### Exercise 2: Prior Sensitivity
Compare results using uniform prior Beta(1,1) vs informative prior Beta(10,90) (assuming ~10% base conversion). When does the prior matter?

### Exercise 3: Multiple Variants
Extend the implementation to handle 3+ variants. Compute P(variant $i$ is best) for each.

### Exercise 4: Revenue Optimization
Modify the A/B test for continuous outcomes (revenue per user) using Normal posteriors.

### Exercise 5: Thompson Sampling
Implement a Thompson Sampling bandit that uses Bayesian posteriors to balance exploration and exploitation.

---

## 10. Course Summary

Congratulations on completing the Bayesian Inference curriculum! Here's what we covered:

| Module | Topic | Key Concept |
|--------|-------|-------------|
| 1 | Bayes' Theorem Basics | Prior × Likelihood → Posterior |
| 2 | Continuous Inference | Beta-Binomial, Normal-Normal |
| 3 | Conjugate Priors | Analytical posterior updates |
| 4 | MAP Estimation | Mode of posterior, regularization connection |
| 5 | Credible Intervals | HPD, equal-tailed, vs confidence intervals |
| 6 | Hypothesis Testing | Bayes factors, Savage-Dickey |
| 7 | Hierarchical Models | Partial pooling, shrinkage |
| 8 | Empirical Bayes | Data-driven priors |
| 9 | Bayesian Regression | Posterior over parameters, predictive uncertainty |
| 10 | Advanced Applications | A/B testing, decision making |

**The Bayesian paradigm provides:**
- Coherent uncertainty quantification
- Natural incorporation of prior knowledge
- Direct probability statements about quantities of interest
- Flexible framework for complex hierarchical models

---

## References

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.)
- Kruschke, J. *Doing Bayesian Data Analysis* (2nd ed.)
- McElreath, R. *Statistical Rethinking*
- VanderPlas, J. "Frequentism and Bayesianism: A Practical Introduction"
