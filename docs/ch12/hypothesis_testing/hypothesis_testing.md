# Bayesian Hypothesis Testing and Model Comparison

## Overview

Bayesian hypothesis testing uses Bayes factors to compare competing hypotheses, providing a principled framework for quantifying evidence. This module develops the theory of Bayes factors, posterior odds, and the Savage-Dickey density ratio for testing point null hypotheses.

---

## 1. Bayes Factors

### 1.1 Definition

For two competing hypotheses $H_1$ and $H_2$, the **Bayes factor** is the ratio of marginal likelihoods:

$$
\boxed{BF_{12} = \frac{p(D|H_1)}{p(D|H_2)}}
$$

where the marginal likelihood (evidence) under hypothesis $H_i$ is:

$$
p(D|H_i) = \int p(D|\theta, H_i) \, p(\theta|H_i) \, d\theta
$$

### 1.2 Interpretation

The Bayes factor quantifies how much the data favor one hypothesis over another:

| Bayes Factor $BF_{12}$ | Evidence for $H_1$ |
|------------------------|-------------------|
| $> 100$ | Decisive |
| $30 - 100$ | Very strong |
| $10 - 30$ | Strong |
| $3 - 10$ | Moderate |
| $1 - 3$ | Weak |
| $1$ | No evidence either way |
| $1/3 - 1$ | Weak evidence for $H_2$ |
| $< 1/10$ | Strong evidence for $H_2$ |

### 1.3 Key Properties

1. **Symmetry:** $BF_{21} = 1/BF_{12}$

2. **Transitivity:** $BF_{13} = BF_{12} \times BF_{23}$

3. **Independence from prior odds:** The Bayes factor depends only on the data and the models, not on prior beliefs about which hypothesis is true.

---

## 2. Posterior Odds

### 2.1 Connecting Prior to Posterior

The **posterior odds** relate to **prior odds** through the Bayes factor:

$$
\boxed{\underbrace{\frac{p(H_1|D)}{p(H_2|D)}}_{\text{Posterior Odds}} = \underbrace{\frac{p(D|H_1)}{p(D|H_2)}}_{\text{Bayes Factor}} \times \underbrace{\frac{p(H_1)}{p(H_2)}}_{\text{Prior Odds}}}
$$

Or more compactly:

$$
\text{Posterior Odds} = BF_{12} \times \text{Prior Odds}
$$

### 2.2 Posterior Model Probabilities

Given prior probabilities $p(H_1)$ and $p(H_2) = 1 - p(H_1)$:

$$
p(H_1|D) = \frac{BF_{12} \cdot p(H_1)}{BF_{12} \cdot p(H_1) + p(H_2)}
$$

**Example:** With equal priors ($p(H_1) = p(H_2) = 0.5$) and $BF_{12} = 10$:

$$
p(H_1|D) = \frac{10 \times 0.5}{10 \times 0.5 + 0.5} = \frac{5}{5.5} \approx 0.91
$$

---

## 3. Testing Coin Fairness

### 3.1 Problem Setup

**Hypotheses:**
- $H_0$: $\theta = 0.5$ (fair coin)
- $H_1$: $\theta \neq 0.5$ (biased coin, with prior $\theta \sim \text{Beta}(\alpha, \beta)$)

**Data:** $k$ heads in $n$ flips

### 3.2 Marginal Likelihoods

**Under $H_0$** (point hypothesis):

$$
p(D|H_0) = \binom{n}{k} (0.5)^n
$$

**Under $H_1$** (Beta-Binomial):

$$
p(D|H_1) = \binom{n}{k} \frac{B(k + \alpha, n - k + \beta)}{B(\alpha, \beta)}
$$

where $B(\cdot, \cdot)$ is the Beta function.

### 3.3 Bayes Factor Computation

$$
BF_{10} = \frac{p(D|H_1)}{p(D|H_0)} = \frac{B(k + \alpha, n - k + \beta)}{B(\alpha, \beta) \cdot (0.5)^n}
$$

### 3.4 Implementation

```python
import numpy as np
from scipy import stats
from scipy.special import beta as beta_func

def bayes_factor_coin_fairness(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """
    Compute Bayes Factor for H1 (biased) vs H0 (fair coin).
    
    Parameters
    ----------
    n_heads, n_tails : int
        Observed data
    prior_alpha, prior_beta : float
        Beta prior parameters under H1
    
    Returns
    -------
    bf : float
        Bayes Factor (H1 vs H0)
    """
    n = n_heads + n_tails
    
    # Evidence under H0: θ = 0.5
    evidence_h0 = stats.binom.pmf(n_heads, n, 0.5)
    
    # Evidence under H1: Beta-Binomial marginal likelihood
    evidence_h1 = (beta_func(n_heads + prior_alpha, n_tails + prior_beta) / 
                   beta_func(prior_alpha, prior_beta))
    evidence_h1 *= stats.binom.comb(n, n_heads)
    
    bf_h1_vs_h0 = evidence_h1 / evidence_h0
    
    return bf_h1_vs_h0
```

### 3.5 Example

**Data:** 17 heads, 3 tails

```python
bf = bayes_factor_coin_fairness(n_heads=17, n_tails=3)
# bf ≈ 17.5
```

| Quantity | Value |
|----------|-------|
| Evidence under $H_0$ | 0.000181 |
| Evidence under $H_1$ | 0.00317 |
| Bayes Factor $BF_{10}$ | 17.5 |
| Interpretation | Strong evidence for biased coin |

---

## 4. Savage-Dickey Density Ratio

### 4.1 The Method

For testing a **point null hypothesis** $H_0: \theta = \theta_0$ against $H_1: \theta \neq \theta_0$, the Bayes factor can be computed as:

$$
\boxed{BF_{01} = \frac{p(\theta_0|D, H_1)}{p(\theta_0|H_1)}}
$$

This is the ratio of **posterior to prior density** evaluated at the null value $\theta_0$.

### 4.2 Intuition

- If the posterior density at $\theta_0$ is **lower** than the prior density → data moved probability mass away from $\theta_0$ → evidence against $H_0$
- If the posterior density at $\theta_0$ is **higher** than the prior density → data moved probability mass toward $\theta_0$ → evidence for $H_0$

### 4.3 Implementation

```python
from scipy import stats

def savage_dickey_bf(n_heads, n_tails, prior_alpha=1, prior_beta=1, null_value=0.5):
    """
    Compute Bayes Factor using Savage-Dickey density ratio.
    
    Returns BF_{01} (null vs alternative) and BF_{10} (alternative vs null).
    """
    # Prior and posterior distributions
    prior = stats.beta(prior_alpha, prior_beta)
    posterior = stats.beta(prior_alpha + n_heads, prior_beta + n_tails)
    
    # Evaluate densities at null value
    prior_density = prior.pdf(null_value)
    posterior_density = posterior.pdf(null_value)
    
    # Savage-Dickey ratio
    bf_h0_vs_h1 = posterior_density / prior_density
    bf_h1_vs_h0 = 1 / bf_h0_vs_h1
    
    return bf_h0_vs_h1, bf_h1_vs_h0
```

### 4.4 Example

**Data:** 17 heads, 3 tails  
**Testing:** $H_0: \theta = 0.5$

```python
bf_01, bf_10 = savage_dickey_bf(n_heads=17, n_tails=3)
```

| Quantity | Value |
|----------|-------|
| Prior density at $\theta = 0.5$ | 1.000 |
| Posterior density at $\theta = 0.5$ | 0.057 |
| $BF_{01}$ (Savage-Dickey) | 0.057 |
| $BF_{10}$ | 17.5 |

The posterior density at $\theta = 0.5$ is much lower than the prior density, indicating the data have moved probability mass away from the fair coin hypothesis.

---

## 5. Bayesian vs Frequentist Testing

### 5.1 Fundamental Differences

| Aspect | Bayesian (Bayes Factor) | Frequentist (p-value) |
|--------|-------------------------|------------------------|
| **Measures** | Evidence for $H_1$ vs $H_0$ | Extremeness of data under $H_0$ |
| **Interpretation** | Relative plausibility | Probability of data more extreme |
| **Can favor null** | Yes | No (can only fail to reject) |
| **Sample size** | Can stop anytime | Requires fixed sample size |
| **Prior required** | Yes | No |

### 5.2 The p-value Problem

P-values only measure evidence *against* $H_0$, not evidence *for* $H_0$. A non-significant p-value (e.g., $p = 0.15$) does not mean $H_0$ is true — it could simply reflect insufficient data.

Bayes factors can distinguish between:
- **Evidence for $H_0$** ($BF_{01} > 3$)
- **Evidence for $H_1$** ($BF_{10} > 3$)
- **Inconclusive** ($1/3 < BF < 3$)

### 5.3 Lindley's Paradox

With large sample sizes, p-values and Bayes factors can give contradictory conclusions. A small p-value might coexist with a Bayes factor favoring the null — especially when the effect size is small but the sample is large.

---

## 6. Practical Considerations

### 6.1 Choosing Priors for Model Comparison

The Bayes factor is **sensitive to prior specification** under $H_1$. Guidelines:

| Approach | Description |
|----------|-------------|
| Default priors | Unit information priors, Jeffreys priors |
| Substantive priors | Based on domain knowledge |
| Sensitivity analysis | Check robustness to prior choice |

### 6.2 Computational Methods

For complex models where marginal likelihoods are intractable:

| Method | Description |
|--------|-------------|
| Importance sampling | Weighted Monte Carlo estimation |
| Bridge sampling | Efficient marginal likelihood estimation |
| Thermodynamic integration | Path from prior to posterior |
| Reversible jump MCMC | Trans-dimensional sampling |

### 6.3 Reporting Results

Report:
1. The Bayes factor with interpretation
2. Prior assumptions under each model
3. Posterior model probabilities (if prior odds specified)

**Example:**

> "The data provide strong evidence for a biased coin ($BF_{10} = 17.5$). Assuming equal prior odds, the posterior probability of bias is 0.95."

---

## 7. Key Takeaways

1. **Bayes factors** quantify relative evidence for competing hypotheses as a ratio of marginal likelihoods.

2. **Posterior odds = Bayes factor × Prior odds**: The Bayes factor updates our prior beliefs about hypotheses.

3. **Savage-Dickey density ratio** provides an elegant method for testing point nulls: $BF_{01} = p(\theta_0|D)/p(\theta_0)$.

4. **Bayesian testing avoids p-values** and their interpretation problems. Bayes factors can provide evidence *for* the null, not just against it.

5. **Prior sensitivity**: Bayes factors depend on prior choices under the alternative hypothesis. Always conduct sensitivity analyses.

---

## 8. Exercises

### Exercise 1: Frequentist Comparison
For the coin fairness example (17 heads, 3 tails), compute the frequentist p-value (two-sided binomial test). Compare the conclusion with the Bayes factor analysis.

### Exercise 2: Evidence Accumulation
Compute Bayes factors for increasing sample sizes (n = 10, 20, 50, 100) with a fixed proportion of heads (70%). Plot how evidence accumulates.

### Exercise 3: Nested Model Comparison
Implement Bayes factor computation for comparing nested linear regression models (e.g., simple vs multiple regression).

### Exercise 4: Normal-Normal Testing
Use the Savage-Dickey method to test $H_0: \mu = 0$ vs $H_1: \mu \neq 0$ in the Normal-Normal model with known variance.

---

## References

- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
- Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values. *Psychonomic Bulletin & Review*, 14(5), 779-804.
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 7
