# Bayes Factors

The **Bayes factor** is the ratio of model evidences, providing a principled measure of the relative support the data provide for one model over another. Unlike p-values, Bayes factors quantify evidence in favor of a hypothesis, not just against a null, and they naturally account for model complexity through Bayesian Occam's razor.

---

## Motivation: Beyond Null Hypothesis Testing

### Limitations of Classical Hypothesis Testing

Classical hypothesis testing has fundamental limitations:

**Asymmetry**: Can only reject or fail to reject the null
- P-value measures $P(\text{data this extreme} \mid H_0)$
- Cannot measure evidence *for* $H_0$
- "Not significant" ≠ "null is true"

**Sensitivity to sample size**: With enough data, any null is rejected
- Trivial effects become "significant"
- Practical significance conflated with statistical significance

**No quantification of evidence strength**: Only binary decisions
- $p = 0.049$ and $p = 0.0001$ both lead to "reject"
- $p = 0.051$ and $p = 0.5$ both lead to "fail to reject"

### What Bayes Factors Provide

The Bayes factor addresses these limitations:

1. **Symmetric comparison**: Evidence for either hypothesis
2. **Continuous measure**: Quantifies strength of evidence
3. **Automatic Occam's razor**: Penalizes unnecessary complexity
4. **Coherent probability calculus**: Updates beliefs via Bayes' theorem

---

## Definition and Interpretation

### Formal Definition

The **Bayes factor** comparing model $\mathcal{M}_1$ to model $\mathcal{M}_2$ is:

$$
\boxed{B_{12} = \frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_2)}}
$$

where $p(\mathcal{D} \mid \mathcal{M}_k)$ is the **model evidence** (marginal likelihood):

$$
p(\mathcal{D} \mid \mathcal{M}_k) = \int p(\mathcal{D} \mid \theta_k, \mathcal{M}_k) \, p(\theta_k \mid \mathcal{M}_k) \, d\theta_k
$$

### Relationship to Posterior Odds

Bayes factors connect prior to posterior beliefs about models:

$$
\underbrace{\frac{p(\mathcal{M}_1 \mid \mathcal{D})}{p(\mathcal{M}_2 \mid \mathcal{D})}}_{\text{Posterior odds}} = \underbrace{B_{12}}_{\text{Bayes factor}} \times \underbrace{\frac{p(\mathcal{M}_1)}{p(\mathcal{M}_2)}}_{\text{Prior odds}}
$$

**Key insight**: The Bayes factor is the factor by which the data update our relative beliefs.

### Log Bayes Factor

For numerical stability, work with log Bayes factors:

$$
\log B_{12} = \log p(\mathcal{D} \mid \mathcal{M}_1) - \log p(\mathcal{D} \mid \mathcal{M}_2)
$$

**Properties**:
- $\log B_{12} = -\log B_{21}$ (antisymmetric)
- $\log B_{13} = \log B_{12} + \log B_{23}$ (transitive)
- Additive across independent datasets: $\log B_{12}^{\text{total}} = \sum_i \log B_{12}^{(i)}$

---

## Interpretation Guidelines

### Kass and Raftery (1995) Scale

A widely used interpretation scale:

| $\log_{10} B_{12}$ | $\log B_{12}$ | $B_{12}$ | Evidence for $\mathcal{M}_1$ |
|-------------------|---------------|----------|------------------------------|
| 0 to 0.5 | 0 to 1.15 | 1 to 3.2 | Barely worth mentioning |
| 0.5 to 1 | 1.15 to 2.3 | 3.2 to 10 | Substantial |
| 1 to 2 | 2.3 to 4.6 | 10 to 100 | Strong |
| > 2 | > 4.6 | > 100 | Decisive |

**Symmetric interpretation**: $B_{12} = 0.1$ means strong evidence for $\mathcal{M}_2$.

### Jeffreys (1961) Scale

Harold Jeffreys' original scale (using $\log_{10}$):

| $\log_{10} B_{12}$ | Interpretation |
|-------------------|----------------|
| 0 | No evidence |
| 0 to 0.5 | Barely worth mentioning |
| 0.5 to 1 | Substantial |
| 1 to 1.5 | Strong |
| 1.5 to 2 | Very strong |
| > 2 | Decisive |

### Converting to Posterior Probabilities

With equal prior odds:

$$
p(\mathcal{M}_1 \mid \mathcal{D}) = \frac{B_{12}}{1 + B_{12}} = \frac{1}{1 + B_{21}}
$$

| $B_{12}$ | $p(\mathcal{M}_1 \mid \mathcal{D})$ |
|----------|-------------------------------------|
| 1 | 0.50 |
| 3 | 0.75 |
| 10 | 0.91 |
| 100 | 0.99 |
| 1000 | 0.999 |

### Cautionary Notes on Interpretation

1. **Not a hypothesis test**: Bayes factors measure relative evidence, not "significance"
2. **Scale is arbitrary**: Different scales exist; interpret comparatively
3. **Prior-dependent**: Strong dependence on prior specification (unlike posterior inference)
4. **Model adequacy**: High Bayes factor doesn't mean the model is good, only better than alternatives

---

## Mathematical Properties

### Symmetry and Transitivity

**Reciprocal relationship**:
$$
B_{21} = \frac{1}{B_{12}}
$$

**Transitivity** (for pairwise comparison):
$$
B_{13} = B_{12} \cdot B_{23}
$$

### Additivity of Log Evidence

For independent datasets $\mathcal{D}_1, \mathcal{D}_2$:

$$
\log B_{12}^{\text{total}} = \log B_{12}^{(\mathcal{D}_1)} + \log B_{12}^{(\mathcal{D}_2)}
$$

This enables sequential updating of model comparisons.

### Consistency

Under regularity conditions, if $\mathcal{M}_1$ is true:

$$
\log B_{12} \xrightarrow{p} +\infty \quad \text{as } n \to \infty
$$

If $\mathcal{M}_2$ is true:

$$
\log B_{12} \xrightarrow{p} -\infty \quad \text{as } n \to \infty
$$

The Bayes factor is **consistent**: it will eventually favor the true model.

### Asymptotic Behavior

For nested models with $k$ extra parameters:

$$
\log B_{12} \approx \log p(\mathcal{D} \mid \hat{\theta}_1) - \log p(\mathcal{D} \mid \hat{\theta}_2) - \frac{k}{2} \log n + O(1)
$$

This connects to the BIC approximation.

---

## Bayes Factors for Common Tests

### Point Null vs Alternative

**Testing $H_0: \theta = \theta_0$ vs $H_1: \theta \neq \theta_0$**

Under $H_0$: $p(\mathcal{D} \mid H_0) = p(\mathcal{D} \mid \theta_0)$

Under $H_1$: $p(\mathcal{D} \mid H_1) = \int p(\mathcal{D} \mid \theta) \, p(\theta \mid H_1) \, d\theta$

**Bayes factor**:

$$
B_{01} = \frac{p(\mathcal{D} \mid \theta_0)}{\int p(\mathcal{D} \mid \theta) \, p(\theta \mid H_1) \, d\theta}
$$

### Savage-Dickey Density Ratio

For nested models where $H_0: \theta = \theta_0$ is a special case of $H_1$:

$$
B_{01} = \frac{p(\theta_0 \mid \mathcal{D}, H_1)}{p(\theta_0 \mid H_1)}
$$

**Interpretation**: Ratio of posterior to prior density at the null value.

**Derivation**:

$$
B_{01} = \frac{p(\mathcal{D} \mid H_0)}{p(\mathcal{D} \mid H_1)} = \frac{p(\mathcal{D} \mid \theta_0)}{\int p(\mathcal{D} \mid \theta) \, p(\theta \mid H_1) \, d\theta}
$$

Using Bayes' theorem for $H_1$:

$$
p(\theta_0 \mid \mathcal{D}, H_1) = \frac{p(\mathcal{D} \mid \theta_0) \, p(\theta_0 \mid H_1)}{p(\mathcal{D} \mid H_1)}
$$

Rearranging:

$$
\frac{p(\mathcal{D} \mid \theta_0)}{p(\mathcal{D} \mid H_1)} = \frac{p(\theta_0 \mid \mathcal{D}, H_1)}{p(\theta_0 \mid H_1)} = B_{01}
$$

### Two-Sample Comparison

**Testing equal means**: $H_0: \mu_1 = \mu_2$ vs $H_1: \mu_1 \neq \mu_2$

For Gaussian data with known variance, the Bayes factor has closed form. For unknown variance, numerical integration or approximations are needed.

### ANOVA: Multiple Group Comparison

**Testing all equal**: $H_0: \mu_1 = \mu_2 = \cdots = \mu_K$

The Bayes factor compares:
- $\mathcal{M}_0$: Single mean for all groups
- $\mathcal{M}_1$: Separate mean for each group

Intermediate models (some groups equal) require pairwise comparisons.

---

## Lindley's Paradox

### The Paradox Stated

**Lindley's paradox** (Lindley, 1957): For a fixed significance level $\alpha$ and large sample size $n$, a result can be:
- Statistically significant (p-value $< \alpha$)
- Yet Bayes factor strongly favors the null

### Mathematical Illustration

Consider testing $H_0: \mu = 0$ vs $H_1: \mu \neq 0$ for $\bar{x} \sim \mathcal{N}(\mu, \sigma^2/n)$.

**P-value approach**: Reject if $|\bar{x}| > z_{\alpha/2} \cdot \sigma/\sqrt{n}$

**Bayes factor** with $\mu \sim \mathcal{N}(0, \tau^2)$ under $H_1$:

$$
B_{01} = \sqrt{1 + n\tau^2/\sigma^2} \cdot \exp\left(-\frac{n\bar{x}^2}{2\sigma^2} \cdot \frac{n\tau^2/\sigma^2}{1 + n\tau^2/\sigma^2}\right)
$$

For fixed $\bar{x}$ at the significance boundary ($|\bar{x}| = z_{\alpha/2} \cdot \sigma/\sqrt{n}$):

As $n \to \infty$: p-value stays at $\alpha$, but $B_{01} \to \infty$ (favors null)!

### Resolution

The paradox arises because:
1. Fixed effect size relative to noise ($|\bar{x}|/(\sigma/\sqrt{n})$) is held constant
2. As $n \to \infty$, this means the actual effect $|\bar{x}|$ shrinks to zero
3. Bayes factor correctly detects that the effect is vanishingly small

**Lesson**: Bayes factors and p-values answer different questions. The Bayes factor asks "how probable is this data under each model?" — not "how extreme is this data assuming the null?"

---

## Sensitivity to Prior Specification

### Prior Dependence

Unlike posterior inference, Bayes factors are **highly sensitive** to prior specification:

$$
B_{12} = \frac{\int p(\mathcal{D} \mid \theta_1) \, p(\theta_1 \mid \mathcal{M}_1) \, d\theta_1}{\int p(\mathcal{D} \mid \theta_2) \, p(\theta_2 \mid \mathcal{M}_2) \, d\theta_2}
$$

The priors $p(\theta_k \mid \mathcal{M}_k)$ directly affect both numerator and denominator.

### Improper Priors

**Critical issue**: Improper priors make Bayes factors undefined!

If $\int p(\theta_k) \, d\theta_k = \infty$, then $p(\mathcal{D} \mid \mathcal{M}_k)$ is only determined up to an arbitrary constant.

**Consequence**: Never use improper priors for model comparison.

### Jeffreys-Lindley Paradox with Vague Priors

Using very diffuse priors (large variance) unfairly penalizes the alternative:

$$
p(\theta \mid H_1) = \mathcal{N}(0, 10^6) \quad \text{(very vague)}
$$

This prior assigns negligible probability to any reasonable effect size, so even strong effects favor $H_0$.

### Recommended Practices

**1. Default Bayes factors**: Use established, principled priors
- JZS (Jeffreys-Zellner-Siow) for $t$-tests
- BIC as rough approximation

**2. Prior sensitivity analysis**:
- Compute Bayes factors under several reasonable priors
- Report range of conclusions

**3. Data-dependent priors** (with caution):
- Fractional Bayes factors
- Intrinsic Bayes factors
- Posterior Bayes factors

---

## Default and Objective Bayes Factors

### JZS Bayes Factor

For comparing $H_0: \delta = 0$ vs $H_1: \delta \neq 0$ (standardized effect):

**Prior under $H_1$**: Cauchy prior (default scale $r = \sqrt{2}/2$)

$$
\delta \mid H_1 \sim \text{Cauchy}(0, r)
$$

**Properties**:
- Heavy tails allow large effects
- Scale parameter $r$ controls expected effect size
- Well-calibrated for psychological research

### Intrinsic Bayes Factors

Use part of the data to construct a proper reference prior:

$$
B_{12}^I = \frac{p(\mathcal{D}^{(-)} \mid \mathcal{D}^{(*)}, \mathcal{M}_1)}{p(\mathcal{D}^{(-)} \mid \mathcal{D}^{(*)}, \mathcal{M}_2)}
$$

where $\mathcal{D}^{(*)}$ is the "training" subset and $\mathcal{D}^{(-)}$ is the rest.

### Fractional Bayes Factors

Use a fraction of the likelihood to define the prior:

$$
p_b(\theta \mid \mathcal{M}) \propto p(\mathcal{D} \mid \theta, \mathcal{M})^b \, p_0(\theta \mid \mathcal{M})
$$

where $b < 1$ is the fraction and $p_0$ is the original prior.

**Advantage**: Reduces sensitivity to prior specification.

---

## Multiple Model Comparison

### Posterior Model Probabilities

For $K$ models $\mathcal{M}_1, \ldots, \mathcal{M}_K$ with prior probabilities $p(\mathcal{M}_k)$:

$$
p(\mathcal{M}_k \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathcal{M}_k) \, p(\mathcal{M}_k)}{\sum_{j=1}^K p(\mathcal{D} \mid \mathcal{M}_j) \, p(\mathcal{M}_j)}
$$

### Computing Posterior Probabilities from Bayes Factors

With equal prior probabilities $p(\mathcal{M}_k) = 1/K$:

$$
p(\mathcal{M}_k \mid \mathcal{D}) = \frac{B_{k1}}{\sum_{j=1}^K B_{j1}}
$$

where $B_{j1}$ is the Bayes factor of model $j$ versus model 1 (any reference model).

### Model Averaging

Instead of selecting a single model, average predictions:

$$
p(y^* \mid x^*, \mathcal{D}) = \sum_{k=1}^K p(y^* \mid x^*, \mathcal{D}, \mathcal{M}_k) \, p(\mathcal{M}_k \mid \mathcal{D})
$$

**Advantages**:
- Accounts for model uncertainty
- More robust predictions
- Often better calibration

### Pairwise vs Simultaneous Comparison

**Pairwise**: Compare models two at a time
- Simpler to compute
- Transitivity: $B_{13} = B_{12} \cdot B_{23}$

**Simultaneous**: Compare all models at once
- Requires specifying $p(\mathcal{M}_k)$ for all models
- More principled for model averaging

---

## Computational Methods

### Exact Computation (Conjugate Models)

For conjugate models, use the ratio of normalizing constants:

$$
B_{12} = \frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_2)}
$$

where each evidence is computed analytically.

### Savage-Dickey for Nested Models

For testing $H_0: \theta = \theta_0$ within $H_1$:

$$
B_{01} = \frac{p(\theta_0 \mid \mathcal{D}, H_1)}{p(\theta_0 \mid H_1)}
$$

Requires only the posterior and prior densities at one point.

### Bridge Sampling

Estimate the Bayes factor using samples from both posteriors:

$$
\hat{B}_{12} = \frac{\frac{1}{n_2} \sum_{j=1}^{n_2} h(\theta_2^{(j)}) \, p(\mathcal{D} \mid \theta_2^{(j)}, \mathcal{M}_1) \, p(\theta_2^{(j)} \mid \mathcal{M}_1)}{\frac{1}{n_1} \sum_{i=1}^{n_1} h(\theta_1^{(i)}) \, p(\mathcal{D} \mid \theta_1^{(i)}, \mathcal{M}_2) \, p(\theta_1^{(i)} \mid \mathcal{M}_2)}
$$

where $h(\cdot)$ is an optimal bridge function.

### Thermodynamic Integration

Use a path from model 1 to model 2:

$$
p_t(\theta) \propto p(\theta \mid \mathcal{M}_1)^{1-t} \, p(\theta \mid \mathcal{M}_2)^t \, p(\mathcal{D} \mid \theta)
$$

The log Bayes factor is:

$$
\log B_{12} = \int_0^1 \mathbb{E}_{p_t}\left[\log \frac{p(\theta \mid \mathcal{M}_1)}{p(\theta \mid \mathcal{M}_2)}\right] dt
$$

### Approximate Methods

**Laplace approximation**:
$$
\log B_{12} \approx \log p(\mathcal{D} \mid \hat{\theta}_1) - \log p(\mathcal{D} \mid \hat{\theta}_2) + \log p(\hat{\theta}_1) - \log p(\hat{\theta}_2) + \frac{d_1 - d_2}{2}\log(2\pi) - \frac{1}{2}\log\frac{|H_1|}{|H_2|}
$$

**BIC approximation**:
$$
\log B_{12} \approx \log p(\mathcal{D} \mid \hat{\theta}_1) - \log p(\mathcal{D} \mid \hat{\theta}_2) - \frac{d_1 - d_2}{2}\log n
$$

---

## Python Implementation

```python
"""
Bayes Factors: Complete Implementation

This module provides computation and interpretation of Bayes factors
for model comparison, including exact methods for conjugate models,
approximations, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, logsumexp
from scipy.integrate import quad
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Interpretation Utilities
# =============================================================================

def interpret_bayes_factor(log_bf: float, scale: str = 'kass_raftery') -> str:
    """
    Interpret log Bayes factor using standard scales.
    
    Parameters
    ----------
    log_bf : float
        Natural log of Bayes factor (ln B_12)
    scale : str
        'kass_raftery' or 'jeffreys'
    
    Returns
    -------
    str
        Interpretation of evidence strength
    """
    # Convert to log10 for standard scales
    log10_bf = log_bf / np.log(10)
    
    if scale == 'kass_raftery':
        if log10_bf > 2:
            return f"Decisive evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > 1:
            return f"Strong evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > 0.5:
            return f"Substantial evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > 0:
            return f"Weak evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > -0.5:
            return f"Weak evidence for M2 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > -1:
            return f"Substantial evidence for M2 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > -2:
            return f"Strong evidence for M2 (log10 BF = {log10_bf:.2f})"
        else:
            return f"Decisive evidence for M2 (log10 BF = {log10_bf:.2f})"
    
    elif scale == 'jeffreys':
        abs_log = abs(log10_bf)
        direction = "M1" if log10_bf > 0 else "M2"
        
        if abs_log > 2:
            strength = "Decisive"
        elif abs_log > 1.5:
            strength = "Very strong"
        elif abs_log > 1:
            strength = "Strong"
        elif abs_log > 0.5:
            strength = "Substantial"
        else:
            strength = "Barely worth mentioning"
        
        return f"{strength} evidence for {direction} (log10 BF = {log10_bf:.2f})"
    
    else:
        raise ValueError(f"Unknown scale: {scale}")


def log_bf_to_posterior_prob(log_bf: float, prior_odds: float = 1.0) -> float:
    """
    Convert log Bayes factor to posterior probability of M1.
    
    Parameters
    ----------
    log_bf : float
        Natural log of Bayes factor B_12
    prior_odds : float
        Prior odds p(M1)/p(M2), default 1 (equal priors)
    
    Returns
    -------
    float
        Posterior probability p(M1 | D)
    """
    log_posterior_odds = log_bf + np.log(prior_odds)
    # p(M1|D) = posterior_odds / (1 + posterior_odds)
    #         = 1 / (1 + 1/posterior_odds)
    #         = 1 / (1 + exp(-log_posterior_odds))
    return 1.0 / (1.0 + np.exp(-log_posterior_odds))


def posterior_probs_from_log_evidences(
    log_evidences: np.ndarray,
    prior_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute posterior model probabilities from log evidences.
    
    Parameters
    ----------
    log_evidences : array-like
        Log model evidences for each model
    prior_probs : array-like, optional
        Prior probabilities (uniform if not specified)
    
    Returns
    -------
    ndarray
        Posterior model probabilities
    """
    log_evidences = np.asarray(log_evidences)
    K = len(log_evidences)
    
    if prior_probs is None:
        log_priors = np.zeros(K) - np.log(K)  # Uniform
    else:
        log_priors = np.log(np.asarray(prior_probs))
    
    log_unnorm = log_evidences + log_priors
    log_norm = logsumexp(log_unnorm)
    
    return np.exp(log_unnorm - log_norm)


# =============================================================================
# Exact Bayes Factors for Conjugate Models
# =============================================================================

class BetaBernoulliModel:
    """Beta-Bernoulli model with exact evidence computation."""
    
    def __init__(self, alpha: float, beta: float):
        """
        Parameters
        ----------
        alpha, beta : float
            Beta prior parameters
        """
        self.alpha0 = alpha
        self.beta0 = beta
    
    def log_evidence(self, data: np.ndarray) -> float:
        """Compute log marginal likelihood."""
        n = len(data)
        s = np.sum(data)  # Successes
        f = n - s         # Failures
        
        alpha_n = self.alpha0 + s
        beta_n = self.beta0 + f
        
        # log B(alpha_n, beta_n) - log B(alpha_0, beta_0)
        log_ev = (
            gammaln(alpha_n) + gammaln(beta_n) - gammaln(alpha_n + beta_n)
            - gammaln(self.alpha0) - gammaln(self.beta0) + gammaln(self.alpha0 + self.beta0)
        )
        
        return log_ev


def bayes_factor_beta_bernoulli(
    data: np.ndarray,
    model1_params: Tuple[float, float],
    model2_params: Tuple[float, float]
) -> float:
    """
    Compute Bayes factor for two Beta-Bernoulli models.
    
    Parameters
    ----------
    data : ndarray
        Binary observations
    model1_params : tuple
        (alpha, beta) for model 1
    model2_params : tuple
        (alpha, beta) for model 2
    
    Returns
    -------
    float
        Log Bayes factor log(B_12)
    """
    model1 = BetaBernoulliModel(*model1_params)
    model2 = BetaBernoulliModel(*model2_params)
    
    return model1.log_evidence(data) - model2.log_evidence(data)


class GaussianKnownVarianceModel:
    """Gaussian model with known variance and Normal prior on mean."""
    
    def __init__(self, mu0: float, sigma0_sq: float, sigma_sq: float):
        """
        Parameters
        ----------
        mu0 : float
            Prior mean
        sigma0_sq : float
            Prior variance
        sigma_sq : float
            Known data variance
        """
        self.mu0 = mu0
        self.sigma0_sq = sigma0_sq
        self.sigma_sq = sigma_sq
    
    def log_evidence(self, data: np.ndarray) -> float:
        """Compute log marginal likelihood."""
        n = len(data)
        x_bar = np.mean(data)
        
        # Posterior precision
        tau0 = 1.0 / self.sigma0_sq
        tau = 1.0 / self.sigma_sq
        tau_n = tau0 + n * tau
        
        # Log evidence
        log_ev = (
            -0.5 * n * np.log(2 * np.pi * self.sigma_sq)
            + 0.5 * np.log(tau0 / tau_n)
            - 0.5 / self.sigma_sq * np.sum((data - x_bar)**2)
            - 0.5 * tau0 * n * tau / tau_n * (x_bar - self.mu0)**2
        )
        
        return log_ev


def bayes_factor_one_sample_t(
    data: np.ndarray,
    null_value: float = 0.0,
    prior_scale: float = np.sqrt(2) / 2
) -> Tuple[float, float]:
    """
    JZS Bayes factor for one-sample t-test.
    
    H0: delta = 0 (effect size is zero)
    H1: delta ~ Cauchy(0, prior_scale)
    
    Parameters
    ----------
    data : ndarray
        Observed data
    null_value : float
        Value under null hypothesis
    prior_scale : float
        Scale of Cauchy prior (default: sqrt(2)/2)
    
    Returns
    -------
    log_bf_01 : float
        Log Bayes factor in favor of null
    log_bf_10 : float
        Log Bayes factor in favor of alternative
    """
    n = len(data)
    t_stat = (np.mean(data) - null_value) / (np.std(data, ddof=1) / np.sqrt(n))
    
    # Integrate numerically
    def integrand(g):
        """Integrand for JZS Bayes factor."""
        if g <= 0:
            return 0
        return (
            (1 + g)**(-0.5)
            * (1 + t_stat**2 / ((1 + n * g) * (n - 1)))**(-(n) / 2)
            * (2 * np.pi)**(-0.5) * g**(-1.5)
            * np.exp(-1 / (2 * g * prior_scale**2))
        )
    
    # Numerical integration
    result, _ = quad(integrand, 0, np.inf)
    
    # Compare to null (no g, just t-distribution)
    null_density = stats.t.pdf(t_stat, df=n-1)
    
    log_bf_01 = np.log(null_density) - np.log(result) if result > 0 else np.inf
    log_bf_10 = -log_bf_01
    
    return log_bf_01, log_bf_10


# =============================================================================
# Savage-Dickey Density Ratio
# =============================================================================

def savage_dickey_ratio(
    posterior_density_at_null: float,
    prior_density_at_null: float
) -> float:
    """
    Compute Bayes factor using Savage-Dickey density ratio.
    
    B_01 = p(theta_0 | D, H1) / p(theta_0 | H1)
    
    Parameters
    ----------
    posterior_density_at_null : float
        Posterior density at the null value
    prior_density_at_null : float
        Prior density at the null value
    
    Returns
    -------
    float
        Log Bayes factor in favor of null
    """
    return np.log(posterior_density_at_null) - np.log(prior_density_at_null)


def savage_dickey_gaussian(
    data: np.ndarray,
    null_value: float,
    prior_mean: float,
    prior_var: float,
    known_var: float
) -> float:
    """
    Savage-Dickey for Gaussian model with Normal prior.
    
    Tests H0: mu = null_value vs H1: mu ~ N(prior_mean, prior_var)
    
    Parameters
    ----------
    data : ndarray
        Observed data
    null_value : float
        Value under null hypothesis
    prior_mean : float
        Prior mean under H1
    prior_var : float
        Prior variance under H1
    known_var : float
        Known data variance
    
    Returns
    -------
    float
        Log Bayes factor B_01
    """
    n = len(data)
    x_bar = np.mean(data)
    
    # Posterior parameters
    tau0 = 1.0 / prior_var
    tau = 1.0 / known_var
    tau_n = tau0 + n * tau
    mu_n = (tau0 * prior_mean + n * tau * x_bar) / tau_n
    var_n = 1.0 / tau_n
    
    # Densities at null value
    prior_density = stats.norm.pdf(null_value, prior_mean, np.sqrt(prior_var))
    posterior_density = stats.norm.pdf(null_value, mu_n, np.sqrt(var_n))
    
    return np.log(posterior_density) - np.log(prior_density)


# =============================================================================
# BIC Approximation to Bayes Factor
# =============================================================================

def bic_bayes_factor(
    log_lik1: float,
    log_lik2: float,
    k1: int,
    k2: int,
    n: int
) -> float:
    """
    BIC approximation to log Bayes factor.
    
    log B_12 ≈ (log L1 - k1/2 * log n) - (log L2 - k2/2 * log n)
    
    Parameters
    ----------
    log_lik1, log_lik2 : float
        Maximized log-likelihoods
    k1, k2 : int
        Number of parameters
    n : int
        Sample size
    
    Returns
    -------
    float
        Approximate log Bayes factor
    """
    bic1 = -2 * log_lik1 + k1 * np.log(n)
    bic2 = -2 * log_lik2 + k2 * np.log(n)
    
    # log B_12 ≈ -0.5 * (BIC1 - BIC2)
    return -0.5 * (bic1 - bic2)


# =============================================================================
# Lindley's Paradox Demonstration
# =============================================================================

def demonstrate_lindley_paradox(
    effect_size: float = 0.2,
    sample_sizes: np.ndarray = np.array([10, 50, 100, 500, 1000]),
    prior_var: float = 1.0,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demonstrate Lindley's paradox.
    
    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d)
    sample_sizes : ndarray
        Sample sizes to examine
    prior_var : float
        Prior variance under H1
    alpha : float
        Significance level
    
    Returns
    -------
    p_values : ndarray
        P-values for each sample size
    bayes_factors : ndarray
        Log Bayes factors B_01 for each sample size
    decisions : ndarray
        Classical decisions (True = reject null)
    """
    np.random.seed(42)
    
    p_values = []
    log_bfs = []
    decisions = []
    
    for n in sample_sizes:
        # Simulate data at fixed effect size
        true_mean = effect_size
        data = np.random.normal(true_mean, 1.0, n)
        
        # Classical t-test
        t_stat, p_val = stats.ttest_1samp(data, 0)
        p_values.append(p_val)
        decisions.append(p_val < alpha)
        
        # Bayes factor (simplified, assuming known variance = 1)
        x_bar = np.mean(data)
        se = 1.0 / np.sqrt(n)
        
        # B_01 for point null
        # Under H0: likelihood is N(0, 1/n)
        log_lik_h0 = stats.norm.logpdf(x_bar, 0, se)
        
        # Under H1 with N(0, prior_var) prior: marginal is N(0, prior_var + 1/n)
        marginal_var = prior_var + se**2
        log_lik_h1 = stats.norm.logpdf(x_bar, 0, np.sqrt(marginal_var))
        
        log_bf_01 = log_lik_h0 - log_lik_h1
        log_bfs.append(log_bf_01)
    
    return np.array(p_values), np.array(log_bfs), np.array(decisions)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_bayes_factor_interpretation(log_bfs: np.ndarray, labels: List[str]) -> plt.Figure:
    """
    Visualize Bayes factors with interpretation regions.
    
    Parameters
    ----------
    log_bfs : ndarray
        Log Bayes factors (natural log)
    labels : list
        Labels for each comparison
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to log10
    log10_bfs = log_bfs / np.log(10)
    
    # Background regions (Kass-Raftery scale)
    regions = [
        (-np.inf, -2, 'Decisive for M2', '#d62728', 0.2),
        (-2, -1, 'Strong for M2', '#ff7f0e', 0.2),
        (-1, -0.5, 'Substantial for M2', '#ffbb78', 0.2),
        (-0.5, 0.5, 'Inconclusive', '#d3d3d3', 0.3),
        (0.5, 1, 'Substantial for M1', '#98df8a', 0.2),
        (1, 2, 'Strong for M1', '#2ca02c', 0.2),
        (2, np.inf, 'Decisive for M1', '#1f77b4', 0.2),
    ]
    
    y_range = len(labels)
    for low, high, label, color, alpha in regions:
        low_plot = max(low, -4)
        high_plot = min(high, 4)
        ax.axvspan(low_plot, high_plot, alpha=alpha, color=color, label=label)
    
    # Plot Bayes factors
    y_pos = np.arange(len(labels))
    colors = ['#2ca02c' if bf > 0 else '#d62728' for bf in log10_bfs]
    
    ax.barh(y_pos, log10_bfs, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('log₁₀ Bayes Factor', fontsize=12)
    ax.set_title('Bayes Factor Comparison', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlim(-4, 4)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value annotations
    for i, (bf, label) in enumerate(zip(log10_bfs, labels)):
        x_pos = bf + 0.1 if bf > 0 else bf - 0.1
        ha = 'left' if bf > 0 else 'right'
        ax.annotate(f'{bf:.2f}', (x_pos, i), va='center', ha=ha, fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_lindley_paradox() -> plt.Figure:
    """
    Visualize Lindley's paradox.
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    sample_sizes = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000])
    p_values, log_bfs, decisions = demonstrate_lindley_paradox(
        effect_size=0.15,
        sample_sizes=sample_sizes,
        prior_var=1.0
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: P-values
    ax = axes[0]
    ax.semilogx(sample_sizes, p_values, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('P-value', fontsize=12)
    ax.set_title('Classical Test: P-values', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.5)
    
    # Right: Bayes factors
    ax = axes[1]
    log10_bfs = log_bfs / np.log(10)
    colors = ['#2ca02c' if bf > 0 else '#d62728' for bf in log10_bfs]
    ax.semilogx(sample_sizes, log10_bfs, 'o-', color='#ff7f0e', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='Substantial for H0')
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7, label='Substantial for H1')
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('log₁₀ Bayes Factor (B₀₁)', fontsize=12)
    ax.set_title('Bayesian Test: Bayes Factors', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(
        "Paradox: p-value significant\nbut BF favors null",
        xy=(500, log10_bfs[4]),
        xytext=(800, -0.8),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray')
    )
    
    plt.tight_layout()
    return fig


def plot_prior_sensitivity(
    data: np.ndarray,
    prior_scales: np.ndarray = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
) -> plt.Figure:
    """
    Demonstrate prior sensitivity of Bayes factors.
    
    Parameters
    ----------
    data : ndarray
        Observed data
    prior_scales : ndarray
        Prior standard deviations to try
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n = len(data)
    x_bar = np.mean(data)
    
    # Compute Bayes factors for each prior scale
    log_bfs_01 = []
    for scale in prior_scales:
        prior_var = scale**2
        se = 1.0 / np.sqrt(n)  # Assuming known variance = 1
        
        log_lik_h0 = stats.norm.logpdf(x_bar, 0, se)
        marginal_var = prior_var + se**2
        log_lik_h1 = stats.norm.logpdf(x_bar, 0, np.sqrt(marginal_var))
        
        log_bfs_01.append(log_lik_h0 - log_lik_h1)
    
    log_bfs_01 = np.array(log_bfs_01)
    log10_bfs = log_bfs_01 / np.log(10)
    
    # Left: Prior densities
    ax = axes[0]
    theta_range = np.linspace(-5, 5, 200)
    
    for i, scale in enumerate(prior_scales[::2]):  # Plot every other for clarity
        prior_pdf = stats.norm.pdf(theta_range, 0, scale)
        ax.plot(theta_range, prior_pdf, label=f'σ = {scale}', linewidth=2)
    
    ax.axvline(x=x_bar, color='red', linestyle='--', label=f'x̄ = {x_bar:.2f}')
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Prior Density', fontsize=12)
    ax.set_title('Prior Distributions for H₁', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Bayes factors vs prior scale
    ax = axes[1]
    ax.semilogx(prior_scales, log10_bfs, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='Substantial for H0')
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7, label='Substantial for H1')
    ax.set_xlabel('Prior Standard Deviation', fontsize=12)
    ax.set_ylabel('log₁₀ Bayes Factor (B₀₁)', fontsize=12)
    ax.set_title('Prior Sensitivity of Bayes Factor', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add interpretation
    ax.fill_between(prior_scales, -2, 0.5, alpha=0.1, color='red')
    ax.fill_between(prior_scales, 0.5, 3, alpha=0.1, color='green')
    
    plt.tight_layout()
    return fig


def plot_model_comparison_sequential(
    data: np.ndarray,
    model1: 'BetaBernoulliModel',
    model2: 'BetaBernoulliModel'
) -> plt.Figure:
    """
    Plot sequential evidence accumulation for two models.
    
    Parameters
    ----------
    data : ndarray
        Sequential observations
    model1, model2 : BetaBernoulliModel
        Models to compare
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    n = len(data)
    log_bfs = []
    
    for t in range(1, n + 1):
        data_t = data[:t]
        log_ev1 = model1.log_evidence(data_t)
        log_ev2 = model2.log_evidence(data_t)
        log_bfs.append(log_ev1 - log_ev2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    log10_bfs = np.array(log_bfs) / np.log(10)
    
    # Plot sequential Bayes factors
    ax.plot(range(1, n + 1), log10_bfs, 'o-', color='#1f77b4', 
            linewidth=2, markersize=4, alpha=0.7)
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    
    # Fill regions
    ax.axhspan(0.5, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.axhspan(ax.get_ylim()[0], -0.5, alpha=0.1, color='red')
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('log₁₀ Bayes Factor (M1 vs M2)', fontsize=12)
    ax.set_title('Sequential Evidence Accumulation', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Annotations
    ax.text(n * 0.9, 1.2, 'Strong for M1', fontsize=10, color='green')
    ax.text(n * 0.9, -1.2, 'Strong for M2', fontsize=10, color='red')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Demonstrations
# =============================================================================

def demo_basic_bayes_factors():
    """Demonstrate basic Bayes factor computation and interpretation."""
    
    print("=" * 70)
    print("BAYES FACTORS: BASIC DEMONSTRATION")
    print("=" * 70)
    
    # Generate coin flip data
    np.random.seed(42)
    true_theta = 0.65
    n = 100
    data = np.random.binomial(1, true_theta, n)
    s = data.sum()
    
    print(f"\nData: {s} successes in {n} trials (true θ = {true_theta})")
    
    # Compare models
    print("\n--- Comparing Prior Beliefs ---")
    
    # Model 1: Uniform prior (θ could be anything)
    # Model 2: Fair coin prior (θ concentrated around 0.5)
    model_uniform = BetaBernoulliModel(1, 1)
    model_fair = BetaBernoulliModel(50, 50)
    model_biased = BetaBernoulliModel(7, 3)  # Expects θ ≈ 0.7
    
    log_ev_uniform = model_uniform.log_evidence(data)
    log_ev_fair = model_fair.log_evidence(data)
    log_ev_biased = model_biased.log_evidence(data)
    
    print(f"\nLog evidences:")
    print(f"  Uniform prior (α=β=1):     {log_ev_uniform:.4f}")
    print(f"  Fair coin (α=β=50):        {log_ev_fair:.4f}")
    print(f"  Biased prior (α=7, β=3):   {log_ev_biased:.4f}")
    
    # Bayes factors
    log_bf_ub = log_ev_uniform - log_ev_biased
    log_bf_fb = log_ev_fair - log_ev_biased
    log_bf_uf = log_ev_uniform - log_ev_fair
    
    print(f"\nBayes factors:")
    print(f"  Uniform vs Biased:  {interpret_bayes_factor(log_bf_ub)}")
    print(f"  Fair vs Biased:     {interpret_bayes_factor(log_bf_fb)}")
    print(f"  Uniform vs Fair:    {interpret_bayes_factor(log_bf_uf)}")
    
    # Posterior model probabilities (equal prior odds)
    log_evs = [log_ev_uniform, log_ev_fair, log_ev_biased]
    probs = posterior_probs_from_log_evidences(log_evs)
    
    print(f"\nPosterior model probabilities (equal priors):")
    print(f"  Uniform: {probs[0]:.4f}")
    print(f"  Fair:    {probs[1]:.4f}")
    print(f"  Biased:  {probs[2]:.4f}")


def demo_savage_dickey():
    """Demonstrate Savage-Dickey density ratio."""
    
    print("\n" + "=" * 70)
    print("SAVAGE-DICKEY DENSITY RATIO")
    print("=" * 70)
    
    np.random.seed(123)
    
    # Generate data
    true_mu = 0.3  # Small effect
    n = 50
    data = np.random.normal(true_mu, 1.0, n)
    
    print(f"\nData: n={n}, mean={np.mean(data):.3f}, std={np.std(data):.3f}")
    print(f"True μ = {true_mu}")
    
    # Prior parameters
    prior_mean = 0.0
    prior_var = 1.0
    known_var = 1.0
    null_value = 0.0
    
    # Compute Savage-Dickey Bayes factor
    log_bf_01 = savage_dickey_gaussian(data, null_value, prior_mean, prior_var, known_var)
    
    print(f"\nTesting H0: μ = 0 vs H1: μ ~ N(0, 1)")
    print(f"Log B_01 = {log_bf_01:.4f}")
    print(f"Interpretation: {interpret_bayes_factor(log_bf_01)}")
    
    # Show the densities
    x_bar = np.mean(data)
    tau0 = 1.0 / prior_var
    tau = 1.0 / known_var
    tau_n = tau0 + n * tau
    mu_n = (tau0 * prior_mean + n * tau * x_bar) / tau_n
    var_n = 1.0 / tau_n
    
    prior_at_null = stats.norm.pdf(null_value, prior_mean, np.sqrt(prior_var))
    posterior_at_null = stats.norm.pdf(null_value, mu_n, np.sqrt(var_n))
    
    print(f"\nDensities at null value (μ = 0):")
    print(f"  Prior density:     {prior_at_null:.6f}")
    print(f"  Posterior density: {posterior_at_null:.6f}")
    print(f"  Ratio (B_01):      {posterior_at_null / prior_at_null:.6f}")


def demo_bic_approximation():
    """Demonstrate BIC approximation to Bayes factors."""
    
    print("\n" + "=" * 70)
    print("BIC APPROXIMATION TO BAYES FACTORS")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate polynomial regression data
    n = 100
    x = np.linspace(-2, 2, n)
    true_coeffs = [1, 0.5, -0.3]  # Quadratic
    y = true_coeffs[0] + true_coeffs[1] * x + true_coeffs[2] * x**2 + np.random.normal(0, 0.5, n)
    
    print(f"\nTrue model: y = 1 + 0.5x - 0.3x² + ε")
    print(f"Sample size: n = {n}")
    
    # Fit models of different degrees
    print("\n--- Model Comparison ---")
    
    results = []
    for degree in [1, 2, 3, 4, 5]:
        # Fit polynomial
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        
        # Compute log-likelihood (assuming known σ² = 0.25)
        residuals = y - y_pred
        sigma_sq = 0.25
        log_lik = -0.5 * n * np.log(2 * np.pi * sigma_sq) - 0.5 * np.sum(residuals**2) / sigma_sq
        
        # BIC
        k = degree + 1  # Number of parameters
        bic = -2 * log_lik + k * np.log(n)
        
        results.append({
            'degree': degree,
            'k': k,
            'log_lik': log_lik,
            'bic': bic
        })
        
        print(f"Degree {degree}: log-lik = {log_lik:.2f}, BIC = {bic:.2f}")
    
    # Compute approximate Bayes factors relative to degree 2 (true model)
    print("\n--- Approximate log Bayes factors vs Degree 2 ---")
    ref_idx = 1  # Degree 2
    
    for i, res in enumerate(results):
        if i != ref_idx:
            log_bf = bic_bayes_factor(
                results[ref_idx]['log_lik'], res['log_lik'],
                results[ref_idx]['k'], res['k'], n
            )
            print(f"Degree 2 vs Degree {res['degree']}: log B = {log_bf:.2f} "
                  f"({interpret_bayes_factor(log_bf).split('(')[0].strip()})")


def demo_lindley_paradox():
    """Demonstrate Lindley's paradox."""
    
    print("\n" + "=" * 70)
    print("LINDLEY'S PARADOX")
    print("=" * 70)
    
    print("\nEffect size: d = 0.15 (small)")
    print("Prior under H1: μ ~ N(0, 1)")
    
    sample_sizes = np.array([20, 50, 100, 500, 1000, 5000])
    p_values, log_bfs, decisions = demonstrate_lindley_paradox(
        effect_size=0.15,
        sample_sizes=sample_sizes
    )
    
    print("\n  n      p-value   Reject H0?   log₁₀ B₀₁   BF conclusion")
    print("-" * 65)
    
    for n, p, dec, log_bf in zip(sample_sizes, p_values, decisions, log_bfs):
        reject = "Yes" if dec else "No"
        log10_bf = log_bf / np.log(10)
        bf_interp = "H0" if log10_bf > 0.5 else "H1" if log10_bf < -0.5 else "Inconclusive"
        
        print(f"{n:5d}    {p:.4f}     {reject:4s}         {log10_bf:+.3f}        {bf_interp}")
    
    print("\n*** The paradox: Large samples show 'significant' p-values")
    print("    but Bayes factors favor the null hypothesis!")


def demo_sequential_evidence():
    """Demonstrate sequential evidence accumulation."""
    
    print("\n" + "=" * 70)
    print("SEQUENTIAL EVIDENCE ACCUMULATION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data from biased coin
    true_theta = 0.65
    n = 200
    data = np.random.binomial(1, true_theta, n)
    
    print(f"\nTrue θ = {true_theta}")
    print(f"Comparing: M1 (uniform) vs M2 (fair coin, α=β=20)")
    
    model1 = BetaBernoulliModel(1, 1)        # Uniform
    model2 = BetaBernoulliModel(20, 20)      # Fair coin belief
    
    # Track evidence at key points
    checkpoints = [10, 25, 50, 100, 150, 200]
    
    print("\nSequential Bayes factors (M1 vs M2):")
    print("  n      Successes   log₁₀ B₁₂   Interpretation")
    print("-" * 55)
    
    for t in checkpoints:
        data_t = data[:t]
        s = data_t.sum()
        log_bf = model1.log_evidence(data_t) - model2.log_evidence(data_t)
        log10_bf = log_bf / np.log(10)
        
        if log10_bf > 1:
            interp = "Strong for uniform"
        elif log10_bf > 0.5:
            interp = "Substantial for uniform"
        elif log10_bf < -1:
            interp = "Strong for fair"
        elif log10_bf < -0.5:
            interp = "Substantial for fair"
        else:
            interp = "Inconclusive"
        
        print(f"{t:4d}     {s:3d}         {log10_bf:+.3f}        {interp}")
    
    print("\n*** Evidence accumulates as more data arrives")


if __name__ == "__main__":
    demo_basic_bayes_factors()
    demo_savage_dickey()
    demo_bic_approximation()
    demo_lindley_paradox()
    demo_sequential_evidence()
```

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Definition** | $B_{12} = p(\mathcal{D} \mid \mathcal{M}_1) / p(\mathcal{D} \mid \mathcal{M}_2)$ |
| **Interpretation** | Factor by which data update relative model beliefs |
| **Log form** | $\log B_{12} = \log p(\mathcal{D} \mid \mathcal{M}_1) - \log p(\mathcal{D} \mid \mathcal{M}_2)$ |
| **Posterior odds** | $\text{Posterior odds} = B_{12} \times \text{Prior odds}$ |

### Interpretation Scale (Kass & Raftery)

| $\log_{10} B_{12}$ | $B_{12}$ | Evidence |
|-------------------|----------|----------|
| 0 to 0.5 | 1 to 3.2 | Barely worth mentioning |
| 0.5 to 1 | 3.2 to 10 | Substantial |
| 1 to 2 | 10 to 100 | Strong |
| > 2 | > 100 | Decisive |

### Key Properties

1. **Symmetry**: $B_{21} = 1/B_{12}$
2. **Transitivity**: $B_{13} = B_{12} \cdot B_{23}$
3. **Consistency**: Eventually favors true model
4. **Prior sensitivity**: Strong dependence on prior (unlike posterior)
5. **Occam's razor**: Built into model evidence

### Computation Methods

| Method | Applicability | Requirements |
|--------|--------------|--------------|
| Exact | Conjugate models | Closed-form evidence |
| Savage-Dickey | Nested models | Posterior at null |
| BIC | Large samples | MLE and parameter count |
| Bridge sampling | General | Samples from both posteriors |

### Important Caveats

1. **Lindley's paradox**: P-values and Bayes factors can disagree
2. **Prior sensitivity**: Results depend strongly on prior choice
3. **Improper priors**: Cannot be used for model comparison
4. **Model adequacy**: High BF ≠ good model, only better than alternative

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Model evidence | Ch13: Model Evidence | Bayes factor numerator/denominator |
| Information criteria | Ch13: Information Criteria | BIC approximates log BF |
| Conjugate models | Ch13: Distributions | Exact Bayes factors |
| Prior selection | Ch13: Foundations | Prior sensitivity |
| BNN comparison | Ch13: BNN | Architecture selection |

### Key References

- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
- Jeffreys, H. (1961). *Theory of Probability* (3rd ed.). Oxford University Press.
- Rouder, J. N., et al. (2009). Bayesian t tests. *Psychonomic Bulletin & Review*, 16(2), 225-237.
- Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values. *Psychonomic Bulletin & Review*, 14(5), 779-804.
