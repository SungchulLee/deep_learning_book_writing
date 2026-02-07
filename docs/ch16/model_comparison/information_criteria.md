# Information Criteria

**Information criteria** provide computationally tractable approximations for model comparison without requiring the full Bayesian evidence computation. These criteria balance goodness-of-fit against model complexity, offering principled trade-offs that connect frequentist model selection to Bayesian principles.

---

## Motivation: Practical Model Comparison

### The Challenge of Model Evidence

Computing the marginal likelihood (model evidence) is often intractable:

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta
$$

**Computational challenges**:
- High-dimensional integrals (no closed form)
- Requires proper priors (no improper priors)
- Monte Carlo estimates have high variance
- Sensitive to prior specification

**Information criteria** bypass these issues by using point estimates and asymptotic approximations.

### What Information Criteria Provide

1. **Computational efficiency**: Requires only maximum likelihood estimate
2. **Complexity penalty**: Automatically discourages overfitting
3. **Interpretability**: Clear decomposition into fit and complexity
4. **Asymptotic connection**: Approximates Bayesian quantities in large samples

---

## Akaike Information Criterion (AIC)

### Derivation from KL Divergence

The **Akaike Information Criterion** minimizes the expected Kullback-Leibler divergence between the true data-generating distribution $p_{\text{true}}$ and the fitted model.

**Setup**: Given data $\mathcal{D} = \{x_1, \ldots, x_n\}$ from unknown $p_{\text{true}}$:

$$
\text{KL}(p_{\text{true}} \| p_{\hat{\theta}}) = \int p_{\text{true}}(x) \log \frac{p_{\text{true}}(x)}{p(x \mid \hat{\theta})} \, dx
$$

Since $p_{\text{true}}$ is unknown, we estimate using the in-sample log-likelihood:

$$
\hat{\ell}(\hat{\theta}) = \frac{1}{n} \sum_{i=1}^n \log p(x_i \mid \hat{\theta})
$$

**Key insight**: The in-sample log-likelihood is an *optimistic* estimate of out-of-sample performance. Akaike showed that the bias equals approximately $k/n$, where $k$ is the number of parameters.

### Definition

$$
\boxed{\text{AIC} = -2 \log p(\mathcal{D} \mid \hat{\theta}) + 2k}
$$

where:
- $\hat{\theta}$ is the maximum likelihood estimate
- $k$ is the number of estimated parameters
- Lower AIC indicates a better model

**Equivalently**:

$$
\text{AIC} = -2 \hat{\ell} + 2k
$$

where $\hat{\ell} = \sum_{i=1}^n \log p(x_i \mid \hat{\theta})$ is the maximized log-likelihood.

### Derivation Details

**Step 1**: Define the quantity of interest as expected log predictive density:

$$
\text{elpd} = \mathbb{E}_{p_{\text{true}}}\left[\log p(\tilde{x} \mid \hat{\theta})\right]
$$

**Step 2**: Use in-sample approximation:

$$
\widehat{\text{elpd}}_{\text{in-sample}} = \frac{1}{n} \sum_{i=1}^n \log p(x_i \mid \hat{\theta})
$$

**Step 3**: Correct for optimism bias. Under regularity conditions:

$$
\mathbb{E}\left[\widehat{\text{elpd}}_{\text{in-sample}} - \text{elpd}\right] \approx \frac{k}{n}
$$

**Step 4**: Corrected estimate:

$$
\widehat{\text{elpd}}_{\text{AIC}} = \frac{1}{n} \sum_{i=1}^n \log p(x_i \mid \hat{\theta}) - \frac{k}{n}
$$

Multiplying by $-2n$ gives the standard AIC formula.

### Properties of AIC

**1. Asymptotic efficiency**: As $n \to \infty$, AIC selects the model that minimizes mean squared prediction error among candidate models.

**2. Not consistent**: AIC does not converge to the true model even as $n \to \infty$ (tends to select overly complex models).

**3. Equivalent models**: Models within 2 AIC units of the best have substantial support.

**AIC weights** for model averaging:

$$
w_i = \frac{\exp(-\Delta_i / 2)}{\sum_j \exp(-\Delta_j / 2)}
$$

where $\Delta_i = \text{AIC}_i - \text{AIC}_{\min}$.

### Corrected AIC (AICc)

For small samples, AIC's bias correction is insufficient. **AICc** provides a second-order correction:

$$
\boxed{\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}}
$$

**When to use**:
- Always safe to use AICc
- Essential when $n/k < 40$
- Converges to AIC as $n \to \infty$

**Derivation**: For linear regression with Gaussian errors, exact bias computation gives:

$$
\text{bias} = k + \frac{k(k+1)}{n - k - 1}
$$

---

## Bayesian Information Criterion (BIC)

### Derivation from Model Evidence

The **Bayesian Information Criterion** approximates the log model evidence using the Laplace approximation.

**Starting point**: Log marginal likelihood

$$
\log p(\mathcal{D} \mid \mathcal{M}) = \log \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

**Laplace approximation**: Expand around the MLE $\hat{\theta}$:

$$
\log p(\mathcal{D} \mid \theta) + \log p(\theta) \approx \log p(\mathcal{D} \mid \hat{\theta}) + \log p(\hat{\theta}) - \frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})
$$

where $H = -\nabla^2_\theta [\log p(\mathcal{D} \mid \theta) + \log p(\theta)]|_{\hat{\theta}}$ is the Hessian.

**Integration**: Using the Gaussian integral:

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) + \log p(\hat{\theta}) + \frac{k}{2}\log(2\pi) - \frac{1}{2}\log |H|
$$

**Asymptotic simplification**: For large $n$, $H \approx n \cdot I(\hat{\theta})$ where $I$ is the Fisher information. With diffuse priors:

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) - \frac{k}{2}\log n + O(1)
$$

### Definition

$$
\boxed{\text{BIC} = -2 \log p(\mathcal{D} \mid \hat{\theta}) + k \log n}
$$

where:
- $\hat{\theta}$ is the maximum likelihood estimate
- $k$ is the number of parameters
- $n$ is the sample size
- Lower BIC indicates a better model

**Connection to Bayes factor**: For models $\mathcal{M}_1$ and $\mathcal{M}_2$:

$$
\log B_{12} \approx -\frac{1}{2}(\text{BIC}_1 - \text{BIC}_2)
$$

### Properties of BIC

**1. Consistency**: As $n \to \infty$, BIC selects the true model with probability 1 (if it's among candidates).

**2. Stronger penalty**: For $n \geq 8$, BIC penalizes complexity more than AIC ($\log n > 2$).

**3. Prior-dependent**: BIC assumes specific (unit information) priors.

**Comparison with AIC**:

| Aspect | AIC | BIC |
|--------|-----|-----|
| Penalty per parameter | 2 | $\log n$ |
| Goal | Best prediction | True model |
| Consistency | No | Yes |
| Complexity preference | More complex | Simpler |
| Derivation | KL divergence | Model evidence |

### When BIC and AIC Disagree

**BIC favors simpler models when**:
- Sample size is large ($n > 8$)
- True model is among candidates
- Goal is identification

**AIC favors more complex models when**:
- Focus is on predictive accuracy
- True model may not be among candidates
- Approximation is acceptable

---

## Deviance Information Criterion (DIC)

### Motivation: Bayesian Model Complexity

For hierarchical Bayesian models, counting parameters is ambiguous:
- Are random effects "parameters"?
- Shrinkage effectively reduces complexity

**DIC** addresses this by defining an effective number of parameters.

### Definition

**Deviance**: $D(\theta) = -2 \log p(\mathcal{D} \mid \theta)$

**Effective number of parameters**:

$$
p_D = \overline{D(\theta)} - D(\bar{\theta})
$$

where:
- $\overline{D(\theta)} = \mathbb{E}_{\theta \mid \mathcal{D}}[D(\theta)]$ is the posterior mean deviance
- $D(\bar{\theta}) = D(\mathbb{E}_{\theta \mid \mathcal{D}}[\theta])$ is the deviance at the posterior mean

**DIC**:

$$
\boxed{\text{DIC} = \overline{D(\theta)} + p_D = 2\overline{D(\theta)} - D(\bar{\theta})}
$$

### Interpretation of $p_D$

The effective number of parameters $p_D$ measures how much the data informs the posterior:

- $p_D \approx k$ when prior is uninformative
- $p_D < k$ when prior constrains parameters (shrinkage)
- $p_D$ can even be negative (rare, indicates problems)

**For a hierarchical model** with groups:
- Fixed effects contribute ~1 each
- Random effects contribute < 1 each (due to shrinkage)

### Alternative Definition (Gelman)

**Variance-based $p_D$**:

$$
p_D = \frac{1}{2} \text{Var}_{\theta \mid \mathcal{D}}[D(\theta)]
$$

This is often more stable and always positive.

### Limitations of DIC

1. **Posterior mean may not be representative**: For multimodal posteriors, $\bar{\theta}$ may be in low-probability regions.

2. **Not invariant to parameterization**: Different parameterizations give different DIC values.

3. **No clear interpretation**: Unlike AIC/BIC, no asymptotic theory justifies DIC.

4. **Can be negative**: $p_D$ can be negative, which is uninterpretable.

---

## Widely Applicable Information Criterion (WAIC)

### Motivation: Fully Bayesian Alternative

**WAIC** (also called Watanabe-Akaike IC) provides a fully Bayesian approach that:
- Uses the entire posterior, not just point estimates
- Estimates out-of-sample predictive accuracy
- Works for singular models where BIC fails

### Definition

**Log pointwise predictive density (lppd)**:

$$
\text{lppd} = \sum_{i=1}^n \log p(y_i \mid \mathcal{D}) = \sum_{i=1}^n \log \int p(y_i \mid \theta) \, p(\theta \mid \mathcal{D}) \, d\theta
$$

Estimated using posterior samples $\{\theta^{(s)}\}_{s=1}^S$:

$$
\widehat{\text{lppd}} = \sum_{i=1}^n \log \left( \frac{1}{S} \sum_{s=1}^S p(y_i \mid \theta^{(s)}) \right)
$$

**Effective number of parameters**:

$$
p_{\text{WAIC}} = \sum_{i=1}^n \text{Var}_{\theta \mid \mathcal{D}}[\log p(y_i \mid \theta)]
$$

Estimated as:

$$
\hat{p}_{\text{WAIC}} = \sum_{i=1}^n \widehat{\text{Var}}_s[\log p(y_i \mid \theta^{(s)})]
$$

**WAIC**:

$$
\boxed{\text{WAIC} = -2 \left( \widehat{\text{lppd}} - \hat{p}_{\text{WAIC}} \right)}
$$

### Properties of WAIC

**1. Asymptotically equivalent to cross-validation**: WAIC $\approx$ leave-one-out cross-validation as $n \to \infty$.

**2. Applies to singular models**: Works when BIC fails (e.g., mixture models, neural networks).

**3. Fully Bayesian**: Accounts for posterior uncertainty.

**4. Pointwise computation**: Can identify influential observations.

### Comparison with DIC

| Aspect | DIC | WAIC |
|--------|-----|------|
| Point estimate | Posterior mean | Full posterior |
| Complexity | $\bar{D} - D(\bar{\theta})$ | Variance-based |
| Invariance | Parameterization-dependent | Invariant |
| Singular models | May fail | Works |

---

## Leave-One-Out Cross-Validation (LOO-CV)

### Definition

The gold standard for predictive assessment is **leave-one-out cross-validation**:

$$
\text{LOO-CV} = \sum_{i=1}^n \log p(y_i \mid \mathcal{D}_{-i})
$$

where $\mathcal{D}_{-i}$ denotes the data with observation $i$ removed.

**Exact computation** requires fitting $n$ models, which is expensive.

### Pareto Smoothed Importance Sampling (PSIS-LOO)

**Key idea**: Approximate LOO using importance sampling:

$$
p(y_i \mid \mathcal{D}_{-i}) \approx \frac{\sum_s w_i^{(s)} p(y_i \mid \theta^{(s)})}{\sum_s w_i^{(s)}}
$$

where $w_i^{(s)} \propto 1/p(y_i \mid \theta^{(s)})$ are importance weights.

**Problem**: Importance weights have high variance.

**Solution**: **Pareto smoothing** stabilizes the weights:

1. Fit a generalized Pareto distribution to the largest weights
2. Replace extreme weights with expected order statistics
3. Use stabilized weights for estimation

**Diagnostic**: The shape parameter $\hat{k}$ indicates reliability:
- $\hat{k} < 0.5$: Excellent, LOO estimate reliable
- $0.5 < \hat{k} < 0.7$: Good, slight bias possible
- $0.7 < \hat{k} < 1$: Fair, consider exact LOO for these points
- $\hat{k} > 1$: Bad, importance sampling fails

### Expected Log Predictive Density (ELPD)

All information criteria estimate the **expected log predictive density**:

$$
\text{elpd} = \sum_{i=1}^n \int p_{\text{true}}(\tilde{y}_i) \log p(\tilde{y}_i \mid \mathcal{D}) \, d\tilde{y}_i
$$

**Relationships**:

| Criterion | Estimate of |
|-----------|-------------|
| AIC | $-2 \cdot \text{elpd}$ (large sample) |
| BIC | $2 \log p(\mathcal{D} \mid \mathcal{M})$ |
| DIC | $-2 \cdot \text{elpd}$ (Bayesian plug-in) |
| WAIC | $-2 \cdot \text{elpd}$ (fully Bayesian) |
| LOO-CV | Direct estimate of elpd |

---

## Relationships and Connections

### Asymptotic Relationships

**For regular models with large $n$**:

$$
\text{WAIC} \approx \text{LOO-CV} \approx \text{DIC}
$$

**BIC approximates log Bayes factor**:

$$
\text{BIC} \approx -2 \log p(\mathcal{D} \mid \mathcal{M}) + \text{constant}
$$

### Penalty Comparison

For models with $k$ parameters and $n$ observations:

| Criterion | Complexity Penalty |
|-----------|-------------------|
| AIC | $2k$ |
| AICc | $2k + \frac{2k(k+1)}{n-k-1}$ |
| BIC | $k \log n$ |
| DIC | $2 p_D$ |
| WAIC | $2 p_{\text{WAIC}}$ |

**Crossover point**: AIC and BIC penalties equal when $n = e^2 \approx 7.4$.

### When to Use Each Criterion

**Use AIC/AICc when**:
- Goal is prediction
- True model may not be among candidates
- Want to compare non-nested models

**Use BIC when**:
- Goal is to identify the true model
- Believe true model is among candidates
- Want approximation to Bayes factor

**Use DIC when**:
- Working with hierarchical Bayesian models
- Need computationally cheap criterion
- MCMC samples already available

**Use WAIC/LOO-CV when**:
- Need fully Bayesian assessment
- Working with complex or singular models
- Want reliable uncertainty quantification

---

## Practical Considerations

### Model Selection vs Model Averaging

**Model selection**: Choose single best model
- Ignores model uncertainty
- Simpler for interpretation
- May be overconfident

**Model averaging**: Weight models by criterion

$$
p(\tilde{y} \mid \mathcal{D}) = \sum_k w_k \cdot p(\tilde{y} \mid \mathcal{D}, \mathcal{M}_k)
$$

Weights using AIC:

$$
w_k = \frac{\exp(-\text{AIC}_k / 2)}{\sum_j \exp(-\text{AIC}_j / 2)}
$$

Weights using BIC (pseudo-Bayes factors):

$$
w_k = \frac{\exp(-\text{BIC}_k / 2)}{\sum_j \exp(-\text{BIC}_j / 2)}
$$

### Reporting and Interpretation

**Standard practice**:
- Report multiple criteria (don't cherry-pick)
- Report differences from best model ($\Delta$)
- Consider uncertainty in criterion estimates

**Rules of thumb for $\Delta$ (AIC/WAIC)**:
- $\Delta < 2$: Substantial support
- $2 < \Delta < 7$: Considerably less support
- $\Delta > 10$: Essentially no support

### Common Pitfalls

**1. Comparing incomparable models**:
- Must use same data for all comparisons
- Cannot compare models with different response variables

**2. Ignoring uncertainty**:
- IC differences have estimation error
- Small differences may not be meaningful

**3. Over-relying on any single criterion**:
- Different criteria answer different questions
- Sensitivity analysis is important

**4. Forgetting computational shortcuts**:
- AIC requires only MLE
- BIC requires only MLE
- WAIC requires MCMC samples

---

## Python Implementation

```python
"""
Information Criteria: Complete Implementation

This module provides computation of AIC, AICc, BIC, DIC, WAIC, and LOO-CV
for model comparison, along with visualization and model averaging tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp, gammaln
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Core Information Criteria Functions
# =============================================================================

def aic(log_likelihood: float, k: int) -> float:
    """
    Compute Akaike Information Criterion.
    
    Parameters
    ----------
    log_likelihood : float
        Maximized log-likelihood
    k : int
        Number of estimated parameters
    
    Returns
    -------
    float
        AIC value (lower is better)
    """
    return -2 * log_likelihood + 2 * k


def aicc(log_likelihood: float, k: int, n: int) -> float:
    """
    Compute corrected AIC (AICc) for small samples.
    
    Parameters
    ----------
    log_likelihood : float
        Maximized log-likelihood
    k : int
        Number of estimated parameters
    n : int
        Sample size
    
    Returns
    -------
    float
        AICc value (lower is better)
    
    Raises
    ------
    ValueError
        If n - k - 1 <= 0
    """
    if n - k - 1 <= 0:
        raise ValueError(f"Sample size n={n} too small for k={k} parameters")
    
    base_aic = aic(log_likelihood, k)
    correction = 2 * k * (k + 1) / (n - k - 1)
    
    return base_aic + correction


def bic(log_likelihood: float, k: int, n: int) -> float:
    """
    Compute Bayesian Information Criterion.
    
    Parameters
    ----------
    log_likelihood : float
        Maximized log-likelihood
    k : int
        Number of estimated parameters
    n : int
        Sample size
    
    Returns
    -------
    float
        BIC value (lower is better)
    """
    return -2 * log_likelihood + k * np.log(n)


def bic_to_log_bayes_factor(bic1: float, bic2: float) -> float:
    """
    Convert BIC difference to approximate log Bayes factor.
    
    Parameters
    ----------
    bic1, bic2 : float
        BIC values for models 1 and 2
    
    Returns
    -------
    float
        Approximate log B_12 (model 1 vs model 2)
    """
    return -0.5 * (bic1 - bic2)


# =============================================================================
# Bayesian Information Criteria (require posterior samples)
# =============================================================================

def dic(
    log_lik_samples: np.ndarray,
    log_lik_at_mean: float
) -> Tuple[float, float]:
    """
    Compute Deviance Information Criterion.
    
    Parameters
    ----------
    log_lik_samples : ndarray
        Log-likelihood evaluated at each posterior sample
    log_lik_at_mean : float
        Log-likelihood evaluated at posterior mean of parameters
    
    Returns
    -------
    dic : float
        DIC value
    p_d : float
        Effective number of parameters
    """
    # Mean deviance
    mean_deviance = -2 * np.mean(log_lik_samples)
    
    # Deviance at posterior mean
    deviance_at_mean = -2 * log_lik_at_mean
    
    # Effective parameters
    p_d = mean_deviance - deviance_at_mean
    
    # DIC
    dic_value = mean_deviance + p_d
    
    return dic_value, p_d


def dic_alternative(log_lik_samples: np.ndarray) -> Tuple[float, float]:
    """
    Compute DIC using variance-based effective parameters (Gelman version).
    
    Parameters
    ----------
    log_lik_samples : ndarray
        Log-likelihood evaluated at each posterior sample
    
    Returns
    -------
    dic : float
        DIC value
    p_d : float
        Effective number of parameters (variance-based)
    """
    # Mean deviance
    mean_deviance = -2 * np.mean(log_lik_samples)
    
    # Variance-based effective parameters
    p_d = 0.5 * np.var(-2 * log_lik_samples)
    
    # DIC
    dic_value = mean_deviance + p_d
    
    return dic_value, p_d


def waic(
    log_lik_matrix: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute Widely Applicable Information Criterion.
    
    Parameters
    ----------
    log_lik_matrix : ndarray of shape (n_samples, n_observations)
        Log-likelihood for each posterior sample and observation
        Entry [s, i] = log p(y_i | theta^(s))
    
    Returns
    -------
    waic : float
        WAIC value
    lppd : float
        Log pointwise predictive density
    p_waic : float
        Effective number of parameters
    """
    S, n = log_lik_matrix.shape
    
    # Log pointwise predictive density
    # lppd = sum_i log( mean_s p(y_i | theta^s) )
    lppd = np.sum(logsumexp(log_lik_matrix, axis=0) - np.log(S))
    
    # Effective parameters (pointwise variance)
    # p_waic = sum_i Var_s(log p(y_i | theta^s))
    p_waic = np.sum(np.var(log_lik_matrix, axis=0))
    
    # WAIC
    waic_value = -2 * (lppd - p_waic)
    
    return waic_value, lppd, p_waic


def waic_pointwise(
    log_lik_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pointwise WAIC contributions.
    
    Parameters
    ----------
    log_lik_matrix : ndarray of shape (n_samples, n_observations)
        Log-likelihood for each posterior sample and observation
    
    Returns
    -------
    elpd_i : ndarray
        Pointwise expected log predictive density
    p_waic_i : ndarray
        Pointwise effective parameters
    waic_i : ndarray
        Pointwise WAIC contributions
    """
    S, n = log_lik_matrix.shape
    
    # Pointwise lppd
    lppd_i = logsumexp(log_lik_matrix, axis=0) - np.log(S)
    
    # Pointwise p_waic
    p_waic_i = np.var(log_lik_matrix, axis=0)
    
    # Pointwise WAIC
    elpd_i = lppd_i - p_waic_i
    waic_i = -2 * elpd_i
    
    return elpd_i, p_waic_i, waic_i


# =============================================================================
# Leave-One-Out Cross-Validation
# =============================================================================

def psis_loo(
    log_lik_matrix: np.ndarray,
    return_diagnostics: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """
    Compute LOO-CV using Pareto Smoothed Importance Sampling.
    
    This is a simplified implementation. For production use, 
    consider the `arviz` package.
    
    Parameters
    ----------
    log_lik_matrix : ndarray of shape (n_samples, n_observations)
        Log-likelihood for each posterior sample and observation
    return_diagnostics : bool
        Whether to return Pareto k diagnostics
    
    Returns
    -------
    loo : float
        LOO-CV estimate (elpd scale, not deviance)
    p_loo : float
        Effective number of parameters
    k_hat : ndarray (if return_diagnostics=True)
        Pareto k estimates for each observation
    """
    S, n = log_lik_matrix.shape
    
    elpd_loo = np.zeros(n)
    k_hat = np.zeros(n)
    
    for i in range(n):
        # Raw importance weights: 1 / p(y_i | theta)
        log_weights = -log_lik_matrix[:, i]
        
        # Stabilize and apply Pareto smoothing
        # (Simplified: just stabilize the weights)
        log_weights_centered = log_weights - np.max(log_weights)
        weights = np.exp(log_weights_centered)
        
        # Fit Pareto to largest weights (simplified)
        M = max(int(np.sqrt(S)), 10)
        sorted_weights = np.sort(weights)[-M:]
        
        # Estimate Pareto k (simplified moment estimator)
        if sorted_weights[-1] > sorted_weights[0]:
            log_ratios = np.log(sorted_weights[-1] / sorted_weights[:-1])
            k_hat[i] = np.mean(log_ratios)
        else:
            k_hat[i] = 0
        
        # Normalize weights
        weights_normalized = weights / np.sum(weights)
        
        # LOO predictive density
        log_lik_i = log_lik_matrix[:, i]
        elpd_loo[i] = np.log(np.sum(weights_normalized * np.exp(log_lik_i)))
    
    loo = np.sum(elpd_loo)
    
    # Effective parameters (approximation)
    lppd = np.sum(logsumexp(log_lik_matrix, axis=0) - np.log(S))
    p_loo = lppd - loo
    
    if return_diagnostics:
        return loo, p_loo, k_hat
    return loo, p_loo


def exact_loo_cv(
    y: np.ndarray,
    X: np.ndarray,
    fit_func,
    predict_log_lik_func
) -> float:
    """
    Compute exact LOO-CV by refitting.
    
    Parameters
    ----------
    y : ndarray
        Response variable
    X : ndarray
        Design matrix
    fit_func : callable
        Function that fits model, returns parameters
    predict_log_lik_func : callable
        Function(y_i, X_i, params) -> log p(y_i | X_i, params)
    
    Returns
    -------
    float
        LOO-CV (sum of log predictive densities)
    """
    n = len(y)
    elpd = 0.0
    
    for i in range(n):
        # Leave out observation i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        y_train = y[mask]
        X_train = X[mask] if X.ndim > 1 else X[mask]
        
        # Fit on n-1 observations
        params = fit_func(y_train, X_train)
        
        # Evaluate on held-out observation
        elpd += predict_log_lik_func(y[i], X[i:i+1] if X.ndim > 1 else X[i], params)
    
    return elpd


# =============================================================================
# Model Comparison Utilities
# =============================================================================

@dataclass
class ModelComparison:
    """Container for model comparison results."""
    names: List[str]
    aic: np.ndarray
    aicc: np.ndarray
    bic: np.ndarray
    n_params: np.ndarray
    log_lik: np.ndarray
    n_obs: int
    
    def summary(self) -> str:
        """Return formatted comparison table."""
        lines = []
        lines.append("Model Comparison Summary")
        lines.append("=" * 70)
        lines.append(f"{'Model':<20} {'k':>5} {'LL':>12} {'AIC':>10} {'AICc':>10} {'BIC':>10}")
        lines.append("-" * 70)
        
        # Sort by AIC
        order = np.argsort(self.aic)
        
        for i in order:
            lines.append(
                f"{self.names[i]:<20} {self.n_params[i]:>5} "
                f"{self.log_lik[i]:>12.2f} {self.aic[i]:>10.2f} "
                f"{self.aicc[i]:>10.2f} {self.bic[i]:>10.2f}"
            )
        
        lines.append("-" * 70)
        
        # Delta values
        lines.append("\nDifferences from best model (AIC):")
        delta_aic = self.aic - np.min(self.aic)
        delta_bic = self.bic - np.min(self.bic)
        
        for i in order:
            lines.append(
                f"  {self.names[i]:<20} ΔAIC = {delta_aic[i]:>7.2f}, "
                f"ΔBIC = {delta_bic[i]:>7.2f}"
            )
        
        return "\n".join(lines)
    
    def weights(self, criterion: str = 'aic') -> np.ndarray:
        """
        Compute model weights (Akaike weights or pseudo-Bayes factors).
        
        Parameters
        ----------
        criterion : str
            'aic' or 'bic'
        
        Returns
        -------
        ndarray
            Model weights (sum to 1)
        """
        if criterion == 'aic':
            values = self.aic
        elif criterion == 'bic':
            values = self.bic
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        delta = values - np.min(values)
        weights = np.exp(-0.5 * delta)
        
        return weights / np.sum(weights)


def compare_models(
    models: Dict[str, Tuple[float, int]],
    n_obs: int
) -> ModelComparison:
    """
    Compare multiple models using information criteria.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping model names to (log_likelihood, n_params) tuples
    n_obs : int
        Number of observations
    
    Returns
    -------
    ModelComparison
        Comparison results
    """
    names = list(models.keys())
    n_models = len(names)
    
    log_liks = np.array([models[name][0] for name in names])
    n_params = np.array([models[name][1] for name in names])
    
    aic_vals = np.array([aic(ll, k) for ll, k in zip(log_liks, n_params)])
    aicc_vals = np.array([aicc(ll, k, n_obs) for ll, k in zip(log_liks, n_params)])
    bic_vals = np.array([bic(ll, k, n_obs) for ll, k in zip(log_liks, n_params)])
    
    return ModelComparison(
        names=names,
        aic=aic_vals,
        aicc=aicc_vals,
        bic=bic_vals,
        n_params=n_params,
        log_lik=log_liks,
        n_obs=n_obs
    )


# =============================================================================
# Linear Regression Example
# =============================================================================

class LinearRegressionIC:
    """
    Linear regression with information criteria computation.
    
    Model: y = X @ beta + epsilon, epsilon ~ N(0, sigma^2)
    """
    
    def __init__(self):
        self.beta_hat = None
        self.sigma_hat = None
        self.n = None
        self.k = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit linear regression via MLE.
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Design matrix
        y : ndarray of shape (n,)
            Response
        """
        self.n = len(y)
        self.k = X.shape[1] + 1  # +1 for sigma
        
        # OLS for beta
        self.beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # MLE for sigma
        residuals = y - X @ self.beta_hat
        self.sigma_hat = np.sqrt(np.sum(residuals**2) / self.n)
    
    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute log-likelihood at MLE."""
        residuals = y - X @ self.beta_hat
        ll = (
            -0.5 * self.n * np.log(2 * np.pi)
            - self.n * np.log(self.sigma_hat)
            - 0.5 * np.sum(residuals**2) / self.sigma_hat**2
        )
        return ll
    
    def aic(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute AIC."""
        return aic(self.log_likelihood(X, y), self.k)
    
    def aicc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute AICc."""
        return aicc(self.log_likelihood(X, y), self.k, self.n)
    
    def bic(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute BIC."""
        return bic(self.log_likelihood(X, y), self.k, self.n)


# =============================================================================
# Bayesian Linear Regression with WAIC
# =============================================================================

class BayesianLinearRegression:
    """
    Bayesian linear regression with conjugate prior.
    
    Model: y = X @ beta + epsilon, epsilon ~ N(0, sigma^2)
    Prior: beta | sigma^2 ~ N(0, sigma^2 * g * (X'X)^{-1})  [g-prior]
           sigma^2 ~ Inv-Gamma(a0/2, b0/2)
    """
    
    def __init__(self, g: float = 100.0, a0: float = 1.0, b0: float = 1.0):
        """
        Parameters
        ----------
        g : float
            g-prior scale parameter
        a0, b0 : float
            Inverse-gamma prior parameters for sigma^2
        """
        self.g = g
        self.a0 = a0
        self.b0 = b0
        
        # Posterior parameters (set after fit)
        self.beta_mean = None
        self.beta_cov = None
        self.a_n = None
        self.b_n = None
        self.X = None
        self.y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Compute posterior parameters."""
        self.X = X
        self.y = y
        n, p = X.shape
        
        # OLS estimate
        XtX = X.T @ X
        beta_ols = np.linalg.solve(XtX, X.T @ y)
        
        # Posterior for beta (conditional on sigma^2)
        # Mean: g/(1+g) * beta_ols
        self.beta_mean = self.g / (1 + self.g) * beta_ols
        
        # Residual sum of squares
        residuals = y - X @ beta_ols
        SSR = np.sum(residuals**2)
        
        # Posterior for sigma^2
        self.a_n = self.a0 + n
        self.b_n = self.b0 + SSR + beta_ols.T @ XtX @ beta_ols / (1 + self.g)
        
        # Posterior covariance of beta (marginalizing over sigma^2)
        sigma2_mean = self.b_n / (self.a_n - 2) if self.a_n > 2 else self.b_n / self.a_n
        self.beta_cov = sigma2_mean * self.g / (1 + self.g) * np.linalg.inv(XtX)
    
    def sample_posterior(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from posterior distribution.
        
        Returns
        -------
        beta_samples : ndarray of shape (n_samples, p)
        sigma2_samples : ndarray of shape (n_samples,)
        """
        p = len(self.beta_mean)
        
        # Sample sigma^2 from inverse-gamma
        sigma2_samples = stats.invgamma.rvs(
            self.a_n / 2,
            scale=self.b_n / 2,
            size=n_samples
        )
        
        # Sample beta | sigma^2 from multivariate normal
        XtX = self.X.T @ self.X
        beta_samples = np.zeros((n_samples, p))
        
        for s in range(n_samples):
            cov_s = sigma2_samples[s] * self.g / (1 + self.g) * np.linalg.inv(XtX)
            beta_samples[s] = np.random.multivariate_normal(self.beta_mean, cov_s)
        
        return beta_samples, sigma2_samples
    
    def compute_waic(self, n_samples: int = 1000) -> Tuple[float, float, float]:
        """
        Compute WAIC for this model.
        
        Returns
        -------
        waic_value : float
        lppd : float
        p_waic : float
        """
        beta_samples, sigma2_samples = self.sample_posterior(n_samples)
        
        n = len(self.y)
        log_lik_matrix = np.zeros((n_samples, n))
        
        for s in range(n_samples):
            mu = self.X @ beta_samples[s]
            sigma = np.sqrt(sigma2_samples[s])
            log_lik_matrix[s] = stats.norm.logpdf(self.y, mu, sigma)
        
        return waic(log_lik_matrix)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_ic_comparison(
    comparison: ModelComparison,
    criterion: str = 'all',
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Visualize information criteria comparison.
    
    Parameters
    ----------
    comparison : ModelComparison
        Results from compare_models
    criterion : str
        'aic', 'bic', 'all'
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: IC values
    ax = axes[0]
    x = np.arange(len(comparison.names))
    width = 0.25
    
    if criterion in ['aic', 'all']:
        ax.bar(x - width, comparison.aic, width, label='AIC', alpha=0.8)
    if criterion in ['bic', 'all']:
        ax.bar(x, comparison.bic, width, label='BIC', alpha=0.8)
    if criterion in ['aicc', 'all']:
        ax.bar(x + width, comparison.aicc, width, label='AICc', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.names, rotation=45, ha='right')
    ax.set_ylabel('Information Criterion')
    ax.legend()
    ax.set_title('Model Comparison')
    
    # Right: Model weights
    ax = axes[1]
    aic_weights = comparison.weights('aic')
    bic_weights = comparison.weights('bic')
    
    ax.bar(x - width/2, aic_weights, width, label='AIC weights', alpha=0.8)
    ax.bar(x + width/2, bic_weights, width, label='BIC weights', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.names, rotation=45, ha='right')
    ax.set_ylabel('Model Weight')
    ax.legend()
    ax.set_title('Model Weights')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


def plot_penalty_comparison(
    max_params: int = 20,
    sample_sizes: List[int] = [10, 50, 100, 500]
):
    """
    Compare penalty terms of AIC vs BIC across sample sizes.
    
    Parameters
    ----------
    max_params : int
        Maximum number of parameters to plot
    sample_sizes : list
        Sample sizes to compare
    """
    k = np.arange(1, max_params + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Penalty per parameter
    ax = axes[0]
    ax.axhline(2, color='blue', linestyle='--', label='AIC: 2', linewidth=2)
    
    for n in sample_sizes:
        ax.axhline(np.log(n), linestyle='-', alpha=0.7, 
                   label=f'BIC (n={n}): {np.log(n):.2f}')
    
    ax.set_xlabel('(Reference)')
    ax.set_ylabel('Penalty per Parameter')
    ax.legend()
    ax.set_title('Penalty per Parameter')
    ax.set_ylim(0, 8)
    
    # Right: Total penalty
    ax = axes[1]
    ax.plot(k, 2 * k, 'b--', label='AIC: 2k', linewidth=2)
    
    for n in sample_sizes:
        ax.plot(k, k * np.log(n), label=f'BIC (n={n}): k·log({n})', alpha=0.7)
    
    ax.set_xlabel('Number of Parameters (k)')
    ax.set_ylabel('Total Complexity Penalty')
    ax.legend()
    ax.set_title('Total Penalty vs Model Complexity')
    
    plt.tight_layout()
    plt.show()


def plot_waic_diagnostics(
    elpd_i: np.ndarray,
    p_waic_i: np.ndarray,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Plot pointwise WAIC diagnostics.
    
    Parameters
    ----------
    elpd_i : ndarray
        Pointwise expected log predictive density
    p_waic_i : ndarray
        Pointwise effective parameters
    figsize : tuple
        Figure size
    """
    n = len(elpd_i)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Pointwise elpd
    ax = axes[0]
    ax.scatter(range(n), elpd_i, alpha=0.6, s=20)
    ax.axhline(np.mean(elpd_i), color='red', linestyle='--', 
               label=f'Mean: {np.mean(elpd_i):.2f}')
    
    # Highlight problematic observations
    threshold = np.percentile(elpd_i, 5)
    problematic = elpd_i < threshold
    ax.scatter(np.where(problematic)[0], elpd_i[problematic], 
               color='red', s=40, label=f'Bottom 5% (n={np.sum(problematic)})')
    
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Pointwise ELPD')
    ax.legend()
    ax.set_title('Pointwise Expected Log Predictive Density')
    
    # Right: Pointwise p_waic
    ax = axes[1]
    ax.scatter(range(n), p_waic_i, alpha=0.6, s=20)
    ax.axhline(np.mean(p_waic_i), color='red', linestyle='--',
               label=f'Mean: {np.mean(p_waic_i):.3f}')
    
    # Highlight high influence observations
    threshold = np.percentile(p_waic_i, 95)
    high_influence = p_waic_i > threshold
    ax.scatter(np.where(high_influence)[0], p_waic_i[high_influence],
               color='red', s=40, label=f'Top 5% (n={np.sum(high_influence)})')
    
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Pointwise p_WAIC')
    ax.legend()
    ax.set_title('Pointwise Effective Parameters')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Demo Functions
# =============================================================================

def demo_basic_ic():
    """Demonstrate basic AIC/BIC comparison for polynomial regression."""
    
    print("=" * 70)
    print("BASIC INFORMATION CRITERIA: POLYNOMIAL REGRESSION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data from quadratic model
    n = 100
    x = np.linspace(-3, 3, n)
    y_true = 1 + 0.5 * x - 0.3 * x**2
    y = y_true + np.random.normal(0, 0.5, n)
    
    print(f"\nTrue model: y = 1 + 0.5x - 0.3x² + ε, ε ~ N(0, 0.25)")
    print(f"Sample size: n = {n}")
    
    # Fit models of different degrees
    print("\n--- Fitting Polynomial Models ---")
    
    models = {}
    
    for degree in range(1, 7):
        # Create design matrix
        X = np.column_stack([x**i for i in range(degree + 1)])
        
        # Fit model
        model = LinearRegressionIC()
        model.fit(X, y)
        
        ll = model.log_likelihood(X, y)
        k = degree + 2  # coefficients + sigma
        
        models[f'Degree {degree}'] = (ll, k)
        
        print(f"Degree {degree}: k={k}, log-lik={ll:.2f}")
    
    # Compare models
    comparison = compare_models(models, n)
    print("\n" + comparison.summary())
    
    # Weights
    print("\nModel weights:")
    aic_w = comparison.weights('aic')
    bic_w = comparison.weights('bic')
    
    for i, name in enumerate(comparison.names):
        print(f"  {name}: AIC weight = {aic_w[i]:.3f}, BIC weight = {bic_w[i]:.3f}")
    
    return comparison


def demo_aic_vs_bic():
    """Demonstrate when AIC and BIC disagree."""
    
    print("\n" + "=" * 70)
    print("AIC VS BIC: WHEN THEY DISAGREE")
    print("=" * 70)
    
    np.random.seed(123)
    
    # True model is simple (degree 1)
    true_degree = 1
    
    sample_sizes = [20, 50, 100, 500, 1000]
    
    print("\nTrue model: y = 1 + 0.5x + ε")
    print("Comparing degree 1 vs degree 3 polynomials")
    print("\n  n     AIC₁   AIC₃    BIC₁   BIC₃   AIC choice  BIC choice")
    print("-" * 65)
    
    for n in sample_sizes:
        x = np.linspace(-3, 3, n)
        y = 1 + 0.5 * x + np.random.normal(0, 0.5, n)
        
        results = []
        for degree in [1, 3]:
            X = np.column_stack([x**i for i in range(degree + 1)])
            model = LinearRegressionIC()
            model.fit(X, y)
            
            ll = model.log_likelihood(X, y)
            k = degree + 2
            
            results.append({
                'aic': aic(ll, k),
                'bic': bic(ll, k, n)
            })
        
        aic_choice = 1 if results[0]['aic'] < results[1]['aic'] else 3
        bic_choice = 1 if results[0]['bic'] < results[1]['bic'] else 3
        
        print(f"{n:5d}  {results[0]['aic']:6.1f} {results[1]['aic']:6.1f}  "
              f"{results[0]['bic']:6.1f} {results[1]['bic']:6.1f}   "
              f"Degree {aic_choice}      Degree {bic_choice}")
    
    print("\n*** BIC's stronger penalty correctly identifies the simpler true model")
    print("    even when AIC might prefer the more complex model")


def demo_waic():
    """Demonstrate WAIC computation for Bayesian models."""
    
    print("\n" + "=" * 70)
    print("WAIC: FULLY BAYESIAN MODEL COMPARISON")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    n = 50
    x = np.linspace(-2, 2, n)
    y_true = 1 + 0.5 * x - 0.2 * x**2
    y = y_true + np.random.normal(0, 0.3, n)
    
    print(f"\nTrue model: y = 1 + 0.5x - 0.2x² + ε")
    print(f"Sample size: n = {n}")
    
    print("\n--- Computing WAIC for Different Polynomial Degrees ---")
    
    results = []
    
    for degree in [1, 2, 3, 4]:
        X = np.column_stack([x**i for i in range(degree + 1)])
        
        model = BayesianLinearRegression(g=n)
        model.fit(X, y)
        
        waic_val, lppd, p_waic = model.compute_waic(n_samples=2000)
        
        results.append({
            'degree': degree,
            'waic': waic_val,
            'lppd': lppd,
            'p_waic': p_waic
        })
        
        print(f"Degree {degree}: WAIC = {waic_val:.2f}, "
              f"lppd = {lppd:.2f}, p_WAIC = {p_waic:.2f}")
    
    # Find best model
    best = min(results, key=lambda x: x['waic'])
    print(f"\n*** Best model by WAIC: Degree {best['degree']}")
    
    return results


def demo_dic():
    """Demonstrate DIC computation."""
    
    print("\n" + "=" * 70)
    print("DIC: DEVIANCE INFORMATION CRITERION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    n = 100
    x = np.linspace(0, 10, n)
    true_rate = 0.5
    y = np.random.poisson(np.exp(true_rate * x) * 0.1, n)
    
    print(f"\nPoisson regression example")
    print(f"Sample size: n = {n}")
    
    # Simulate posterior samples for two models
    # Model 1: Constant rate
    # Model 2: Linear rate
    
    print("\n--- Model 1: Constant rate ---")
    
    # Simulate MCMC samples (simplified)
    n_samples = 2000
    
    # Model 1: lambda = exp(beta0)
    beta0_samples = np.random.normal(1.5, 0.1, n_samples)
    
    log_lik_1 = np.zeros(n_samples)
    for s in range(n_samples):
        rate = np.exp(beta0_samples[s]) * np.ones(n)
        log_lik_1[s] = np.sum(stats.poisson.logpmf(y, rate))
    
    # Log-likelihood at posterior mean
    rate_mean = np.exp(np.mean(beta0_samples)) * np.ones(n)
    log_lik_at_mean_1 = np.sum(stats.poisson.logpmf(y, rate_mean))
    
    dic_1, pd_1 = dic(log_lik_1, log_lik_at_mean_1)
    print(f"  DIC = {dic_1:.2f}, p_D = {pd_1:.2f}")
    
    print("\n--- Model 2: Linear rate ---")
    
    # Model 2: lambda = exp(beta0 + beta1 * x)
    beta0_samples_2 = np.random.normal(0.5, 0.05, n_samples)
    beta1_samples = np.random.normal(0.1, 0.01, n_samples)
    
    log_lik_2 = np.zeros(n_samples)
    for s in range(n_samples):
        rate = np.exp(beta0_samples_2[s] + beta1_samples[s] * x)
        log_lik_2[s] = np.sum(stats.poisson.logpmf(y, rate))
    
    # Log-likelihood at posterior mean
    rate_mean = np.exp(np.mean(beta0_samples_2) + np.mean(beta1_samples) * x)
    log_lik_at_mean_2 = np.sum(stats.poisson.logpmf(y, rate_mean))
    
    dic_2, pd_2 = dic(log_lik_2, log_lik_at_mean_2)
    print(f"  DIC = {dic_2:.2f}, p_D = {pd_2:.2f}")
    
    print(f"\n*** Model comparison: ΔDIC = {dic_1 - dic_2:.2f}")
    if dic_1 < dic_2:
        print("    Constant rate model preferred")
    else:
        print("    Linear rate model preferred")


def demo_model_averaging():
    """Demonstrate model averaging with IC weights."""
    
    print("\n" + "=" * 70)
    print("MODEL AVERAGING WITH INFORMATION CRITERIA")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    n = 50
    x = np.linspace(-2, 2, n)
    y_true = 1 + 0.3 * x - 0.15 * x**2 + 0.05 * x**3
    y = y_true + np.random.normal(0, 0.3, n)
    
    print(f"\nTrue model has small cubic term")
    print("Comparing models of degrees 1, 2, 3, 4")
    
    # Fit models
    models = {}
    predictions = {}
    
    x_new = np.linspace(-2.5, 2.5, 100)
    
    for degree in [1, 2, 3, 4]:
        X = np.column_stack([x**i for i in range(degree + 1)])
        X_new = np.column_stack([x_new**i for i in range(degree + 1)])
        
        model = LinearRegressionIC()
        model.fit(X, y)
        
        ll = model.log_likelihood(X, y)
        k = degree + 2
        
        models[f'Degree {degree}'] = (ll, k)
        predictions[degree] = X_new @ model.beta_hat
    
    # Compute weights
    comparison = compare_models(models, n)
    weights = comparison.weights('aic')
    
    print("\nAIC weights:")
    for i, name in enumerate(comparison.names):
        print(f"  {name}: {weights[i]:.3f}")
    
    # Model-averaged prediction
    y_avg = np.zeros_like(x_new)
    for i, degree in enumerate([1, 2, 3, 4]):
        y_avg += weights[i] * predictions[degree]
    
    print(f"\n*** Model-averaged prediction incorporates uncertainty")
    print(f"    about which model is correct")
    
    return comparison, predictions, y_avg


def demo_criteria_consistency():
    """Demonstrate BIC consistency vs AIC efficiency."""
    
    print("\n" + "=" * 70)
    print("CONSISTENCY VS EFFICIENCY: AIC VS BIC")
    print("=" * 70)
    
    np.random.seed(42)
    
    # True model: simple linear
    true_degree = 1
    
    sample_sizes = [20, 50, 100, 200, 500, 1000]
    n_simulations = 100
    
    print(f"\nTrue model: degree {true_degree}")
    print(f"Simulations per sample size: {n_simulations}")
    
    print("\n  n     AIC correct   BIC correct")
    print("-" * 40)
    
    for n in sample_sizes:
        aic_correct = 0
        bic_correct = 0
        
        for _ in range(n_simulations):
            x = np.linspace(-3, 3, n)
            y = 1 + 0.5 * x + np.random.normal(0, 0.5, n)
            
            models = {}
            for degree in [1, 2, 3, 4]:
                X = np.column_stack([x**i for i in range(degree + 1)])
                model = LinearRegressionIC()
                model.fit(X, y)
                
                ll = model.log_likelihood(X, y)
                k = degree + 2
                models[degree] = (ll, k)
            
            comparison = compare_models(
                {f'd{d}': models[d] for d in models},
                n
            )
            
            aic_best = np.argmin(comparison.aic) + 1
            bic_best = np.argmin(comparison.bic) + 1
            
            if aic_best == true_degree:
                aic_correct += 1
            if bic_best == true_degree:
                bic_correct += 1
        
        print(f"{n:5d}     {aic_correct/n_simulations*100:5.1f}%        "
              f"{bic_correct/n_simulations*100:5.1f}%")
    
    print("\n*** BIC is consistent: converges to true model as n → ∞")
    print("    AIC is efficient: minimizes prediction error but may overfit")


if __name__ == "__main__":
    comparison = demo_basic_ic()
    demo_aic_vs_bic()
    demo_waic()
    demo_dic()
    demo_model_averaging()
    demo_criteria_consistency()
```

---

## Summary

| Criterion | Formula | Primary Use |
|-----------|---------|-------------|
| **AIC** | $-2\hat{\ell} + 2k$ | Prediction (KL minimization) |
| **AICc** | $\text{AIC} + \frac{2k(k+1)}{n-k-1}$ | Small sample prediction |
| **BIC** | $-2\hat{\ell} + k\log n$ | Model identification |
| **DIC** | $\bar{D} + p_D$ | Hierarchical Bayes |
| **WAIC** | $-2(\text{lppd} - p_{\text{WAIC}})$ | Fully Bayesian |

### Key Properties

| Property | AIC | BIC | DIC | WAIC |
|----------|-----|-----|-----|------|
| Consistent | No | Yes | No | No |
| Efficient | Yes | No | — | Yes |
| Requires MCMC | No | No | Yes | Yes |
| Singular models | Fails | Fails | May fail | Works |

### Effective Parameters

| Criterion | Effective Parameters |
|-----------|---------------------|
| AIC/BIC | $k$ (count) |
| DIC | $p_D = \bar{D} - D(\bar{\theta})$ |
| WAIC | $p_{\text{WAIC}} = \sum_i \text{Var}[\log p(y_i \mid \theta)]$ |

### When to Use

| Situation | Recommended Criterion |
|-----------|----------------------|
| Prediction focus | AIC/AICc |
| True model identification | BIC |
| Hierarchical Bayesian models | DIC or WAIC |
| Complex/singular models | WAIC or LOO-CV |
| Small sample size | AICc |
| Model averaging | AIC weights or BIC weights |

### Interpretation Guidelines

**Delta values** (difference from best model):

| $\Delta$ | Support |
|----------|---------|
| 0–2 | Substantial |
| 2–7 | Considerably less |
| > 10 | Essentially none |

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Bayes factors | Ch13: Bayes Factors | BIC approximates log BF |
| Model evidence | Ch13: Model Evidence | IC avoids full integration |
| Prior selection | Ch13: Foundations | BIC assumes unit info prior |
| Posterior inference | Ch13: Distributions | DIC/WAIC use posterior samples |
| BNN comparison | Ch13: BNN | Architecture selection |

### Key References

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Trans. Automatic Control*, 19(6), 716-723.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
- Spiegelhalter, D. J., et al. (2002). Bayesian measures of model complexity and fit. *JRSS B*, 64(4), 583-639.
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and WAIC. *JMLR*, 11, 3571-3594.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
