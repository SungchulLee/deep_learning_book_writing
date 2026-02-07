# Model Evidence (Marginal Likelihood)

The **model evidence**, also called the **marginal likelihood**, is the probability of observed data under a model after integrating out all parameters. It serves as the cornerstone of Bayesian model comparison, naturally implementing Occam's razor by penalizing models that are overly complex relative to the data they explain.

---

## Motivation: Why Model Evidence?

### The Model Comparison Problem

Given data $\mathcal{D}$ and competing models $\mathcal{M}_1, \mathcal{M}_2, \ldots$, how do we decide which model best explains the data?

**Frequentist approaches**:
- Likelihood ratio tests (nested models only)
- Cross-validation (computationally expensive)
- Information criteria (AIC, BIC) — approximations

**Bayesian approach**:
- Compute the probability of each model given the data
- Use Bayes' theorem at the model level

### From Parameter Inference to Model Inference

Standard Bayesian inference fixes the model and infers parameters:

$$
p(\theta \mid \mathcal{D}, \mathcal{M}) = \frac{p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M})}{p(\mathcal{D} \mid \mathcal{M})}
$$

Model comparison requires the denominator — the **model evidence**:

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta
$$

This integral averages the likelihood over all possible parameter values, weighted by their prior probability.

---

## Definition and Interpretation

### Formal Definition

The **model evidence** (marginal likelihood) for model $\mathcal{M}$ with parameters $\theta$ is:

$$
\boxed{p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta}
$$

**Components**:
- $p(\mathcal{D} \mid \theta, \mathcal{M})$: Likelihood function
- $p(\theta \mid \mathcal{M})$: Prior distribution
- Integration: Over entire parameter space

### Multiple Interpretations

**1. Average likelihood**: The expected likelihood under the prior

$$
p(\mathcal{D} \mid \mathcal{M}) = \mathbb{E}_{\theta \sim p(\theta \mid \mathcal{M})}[p(\mathcal{D} \mid \theta, \mathcal{M})]
$$

**2. Predictive probability**: How well the model predicted the data before seeing it

**3. Normalizing constant**: The denominator in Bayes' theorem

$$
p(\theta \mid \mathcal{D}, \mathcal{M}) = \frac{p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M})}{p(\mathcal{D} \mid \mathcal{M})}
$$

**4. Prior predictive**: The probability assigned to $\mathcal{D}$ by the prior predictive distribution

### Why "Marginal" Likelihood?

The term "marginal" refers to marginalizing (integrating) over parameters:

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D}, \theta \mid \mathcal{M}) \, d\theta = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta
$$

This contrasts with the **conditional likelihood** $p(\mathcal{D} \mid \hat{\theta}, \mathcal{M})$ evaluated at a specific parameter value.

---

## Bayesian Model Comparison

### Posterior Model Probabilities

Given prior probabilities $p(\mathcal{M}_k)$ over models:

$$
p(\mathcal{M}_k \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathcal{M}_k) \, p(\mathcal{M}_k)}{\sum_j p(\mathcal{D} \mid \mathcal{M}_j) \, p(\mathcal{M}_j)}
$$

The model evidence directly determines how much the data updates our model beliefs.

### Posterior Odds

For two models $\mathcal{M}_1$ and $\mathcal{M}_2$:

$$
\underbrace{\frac{p(\mathcal{M}_1 \mid \mathcal{D})}{p(\mathcal{M}_2 \mid \mathcal{D})}}_{\text{Posterior odds}} = \underbrace{\frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_2)}}_{\text{Bayes factor}} \times \underbrace{\frac{p(\mathcal{M}_1)}{p(\mathcal{M}_2)}}_{\text{Prior odds}}
$$

With equal prior odds, the Bayes factor equals the posterior odds.

### Model Averaging

Instead of selecting a single model, we can average predictions:

$$
p(y^* \mid x^*, \mathcal{D}) = \sum_k p(y^* \mid x^*, \mathcal{D}, \mathcal{M}_k) \, p(\mathcal{M}_k \mid \mathcal{D})
$$

This accounts for model uncertainty and often improves predictive performance.

---

## Occam's Razor: Automatic Complexity Penalty

### The Bayesian Occam's Razor

Model evidence naturally penalizes complexity. Consider the integral:

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

**Simple model**: Prior concentrated; if data matches, high evidence
**Complex model**: Prior spread thin; must "pay" for flexibility

### Geometric Intuition

Think of the prior as a "probability budget" that must cover the parameter space:

$$
\int p(\theta) \, d\theta = 1
$$

A complex model spreads this budget over a larger space, so each region gets less probability mass. Unless the extra flexibility is needed to explain the data, the complex model wastes its budget.

### Illustrative Example

Consider fitting a polynomial to data:

| Model | Parameters | Prior Volume | Typical Likelihood | Evidence |
|-------|------------|--------------|-------------------|----------|
| Linear | 2 | Small | Moderate | High (if linear trend) |
| Cubic | 4 | Medium | Higher | Medium |
| Degree-10 | 11 | Large | Highest | Low (overfitting) |

The degree-10 polynomial achieves the highest likelihood but lowest evidence because its prior is spread too thin.

### Mathematical Decomposition

The log evidence can be decomposed as:

$$
\log p(\mathcal{D} \mid \mathcal{M}) = \underbrace{\log p(\mathcal{D} \mid \hat{\theta})}_{\text{Best fit}} - \underbrace{D_{KL}(p(\theta \mid \mathcal{D}) \| p(\theta))}_{\text{Complexity penalty}}
$$

where $\hat{\theta}$ is the MAP estimate and $D_{KL}$ measures how much the posterior differs from the prior.

**Interpretation**:
- First term: How well the model can fit the data (goodness of fit)
- Second term: How much the model had to "learn" from data (complexity)

---

## Analytical Solutions for Conjugate Models

### General Principle

For conjugate models, the evidence is available in closed form because:

$$
p(\mathcal{D} \mid \mathcal{M}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\theta \mid \mathcal{D})}
$$

Since we know all three distributions analytically, we can compute the ratio.

### Beta-Bernoulli Model

**Setup**: $x_i \sim \text{Bernoulli}(\theta)$, $\theta \sim \text{Beta}(\alpha_0, \beta_0)$

**Evidence**:

$$
p(\mathcal{D}) = \frac{B(\alpha_0 + s, \beta_0 + f)}{B(\alpha_0, \beta_0)}
$$

where $s = \sum x_i$ (successes), $f = n - s$ (failures), and $B(\cdot, \cdot)$ is the Beta function.

**Log evidence**:

$$
\log p(\mathcal{D}) = \log B(\alpha_n, \beta_n) - \log B(\alpha_0, \beta_0)
$$

$$
= \log\Gamma(\alpha_n) + \log\Gamma(\beta_n) - \log\Gamma(\alpha_n + \beta_n) - \log\Gamma(\alpha_0) - \log\Gamma(\beta_0) + \log\Gamma(\alpha_0 + \beta_0)
$$

### Gaussian with Known Variance

**Setup**: $x_i \sim \mathcal{N}(\mu, \sigma^2)$ (known $\sigma^2$), $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**Evidence**:

$$
p(\mathcal{D}) = (2\pi\sigma^2)^{-n/2} \cdot \sqrt{\frac{\sigma_0^2}{\sigma_n^2}} \cdot \exp\left(-\frac{1}{2\sigma^2}\sum_i(x_i - \bar{x})^2 - \frac{(\bar{x} - \mu_0)^2}{2(\sigma^2/n + \sigma_0^2)}\right)
$$

**Log evidence**:

$$
\log p(\mathcal{D}) = -\frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2}\log\frac{\tau_0}{\tau_n} - \frac{1}{2\sigma^2}\sum_i(x_i - \bar{x})^2 - \frac{\tau_0 n \tau}{2\tau_n}(\bar{x} - \mu_0)^2
$$

where $\tau = 1/\sigma^2$, $\tau_0 = 1/\sigma_0^2$, $\tau_n = \tau_0 + n\tau$.

### Gaussian with Unknown Variance (NIG Prior)

**Setup**: $x_i \sim \mathcal{N}(\mu, \sigma^2)$, $(\mu, \sigma^2) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$

**Log evidence**:

$$
\log p(\mathcal{D}) = \log\Gamma(\alpha_n) - \log\Gamma(\alpha_0) + \alpha_0\log\beta_0 - \alpha_n\log\beta_n + \frac{1}{2}\log\frac{\kappa_0}{\kappa_n} - \frac{n}{2}\log(2\pi)
$$

where $\kappa_n = \kappa_0 + n$, $\alpha_n = \alpha_0 + n/2$, and $\beta_n$ follows the NIG update formula.

---

## Approximation Methods

For non-conjugate models, exact computation is often intractable. Several approximations exist:

### Laplace Approximation

Approximate the posterior as Gaussian around the MAP estimate $\hat{\theta}$:

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) + \log p(\hat{\theta}) + \frac{d}{2}\log(2\pi) - \frac{1}{2}\log|H|
$$

where $d$ is the parameter dimension and $H$ is the Hessian of the negative log posterior at $\hat{\theta}$.

**Pros**: Fast, only requires optimization
**Cons**: Assumes posterior is approximately Gaussian

### BIC Approximation

The Bayesian Information Criterion approximates log evidence:

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) - \frac{d}{2}\log n
$$

where $d$ is the number of parameters and $n$ is sample size.

**Derivation**: From Laplace approximation, assuming unit information prior.

### Harmonic Mean Estimator

Given posterior samples $\theta^{(1)}, \ldots, \theta^{(S)}$:

$$
p(\mathcal{D})^{-1} = \mathbb{E}_{p(\theta \mid \mathcal{D})}\left[\frac{1}{p(\mathcal{D} \mid \theta)}\right] \approx \frac{1}{S}\sum_{s=1}^S \frac{1}{p(\mathcal{D} \mid \theta^{(s)})}
$$

**Warning**: This estimator has infinite variance and is notoriously unreliable!

### Importance Sampling

Choose a proposal distribution $q(\theta)$:

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta = \int \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{q(\theta)} \, q(\theta) \, d\theta
$$

$$
\approx \frac{1}{S}\sum_{s=1}^S \frac{p(\mathcal{D} \mid \theta^{(s)}) \, p(\theta^{(s)})}{q(\theta^{(s)})}, \quad \theta^{(s)} \sim q
$$

**Good proposal**: Approximate posterior (e.g., from Laplace approximation)

### Annealed Importance Sampling (AIS)

Create a sequence of distributions interpolating from prior to posterior:

$$
p_t(\theta) \propto p(\theta) \, p(\mathcal{D} \mid \theta)^{\beta_t}, \quad 0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1
$$

The evidence is estimated via the product of intermediate normalizing constants.

### Nested Sampling

Transform the evidence integral into a one-dimensional integral over the prior mass:

$$
p(\mathcal{D}) = \int_0^1 L(X) \, dX
$$

where $X(\lambda) = \int_{p(\mathcal{D} \mid \theta) > \lambda} p(\theta) \, d\theta$ is the prior mass with likelihood above $\lambda$.

**Popular implementation**: MultiNest, dynesty

---

## Evidence in Sequential Data

### Online Evidence Computation

For sequential observations $x_1, x_2, \ldots, x_n$:

$$
p(x_1, \ldots, x_n \mid \mathcal{M}) = \prod_{t=1}^n p(x_t \mid x_1, \ldots, x_{t-1}, \mathcal{M})
$$

Each factor is the **one-step-ahead predictive**:

$$
p(x_t \mid x_{1:t-1}) = \int p(x_t \mid \theta) \, p(\theta \mid x_{1:t-1}) \, d\theta
$$

**Log evidence**:

$$
\log p(\mathcal{D} \mid \mathcal{M}) = \sum_{t=1}^n \log p(x_t \mid x_{1:t-1})
$$

This decomposition is useful for:
- Online model comparison
- Detecting model failure (when predictive probability drops)
- Prequential (predictive sequential) validation

### Prequential Interpretation

The log evidence equals the sum of log predictive scores:

$$
\log p(\mathcal{D}) = \sum_{t=1}^n \text{LogScore}_t
$$

This connects evidence to predictive performance — models with higher evidence made better predictions on average.

---

## Sensitivity to Prior Specification

### The Prior's Role in Evidence

Unlike posterior inference (which is often robust to prior choice with enough data), evidence is **highly sensitive** to the prior:

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

Changing the prior directly changes the evidence, even asymptotically.

### Improper Priors

**Critical issue**: Improper priors yield undefined evidence!

If $\int p(\theta) \, d\theta = \infty$, then $p(\mathcal{D} \mid \mathcal{M})$ is only defined up to an arbitrary constant.

**Consequence**: Cannot compare models using improper priors (Bayes factors are meaningless).

### Vague Proper Priors

Even proper but very diffuse priors cause problems:

$$
p(\theta) = \mathcal{N}(0, 10^6) \quad \text{(variance } 10^6 \text{)}
$$

This prior assigns negligible probability to any reasonable parameter region, artificially penalizing the model.

### Prior Sensitivity Analysis

Always check how evidence changes with prior:
1. Compute evidence under several reasonable priors
2. If conclusions are robust, proceed with confidence
3. If sensitive, report the range of conclusions

### Fractional Bayes Factors

One solution: Use part of the data to define a "training" posterior, then compute evidence on the rest:

$$
\text{FBF}_{12} = \frac{p(\mathcal{D}^{\text{test}} \mid \mathcal{D}^{\text{train}}, \mathcal{M}_1)}{p(\mathcal{D}^{\text{test}} \mid \mathcal{D}^{\text{train}}, \mathcal{M}_2)}
$$

This reduces prior sensitivity but requires data splitting.

---

## Python Implementation

```python
"""
Model Evidence (Marginal Likelihood): Complete Implementation

This module provides computation of model evidence for various Bayesian
models, demonstrating exact solutions for conjugate cases and approximations
for general models.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from typing import Tuple, List, Optional, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Abstract Base Class for Bayesian Models
# =============================================================================

class BayesianModel(ABC):
    """Abstract base class for models with evidence computation."""
    
    @abstractmethod
    def log_evidence(self, data: np.ndarray) -> float:
        """Compute log marginal likelihood."""
        pass
    
    @abstractmethod
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """Compute log likelihood at given parameters."""
        pass
    
    @abstractmethod
    def log_prior(self, params: np.ndarray) -> float:
        """Compute log prior density."""
        pass
    
    def log_posterior_unnorm(self, data: np.ndarray, params: np.ndarray) -> float:
        """Compute unnormalized log posterior."""
        return self.log_likelihood(data, params) + self.log_prior(params)


# =============================================================================
# Conjugate Models with Exact Evidence
# =============================================================================

@dataclass
class BetaBernoulliModel(BayesianModel):
    """
    Beta-Bernoulli model with exact evidence computation.
    
    Model: x_i | θ ~ Bernoulli(θ), θ ~ Beta(α₀, β₀)
    
    Parameters
    ----------
    alpha0 : float
        Prior alpha parameter
    beta0 : float
        Prior beta parameter
    """
    alpha0: float = 1.0
    beta0: float = 1.0
    
    def log_evidence(self, data: np.ndarray) -> float:
        """
        Compute log evidence analytically.
        
        log p(D) = log B(α_n, β_n) - log B(α₀, β₀)
        """
        data = np.atleast_1d(data)
        s = data.sum()  # successes
        f = len(data) - s  # failures
        
        alpha_n = self.alpha0 + s
        beta_n = self.beta0 + f
        
        # log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a + b)
        log_B_prior = gammaln(self.alpha0) + gammaln(self.beta0) - gammaln(self.alpha0 + self.beta0)
        log_B_post = gammaln(alpha_n) + gammaln(beta_n) - gammaln(alpha_n + beta_n)
        
        return log_B_post - log_B_prior
    
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """Compute Bernoulli log likelihood."""
        theta = params[0]
        if theta <= 0 or theta >= 1:
            return -np.inf
        s = data.sum()
        f = len(data) - s
        return s * np.log(theta) + f * np.log(1 - theta)
    
    def log_prior(self, params: np.ndarray) -> float:
        """Compute Beta log prior."""
        theta = params[0]
        return stats.beta.logpdf(theta, self.alpha0, self.beta0)
    
    def sequential_log_evidence(self, data: np.ndarray) -> Tuple[float, List[float]]:
        """
        Compute evidence sequentially via predictive probabilities.
        
        Returns total log evidence and list of predictive log probabilities.
        """
        data = np.atleast_1d(data)
        alpha, beta = self.alpha0, self.beta0
        log_probs = []
        
        for x in data:
            # Predictive probability
            if x == 1:
                p = alpha / (alpha + beta)
            else:
                p = beta / (alpha + beta)
            log_probs.append(np.log(p))
            
            # Update
            if x == 1:
                alpha += 1
            else:
                beta += 1
        
        return sum(log_probs), log_probs


@dataclass
class GaussianKnownVarianceModel(BayesianModel):
    """
    Gaussian model with known variance and exact evidence.
    
    Model: x_i | μ ~ N(μ, σ²), μ ~ N(μ₀, σ₀²)
    
    Parameters
    ----------
    mu0 : float
        Prior mean
    sigma0_sq : float
        Prior variance
    sigma_sq : float
        Known data variance
    """
    mu0: float = 0.0
    sigma0_sq: float = 1.0
    sigma_sq: float = 1.0
    
    def log_evidence(self, data: np.ndarray) -> float:
        """Compute log evidence analytically."""
        data = np.atleast_1d(data)
        n = len(data)
        x_bar = data.mean()
        
        tau = 1 / self.sigma_sq
        tau0 = 1 / self.sigma0_sq
        tau_n = tau0 + n * tau
        
        # Sum of squared deviations from sample mean
        ss = ((data - x_bar) ** 2).sum()
        
        # Log evidence
        log_ev = (
            -0.5 * n * np.log(2 * np.pi * self.sigma_sq)  # Likelihood normalization
            + 0.5 * np.log(tau0 / tau_n)  # Prior/posterior precision ratio
            - 0.5 * tau * ss  # Data variability
            - 0.5 * tau0 * n * tau / tau_n * (x_bar - self.mu0) ** 2  # Prior-data discrepancy
        )
        
        return log_ev
    
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """Compute Gaussian log likelihood."""
        mu = params[0]
        return stats.norm.logpdf(data, mu, np.sqrt(self.sigma_sq)).sum()
    
    def log_prior(self, params: np.ndarray) -> float:
        """Compute Gaussian log prior."""
        mu = params[0]
        return stats.norm.logpdf(mu, self.mu0, np.sqrt(self.sigma0_sq))
    
    def sequential_log_evidence(self, data: np.ndarray) -> Tuple[float, List[float]]:
        """Compute evidence sequentially."""
        data = np.atleast_1d(data)
        
        mu_t = self.mu0
        tau_t = 1 / self.sigma0_sq
        tau = 1 / self.sigma_sq
        
        log_probs = []
        
        for x in data:
            # Predictive distribution: N(μ_t, σ² + 1/τ_t)
            pred_var = self.sigma_sq + 1 / tau_t
            log_p = stats.norm.logpdf(x, mu_t, np.sqrt(pred_var))
            log_probs.append(log_p)
            
            # Update
            tau_new = tau_t + tau
            mu_t = (tau_t * mu_t + tau * x) / tau_new
            tau_t = tau_new
        
        return sum(log_probs), log_probs


@dataclass
class NormalInverseGammaModel(BayesianModel):
    """
    Gaussian model with unknown mean and variance.
    
    Model: x_i | μ, σ² ~ N(μ, σ²)
           (μ, σ²) ~ NIG(μ₀, κ₀, α₀, β₀)
    """
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    
    def log_evidence(self, data: np.ndarray) -> float:
        """Compute log evidence analytically."""
        data = np.atleast_1d(data)
        n = len(data)
        
        if n == 0:
            return 0.0
        
        x_bar = data.mean()
        ss = ((data - x_bar) ** 2).sum()
        
        # Posterior parameters
        kappa_n = self.kappa0 + n
        alpha_n = self.alpha0 + n / 2
        beta_n = (self.beta0 + 0.5 * ss + 
                  0.5 * self.kappa0 * n / kappa_n * (x_bar - self.mu0) ** 2)
        
        # Log evidence
        log_ev = (
            gammaln(alpha_n) - gammaln(self.alpha0)
            + self.alpha0 * np.log(self.beta0) - alpha_n * np.log(beta_n)
            + 0.5 * np.log(self.kappa0 / kappa_n)
            - 0.5 * n * np.log(2 * np.pi)
        )
        
        return log_ev
    
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """Compute Gaussian log likelihood."""
        mu, sigma_sq = params
        if sigma_sq <= 0:
            return -np.inf
        return stats.norm.logpdf(data, mu, np.sqrt(sigma_sq)).sum()
    
    def log_prior(self, params: np.ndarray) -> float:
        """Compute NIG log prior."""
        mu, sigma_sq = params
        if sigma_sq <= 0:
            return -np.inf
        
        # p(σ²) = Inv-Gamma(α₀, β₀)
        log_p_sigma = stats.invgamma.logpdf(sigma_sq, self.alpha0, scale=self.beta0)
        
        # p(μ | σ²) = N(μ₀, σ²/κ₀)
        log_p_mu = stats.norm.logpdf(mu, self.mu0, np.sqrt(sigma_sq / self.kappa0))
        
        return log_p_sigma + log_p_mu


# =============================================================================
# Evidence Approximation Methods
# =============================================================================

def laplace_approximation(
    model: BayesianModel,
    data: np.ndarray,
    init_params: np.ndarray,
    param_bounds: Optional[List[Tuple]] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute log evidence via Laplace approximation.
    
    log p(D) ≈ log p(D|θ̂) + log p(θ̂) + (d/2)log(2π) - (1/2)log|H|
    
    Parameters
    ----------
    model : BayesianModel
        Model with log_likelihood and log_prior methods
    data : array
        Observed data
    init_params : array
        Initial parameter values for optimization
    param_bounds : list of tuples, optional
        Bounds for each parameter
    
    Returns
    -------
    log_evidence : float
        Laplace approximation to log evidence
    map_params : array
        MAP parameter estimates
    hessian : array
        Hessian of negative log posterior at MAP
    """
    def neg_log_posterior(params):
        return -model.log_posterior_unnorm(data, params)
    
    # Find MAP estimate
    result = minimize(
        neg_log_posterior,
        init_params,
        method='L-BFGS-B',
        bounds=param_bounds
    )
    
    map_params = result.x
    
    # Compute Hessian numerically
    d = len(map_params)
    eps = 1e-5
    hessian = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            params_pp = map_params.copy()
            params_pm = map_params.copy()
            params_mp = map_params.copy()
            params_mm = map_params.copy()
            
            params_pp[i] += eps
            params_pp[j] += eps
            params_pm[i] += eps
            params_pm[j] -= eps
            params_mp[i] -= eps
            params_mp[j] += eps
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            hessian[i, j] = (
                neg_log_posterior(params_pp) - neg_log_posterior(params_pm)
                - neg_log_posterior(params_mp) + neg_log_posterior(params_mm)
            ) / (4 * eps ** 2)
    
    # Laplace approximation
    log_posterior_at_map = model.log_posterior_unnorm(data, map_params)
    sign, log_det_H = np.linalg.slogdet(hessian)
    
    if sign <= 0:
        # Hessian not positive definite; approximation may be poor
        log_det_H = np.log(np.abs(np.linalg.det(hessian)) + 1e-10)
    
    log_evidence = (
        log_posterior_at_map 
        + 0.5 * d * np.log(2 * np.pi) 
        - 0.5 * log_det_H
    )
    
    return log_evidence, map_params, hessian


def importance_sampling_evidence(
    model: BayesianModel,
    data: np.ndarray,
    proposal_samples: np.ndarray,
    proposal_log_pdf: Callable[[np.ndarray], float]
) -> Tuple[float, float]:
    """
    Estimate log evidence via importance sampling.
    
    Parameters
    ----------
    model : BayesianModel
        Model with log_likelihood and log_prior methods
    data : array
        Observed data
    proposal_samples : array
        Samples from proposal distribution, shape (n_samples, d)
    proposal_log_pdf : callable
        Function computing log density of proposal
    
    Returns
    -------
    log_evidence : float
        Estimated log evidence
    log_evidence_std : float
        Estimated standard error (in log scale)
    """
    n_samples = len(proposal_samples)
    log_weights = np.zeros(n_samples)
    
    for i, params in enumerate(proposal_samples):
        log_num = model.log_posterior_unnorm(data, params)
        log_denom = proposal_log_pdf(params)
        log_weights[i] = log_num - log_denom
    
    # Log-sum-exp for numerical stability
    log_evidence = logsumexp(log_weights) - np.log(n_samples)
    
    # Estimate variance
    weights = np.exp(log_weights - log_weights.max())
    ess = weights.sum() ** 2 / (weights ** 2).sum()  # Effective sample size
    
    # Approximate standard error
    log_evidence_std = np.std(log_weights) / np.sqrt(ess)
    
    return log_evidence, log_evidence_std


def bic_approximation(
    log_likelihood_at_mle: float,
    n_params: int,
    n_samples: int
) -> float:
    """
    Compute BIC approximation to log evidence.
    
    log p(D) ≈ log p(D|θ̂_MLE) - (d/2) log(n)
    
    Parameters
    ----------
    log_likelihood_at_mle : float
        Log likelihood at MLE
    n_params : int
        Number of model parameters
    n_samples : int
        Number of data points
    
    Returns
    -------
    float
        BIC approximation to log evidence
    """
    return log_likelihood_at_mle - 0.5 * n_params * np.log(n_samples)


# =============================================================================
# Model Comparison Utilities
# =============================================================================

def compute_model_probabilities(
    log_evidences: List[float],
    prior_probs: Optional[List[float]] = None
) -> np.ndarray:
    """
    Compute posterior model probabilities from log evidences.
    
    Parameters
    ----------
    log_evidences : list
        Log evidence for each model
    prior_probs : list, optional
        Prior probability for each model (uniform if None)
    
    Returns
    -------
    array
        Posterior model probabilities
    """
    log_evidences = np.array(log_evidences)
    n_models = len(log_evidences)
    
    if prior_probs is None:
        log_priors = np.zeros(n_models) - np.log(n_models)
    else:
        log_priors = np.log(prior_probs)
    
    log_posteriors = log_evidences + log_priors
    log_posteriors -= logsumexp(log_posteriors)
    
    return np.exp(log_posteriors)


def bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
    """
    Compute Bayes factor BF₁₂ = p(D|M₁) / p(D|M₂).
    
    Returns the Bayes factor on log scale for numerical stability.
    """
    return log_evidence_1 - log_evidence_2


def interpret_bayes_factor(log_bf: float) -> str:
    """
    Interpret Bayes factor using Kass & Raftery (1995) scale.
    
    Parameters
    ----------
    log_bf : float
        Log Bayes factor (natural log)
    
    Returns
    -------
    str
        Interpretation of evidence strength
    """
    bf = np.exp(log_bf)
    
    if log_bf > 0:
        model = "Model 1"
        abs_log_bf = log_bf
    else:
        model = "Model 2"
        abs_log_bf = -log_bf
    
    # Convert to log base 10 for Kass & Raftery scale
    log10_bf = abs_log_bf / np.log(10)
    
    if log10_bf < 0.5:
        strength = "Not worth more than a bare mention"
    elif log10_bf < 1:
        strength = "Substantial"
    elif log10_bf < 2:
        strength = "Strong"
    else:
        strength = "Decisive"
    
    return f"{strength} evidence for {model} (log₁₀ BF = {log10_bf:.2f})"


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_evidence_comparison(
    models: Dict[str, BayesianModel],
    data: np.ndarray,
    prior_probs: Optional[Dict[str, float]] = None
) -> plt.Figure:
    """Visualize model comparison via evidence."""
    
    model_names = list(models.keys())
    log_evidences = [models[name].log_evidence(data) for name in model_names]
    
    if prior_probs is None:
        priors = None
    else:
        priors = [prior_probs.get(name, 1/len(models)) for name in model_names]
    
    posteriors = compute_model_probabilities(log_evidences, priors)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Log evidences
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, log_evidences, color=colors)
    ax.set_ylabel('Log Evidence', fontsize=12)
    ax.set_title('Log Marginal Likelihood', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, log_evidences):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=10)
    
    # Posterior probabilities
    ax = axes[1]
    bars = ax.bar(model_names, posteriors, color=colors)
    ax.set_ylabel('Posterior Probability', fontsize=12)
    ax.set_title('Model Posterior Probabilities', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, posteriors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    
    # Bayes factors relative to best model
    ax = axes[2]
    best_idx = np.argmax(log_evidences)
    log_bfs = [log_evidences[best_idx] - le for le in log_evidences]
    
    bars = ax.bar(model_names, log_bfs, color=colors)
    ax.set_ylabel('Log Bayes Factor (vs best)', fontsize=12)
    ax.set_title(f'Evidence Against (vs {model_names[best_idx]})', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(np.log(10), color='orange', linestyle='--', alpha=0.7, label='Strong (10:1)')
    ax.axhline(np.log(100), color='red', linestyle='--', alpha=0.7, label='Decisive (100:1)')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_sequential_evidence(
    models: Dict[str, BayesianModel],
    data: np.ndarray
) -> plt.Figure:
    """Visualize how evidence accumulates sequentially."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    model_names = list(models.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    # Compute sequential evidence for each model
    sequential_log_probs = {}
    cumulative_log_evidence = {}
    
    for name, model in models.items():
        if hasattr(model, 'sequential_log_evidence'):
            total, probs = model.sequential_log_evidence(data)
            sequential_log_probs[name] = probs
            cumulative_log_evidence[name] = np.cumsum(probs)
    
    n = len(data)
    x = np.arange(1, n + 1)
    
    # Top: Cumulative log evidence
    ax = axes[0]
    for i, (name, cum_ev) in enumerate(cumulative_log_evidence.items()):
        ax.plot(x, cum_ev, label=name, color=colors[i], linewidth=2)
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('Cumulative Log Evidence', fontsize=12)
    ax.set_title('Evidence Accumulation', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bottom: Sequential log Bayes factor
    ax = axes[1]
    if len(model_names) >= 2:
        name1, name2 = model_names[0], model_names[1]
        cum1 = cumulative_log_evidence[name1]
        cum2 = cumulative_log_evidence[name2]
        log_bf = cum1 - cum2
        
        ax.plot(x, log_bf, 'b-', linewidth=2)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(np.log(10), color='green', linestyle='--', alpha=0.7, label='Substantial for M1')
        ax.axhline(-np.log(10), color='red', linestyle='--', alpha=0.7, label='Substantial for M2')
        
        ax.fill_between(x, 0, log_bf, where=log_bf > 0, alpha=0.3, color='green')
        ax.fill_between(x, 0, log_bf, where=log_bf < 0, alpha=0.3, color='red')
        
        ax.set_xlabel('Number of Observations', fontsize=12)
        ax.set_ylabel(f'Log BF ({name1} vs {name2})', fontsize=12)
        ax.set_title('Sequential Bayes Factor', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_occam_razor_demo(sample_sizes: List[int] = [10, 50, 200]) -> plt.Figure:
    """Demonstrate Occam's razor via polynomial model comparison."""
    
    np.random.seed(42)
    
    # True model: quadratic
    def true_func(x):
        return 2 + 1.5 * x - 0.5 * x**2
    
    # Generate data
    x_full = np.linspace(-2, 2, 200)
    
    fig, axes = plt.subplots(len(sample_sizes), 2, figsize=(14, 4*len(sample_sizes)))
    
    for row, n in enumerate(sample_sizes):
        np.random.seed(42)
        x = np.random.uniform(-2, 2, n)
        y = true_func(x) + np.random.normal(0, 0.5, n)
        
        # Fit polynomials of different degrees
        degrees = [1, 2, 3, 5, 8]
        log_evidences = []
        
        for deg in degrees:
            # Use BIC as evidence approximation
            X = np.vander(x, deg + 1, increasing=True)
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ coeffs
            residuals = y - y_pred
            
            # MLE variance estimate
            sigma2_mle = (residuals ** 2).mean()
            
            # Log likelihood at MLE
            log_lik = -0.5 * n * np.log(2 * np.pi * sigma2_mle) - 0.5 * n
            
            # BIC approximation
            log_ev = bic_approximation(log_lik, deg + 1, n)
            log_evidences.append(log_ev)
        
        # Left: Data and fits
        ax = axes[row, 0] if len(sample_sizes) > 1 else axes[0]
        ax.scatter(x, y, alpha=0.5, s=30, label='Data')
        
        for i, deg in enumerate([1, 2, 5]):
            X = np.vander(x_full, deg + 1, increasing=True)
            X_fit = np.vander(x, deg + 1, increasing=True)
            coeffs = np.linalg.lstsq(X_fit, y, rcond=None)[0]
            y_fit = X @ coeffs
            ax.plot(x_full, y_fit, label=f'Degree {deg}', linewidth=2)
        
        ax.plot(x_full, true_func(x_full), 'k--', linewidth=2, label='True')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Polynomial Fits (n = {n})', fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Right: Evidence comparison
        ax = axes[row, 1] if len(sample_sizes) > 1 else axes[1]
        
        # Normalize to posterior probabilities
        probs = compute_model_probabilities(log_evidences)
        
        colors = ['red' if d != 2 else 'green' for d in degrees]
        bars = ax.bar([f'Deg {d}' for d in degrees], probs, color=colors, alpha=0.7)
        
        ax.set_ylabel('Model Probability', fontsize=12)
        ax.set_title(f'Model Evidence (n = {n})', fontsize=14)
        ax.set_ylim(0, 1.1)
        
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Demonstrations
# =============================================================================

def demo_conjugate_evidence():
    """Demonstrate exact evidence for conjugate models."""
    
    print("=" * 70)
    print("MODEL EVIDENCE: CONJUGATE MODELS")
    print("=" * 70)
    
    # Beta-Bernoulli
    print("\n--- Beta-Bernoulli Model ---")
    np.random.seed(42)
    true_theta = 0.7
    data = np.random.binomial(1, true_theta, 50)
    
    models = {
        'Uniform prior (α=β=1)': BetaBernoulliModel(1, 1),
        'Informative correct (α=7, β=3)': BetaBernoulliModel(7, 3),
        'Informative wrong (α=3, β=7)': BetaBernoulliModel(3, 7),
    }
    
    print(f"Data: {data.sum()} successes in {len(data)} trials (true θ = {true_theta})")
    
    for name, model in models.items():
        log_ev = model.log_evidence(data)
        _, seq_probs = model.sequential_log_evidence(data)
        print(f"\n{name}:")
        print(f"  Log evidence: {log_ev:.4f}")
        print(f"  Sequential check: {sum(seq_probs):.4f}")
    
    # Gaussian models
    print("\n\n--- Gaussian Models ---")
    np.random.seed(123)
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, 30)
    
    print(f"Data: n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}")
    print(f"True: μ={true_mu}, σ={true_sigma}")
    
    # Known variance model
    model_known = GaussianKnownVarianceModel(mu0=0, sigma0_sq=10, sigma_sq=true_sigma**2)
    log_ev_known = model_known.log_evidence(data)
    print(f"\nKnown variance (σ²={true_sigma**2}):")
    print(f"  Log evidence: {log_ev_known:.4f}")
    
    # Unknown variance model
    model_unknown = NormalInverseGammaModel(mu0=0, kappa0=0.1, alpha0=1, beta0=1)
    log_ev_unknown = model_unknown.log_evidence(data)
    print(f"\nUnknown variance (NIG prior):")
    print(f"  Log evidence: {log_ev_unknown:.4f}")


def demo_laplace_approximation():
    """Demonstrate Laplace approximation for evidence."""
    
    print("\n" + "=" * 70)
    print("LAPLACE APPROXIMATION")
    print("=" * 70)
    
    np.random.seed(42)
    data = np.random.normal(5.0, 2.0, 30)
    
    # Use NIG model
    model = NormalInverseGammaModel(mu0=0, kappa0=0.1, alpha0=1, beta0=1)
    
    # Exact evidence
    exact_log_ev = model.log_evidence(data)
    
    # Laplace approximation
    init_params = np.array([data.mean(), data.var()])
    laplace_log_ev, map_params, hess = laplace_approximation(
        model, data, init_params,
        param_bounds=[(None, None), (0.01, None)]
    )
    
    print(f"\nExact log evidence: {exact_log_ev:.4f}")
    print(f"Laplace approximation: {laplace_log_ev:.4f}")
    print(f"Difference: {abs(exact_log_ev - laplace_log_ev):.4f}")
    print(f"\nMAP estimates: μ = {map_params[0]:.3f}, σ² = {map_params[1]:.3f}")


def demo_model_comparison():
    """Demonstrate model comparison via evidence."""
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data from biased coin
    true_theta = 0.65
    n = 100
    data = np.random.binomial(1, true_theta, n)
    
    print(f"Data: {data.sum()} successes in {n} trials")
    
    # Compare different priors
    models = {
        'Fair coin (α=50, β=50)': BetaBernoulliModel(50, 50),
        'Slight bias allowed (α=5, β=5)': BetaBernoulliModel(5, 5),
        'Uniform (α=1, β=1)': BetaBernoulliModel(1, 1),
        'Biased prior (α=6, β=4)': BetaBernoulliModel(6, 4),
    }
    
    log_evidences = {}
    for name, model in models.items():
        log_ev = model.log_evidence(data)
        log_evidences[name] = log_ev
        print(f"\n{name}:")
        print(f"  Log evidence: {log_ev:.4f}")
    
    # Compute model probabilities
    names = list(log_evidences.keys())
    log_evs = list(log_evidences.values())
    probs = compute_model_probabilities(log_evs)
    
    print("\n--- Posterior Model Probabilities ---")
    for name, prob in zip(names, probs):
        print(f"  {name}: {prob:.4f}")
    
    # Bayes factor interpretation
    best_idx = np.argmax(log_evs)
    print(f"\n--- Bayes Factors vs Best Model ({names[best_idx]}) ---")
    for i, name in enumerate(names):
        if i != best_idx:
            log_bf = log_evs[best_idx] - log_evs[i]
            interp = interpret_bayes_factor(log_bf)
            print(f"  vs {name}: {interp}")


def demo_occam_razor():
    """Demonstrate Occam's razor in polynomial regression."""
    
    print("\n" + "=" * 70)
    print("OCCAM'S RAZOR DEMONSTRATION")
    print("=" * 70)
    
    fig = plot_occam_razor_demo([20, 100, 500])
    fig.savefig('occam_razor_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: occam_razor_demo.png")
    print("\nAs sample size increases, evidence concentrates on true model (degree 2)")


if __name__ == "__main__":
    demo_conjugate_evidence()
    demo_laplace_approximation()
    demo_model_comparison()
    demo_occam_razor()
```

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Definition** | $p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta$ |
| **Interpretation** | Average likelihood under prior; predictive probability of data |
| **Role** | Normalizing constant in Bayes' theorem; basis for model comparison |
| **Occam's Razor** | Automatically penalizes unnecessary complexity |
| **Decomposition** | $\log p(\mathcal{D}) = \text{Fit} - \text{Complexity}$ |

### Computation Methods

| Method | Applicability | Accuracy | Cost |
|--------|--------------|----------|------|
| Exact (conjugate) | Exponential family | Exact | Low |
| Laplace approximation | Smooth posteriors | Good for large $n$ | Medium |
| BIC | Large samples | Asymptotically correct | Low |
| Importance sampling | General | Depends on proposal | Medium-High |
| Nested sampling | General | High | High |

### Key Insights

1. **Marginal over parameters**: Evidence integrates out all parameters
2. **Occam's razor**: Complex models are penalized automatically
3. **Prior sensitivity**: Evidence depends strongly on prior (unlike posterior)
4. **Sequential decomposition**: $\log p(\mathcal{D}) = \sum_t \log p(x_t \mid x_{1:t-1})$
5. **Model averaging**: Use evidence weights for prediction
6. **Improper priors**: Cannot be used for model comparison

### Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Bayes factors | Ch13: Bayes Factor | Ratio of evidences |
| Information criteria | Ch13: Information Criteria | BIC approximates evidence |
| Conjugate models | Ch13: Distributions | Exact evidence available |
| BNN model selection | Ch13: BNN | Selecting architectures |
| Cross-validation | Ch7: Model Selection | Alternative to evidence |

### Key References

- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Chapter 28.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapter 7.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Section 3.4.
