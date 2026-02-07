# Parameter Uncertainty in Finance

## Overview

Financial models rely on estimated parameters — expected returns, volatilities, correlations, factor loadings — that carry substantial estimation uncertainty. Bayesian methods quantify this uncertainty and propagate it through to decisions, yielding more robust portfolio construction, risk management, and strategy evaluation.

---

## Estimation Risk

### The Problem

Consider estimating the expected return $\mu$ of an asset from $T$ monthly observations. The standard error of the sample mean is $\sigma / \sqrt{T}$. For a typical stock with annualized volatility $\sigma \approx 0.30$ (monthly $\approx 0.087$):

| Estimation Window | Standard Error (annualized) | 95% CI Width |
|-------------------|-----------------------------|---------------|
| 5 years (60 months) | 3.87% | ±7.6% |
| 10 years (120 months) | 2.74% | ±5.4% |
| 20 years (240 months) | 1.94% | ±3.8% |

Expected return estimates are **extremely noisy** relative to typical risk premia (3-8% annualized). This is the fundamental challenge of quantitative investing.

### Bayesian Treatment

The Bayesian posterior for the mean return under a Normal-Normal model with diffuse prior:

$$
\mu \mid \mathcal{D} \sim t_{T-1}\left(\bar{r}, \frac{s^2}{T}\right)
$$

The predictive distribution for future returns has fatter tails:

$$
r_{\text{new}} \mid \mathcal{D} \sim t_{T-1}\left(\bar{r}, s^2\left(1 + \frac{1}{T}\right)\right)
$$

The additional variance $s^2/T$ reflects **estimation risk** — uncertainty about the true mean itself.

---

## Posterior Predictive Returns

### Single Asset

For a single asset with $T$ observations of returns $\{r_1, \ldots, r_T\}$, the posterior predictive distribution integrates over parameter uncertainty:

$$
p(r_{\text{new}} \mid \mathcal{D}) = \int p(r_{\text{new}} \mid \mu, \sigma^2) \, p(\mu, \sigma^2 \mid \mathcal{D}) \, d\mu \, d\sigma^2
$$

Under the conjugate Normal-Inverse-Gamma model, this yields a $t$-distribution with $T - 1$ degrees of freedom.

### Portfolio Level

For a portfolio of $N$ assets, the predictive covariance matrix is:

$$
\tilde{\boldsymbol{\Sigma}} = \hat{\boldsymbol{\Sigma}} \cdot \frac{T-1}{T-N-2} \cdot \left(1 + \frac{1}{T}\right)
$$

The inflation factor $\frac{T-1}{T-N-2}$ grows dramatically as $N$ approaches $T$, reflecting the curse of dimensionality in covariance estimation.

---

## Shrinkage Estimators

Bayesian parameter uncertainty naturally leads to **shrinkage** — pulling estimates toward a structured prior:

### Ledoit-Wolf Shrinkage

The shrinkage covariance estimator:

$$
\hat{\boldsymbol{\Sigma}}_{\text{shrink}} = \delta \mathbf{F} + (1-\delta) \hat{\boldsymbol{\Sigma}}_{\text{sample}}
$$

where $\mathbf{F}$ is the structured target (e.g., single-factor model) and $\delta$ is the optimal shrinkage intensity.

This has a Bayesian interpretation: the structured target $\mathbf{F}$ is the prior, the sample covariance is the data, and $\delta$ reflects the relative precision.

### James-Stein Shrinkage for Returns

The James-Stein estimator shrinks expected returns toward a common mean:

$$
\hat{\mu}_i^{\text{JS}} = \bar{\mu} + (1 - \hat{c}) (\hat{\mu}_i - \bar{\mu})
$$

This is the empirical Bayes solution under a Normal hierarchical model (see [Empirical Bayes](../hierarchical/empirical_bayes.md)).

---

## PyTorch Implementation

```python
import torch
import torch.distributions as dist


class BayesianReturnEstimator:
    """
    Bayesian estimation of return parameters with uncertainty propagation.
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_precision: float = 0.01):
        self.mu_0 = prior_mean
        self.kappa_0 = prior_precision  # precision of prior mean
    
    def fit(self, returns: torch.Tensor):
        """
        Compute posterior parameters for multivariate returns.
        
        Parameters
        ----------
        returns : (T, N) tensor of return observations
        """
        T, N = returns.shape
        
        self.T = T
        self.N = N
        self.sample_mean = returns.mean(dim=0)
        self.sample_cov = torch.cov(returns.T)
        
        # Posterior mean (precision-weighted)
        posterior_precision = self.kappa_0 + T
        self.posterior_mean = (
            self.kappa_0 * self.mu_0 + T * self.sample_mean
        ) / posterior_precision
        
        # Predictive covariance (with estimation uncertainty)
        if T > N + 2:
            inflation = (T - 1) / (T - N - 2) * (1 + 1/T)
        else:
            inflation = 2.0  # fallback for small samples
        self.predictive_cov = self.sample_cov * inflation
        
        return self
    
    def predictive_sharpe(self, weights: torch.Tensor) -> dict:
        """
        Compute posterior distribution of portfolio Sharpe ratio.
        """
        port_mean = weights @ self.posterior_mean
        port_var = weights @ self.predictive_cov @ weights
        port_std = port_var.sqrt()
        
        sharpe = port_mean / port_std
        
        # Approximate standard error of Sharpe ratio
        sharpe_se = ((1 + 0.5 * sharpe**2) / self.T).sqrt()
        
        return {
            'sharpe': sharpe.item(),
            'sharpe_se': sharpe_se.item(),
            'prob_positive': dist.Normal(sharpe, sharpe_se).cdf(
                torch.tensor(0.0)).item()
        }
```

---

## Key Insights for Practice

1. **Expected returns are estimated with far less precision than volatilities or correlations.** This asymmetry should inform model design — be skeptical of return forecasts, more trusting of risk estimates.

2. **Bayesian predictive distributions have fatter tails** than plug-in Gaussian models, naturally accounting for estimation risk and producing more conservative VaR/CVaR estimates.

3. **Shrinkage toward structured targets** (factor models, equal-weight) is not an ad hoc regularization trick — it is the optimal Bayesian response to parameter uncertainty.

4. **Sample size requirements grow with dimensionality.** With $N$ assets and $T$ observations, estimation quality degrades rapidly as $N/T$ increases.

---

## References

- Barberis, N. (2000). Investing for the long run when returns are predictable. *Journal of Finance*, 55(1), 225-264.
- Kan, R., & Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *Journal of Financial and Quantitative Analysis*, 42(3), 621-656.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
