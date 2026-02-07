# Bayesian Portfolio Optimization

## Overview

Classical mean-variance portfolio optimization (Markowitz, 1952) treats estimated parameters as known constants, ignoring estimation uncertainty. Bayesian portfolio methods incorporate parameter uncertainty directly into the optimization, producing more robust and better-diversified portfolios.

---

## The Estimation Problem in Classical Optimization

The mean-variance optimal portfolio solves:

$$
\mathbf{w}^* = \frac{1}{\gamma} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}
$$

where $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ are the expected returns and covariance matrix. When these are replaced by sample estimates $\hat{\boldsymbol{\mu}}$ and $\hat{\boldsymbol{\Sigma}}$, the resulting "plug-in" portfolio suffers from:

- **Estimation error amplification**: $\hat{\boldsymbol{\Sigma}}^{-1}$ amplifies errors in $\hat{\boldsymbol{\mu}}$
- **Extreme positions**: Portfolios concentrate in assets with the largest estimated returns
- **Instability**: Small changes in data produce dramatically different allocations

---

## Bayesian Approach

### Predictive Returns Distribution

Instead of using point estimates, the Bayesian approach integrates over parameter uncertainty:

$$
p(\mathbf{r}_{\text{new}} \mid \mathcal{D}) = \int p(\mathbf{r}_{\text{new}} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, p(\boldsymbol{\mu}, \boldsymbol{\Sigma} \mid \mathcal{D}) \, d\boldsymbol{\mu} \, d\boldsymbol{\Sigma}
$$

Under a conjugate Normal-Inverse-Wishart prior, the predictive distribution is a multivariate $t$-distribution with fatter tails than the Gaussian — naturally accounting for estimation risk.

### Bayesian Optimal Portfolio

The Bayesian portfolio maximizes expected utility under the **predictive** distribution:

$$
\mathbf{w}^*_{\text{Bayes}} = \arg\max_{\mathbf{w}} \mathbb{E}_{p(\mathbf{r} \mid \mathcal{D})}[U(\mathbf{w}^\top \mathbf{r})]
$$

For quadratic utility, this yields:

$$
\mathbf{w}^*_{\text{Bayes}} = \frac{1}{\gamma} \tilde{\boldsymbol{\Sigma}}^{-1} \tilde{\boldsymbol{\mu}}
$$

where $\tilde{\boldsymbol{\mu}}$ and $\tilde{\boldsymbol{\Sigma}}$ are the mean and covariance of the predictive distribution.

---

## Black-Litterman Model

The Black-Litterman (1992) model is the most widely used Bayesian portfolio framework. It combines a market equilibrium prior with investor views:

### Prior: Market Equilibrium Returns

$$
\boldsymbol{\pi} = \gamma \boldsymbol{\Sigma} \mathbf{w}_{\text{mkt}}
$$

where $\mathbf{w}_{\text{mkt}}$ are market capitalization weights.

### Views

Investor views are expressed as:

$$
\mathbf{P} \boldsymbol{\mu} = \mathbf{q} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})
$$

where $\mathbf{P}$ is the pick matrix, $\mathbf{q}$ is the vector of expected view returns, and $\boldsymbol{\Omega}$ represents view uncertainty.

### Posterior

$$
\hat{\boldsymbol{\mu}}_{\text{BL}} = \left[(\tau \boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{P}\right]^{-1} \left[(\tau \boldsymbol{\Sigma})^{-1} \boldsymbol{\pi} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{q}\right]
$$

This is a precision-weighted average of the equilibrium prior and investor views — exactly the conjugate Normal-Normal update from Section 16.1.

### PyTorch Implementation

```python
import torch


def black_litterman(
    sigma: torch.Tensor,
    w_mkt: torch.Tensor,
    P: torch.Tensor,
    q: torch.Tensor,
    omega: torch.Tensor,
    gamma: float = 2.5,
    tau: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Black-Litterman posterior returns.
    
    Parameters
    ----------
    sigma : (N, N) covariance matrix
    w_mkt : (N,) market cap weights
    P : (K, N) pick matrix for K views
    q : (K,) expected view returns
    omega : (K, K) view uncertainty
    gamma : risk aversion coefficient
    tau : scaling factor for prior uncertainty
    
    Returns
    -------
    mu_bl : (N,) posterior expected returns
    sigma_bl : (N, N) posterior covariance
    """
    # Equilibrium returns
    pi = gamma * sigma @ w_mkt
    
    # Prior precision
    tau_sigma_inv = torch.linalg.inv(tau * sigma)
    omega_inv = torch.linalg.inv(omega)
    
    # Posterior precision and mean
    posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
    sigma_bl = torch.linalg.inv(posterior_precision)
    mu_bl = sigma_bl @ (tau_sigma_inv @ pi + P.T @ omega_inv @ q)
    
    return mu_bl, sigma_bl
```

---

## Summary

| Approach | Returns Estimate | Uncertainty | Diversification |
|----------|-----------------|-------------|-----------------|
| Markowitz (plug-in) | Sample mean | Ignored | Poor |
| Bayesian (diffuse prior) | Posterior predictive mean | Full posterior | Better |
| Black-Litterman | Equilibrium + views | View confidence | Best |
| Robust optimization | Uncertainty set | Worst-case | Conservative |

---

## References

- Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28-43.
- Meucci, A. (2005). *Risk and Asset Allocation*. Springer.
- Avramov, D., & Zhou, G. (2010). Bayesian portfolio analysis. *Annual Review of Financial Economics*, 2, 25-47.
