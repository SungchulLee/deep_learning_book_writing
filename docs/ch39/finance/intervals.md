# Prediction Intervals for Financial Time Series

## Overview

Prediction intervals quantify the range of plausible future outcomes, incorporating both model uncertainty (epistemic) and inherent market noise (aleatoric). Well-calibrated prediction intervals are essential for risk management.

## Constructing Prediction Intervals

### From Ensemble Predictions

Given ensemble predictions $\{\hat{y}_m\}_{m=1}^M$:

**Gaussian approximation**: $[\bar{y} - z_{\alpha/2} \hat{\sigma}, \; \bar{y} + z_{\alpha/2} \hat{\sigma}]$

**Quantile-based**: Use empirical quantiles of ensemble predictions for non-Gaussian intervals.

### From Heteroscedastic Models

Each model predicts $(\mu, \sigma^2)$, providing input-dependent intervals that widen during volatile periods.

## Implementation

```python
import torch
import numpy as np
from scipy.stats import norm


def gaussian_prediction_interval(
    mean: torch.Tensor, std: torch.Tensor, alpha: float = 0.05
) -> tuple:
    """Gaussian prediction interval."""
    z = norm.ppf(1 - alpha / 2)
    return mean - z * std, mean + z * std


def quantile_prediction_interval(
    ensemble_preds: torch.Tensor, alpha: float = 0.05
) -> tuple:
    """Non-parametric interval from ensemble quantiles."""
    lower = torch.quantile(ensemble_preds, alpha / 2, dim=0)
    upper = torch.quantile(ensemble_preds, 1 - alpha / 2, dim=0)
    return lower, upper


def conformal_prediction_interval(
    cal_residuals: torch.Tensor, test_mean: torch.Tensor,
    test_std: torch.Tensor, alpha: float = 0.05
) -> tuple:
    """
    Conformal prediction interval with finite-sample coverage guarantee.
    
    1. Compute normalized residuals on calibration set
    2. Find the (1-alpha) quantile
    3. Scale test intervals by this quantile
    """
    n = len(cal_residuals)
    q = np.ceil((1 - alpha) * (n + 1)) / n
    quantile = torch.quantile(cal_residuals.abs(), min(q, 1.0))
    
    return test_mean - quantile * test_std, test_mean + quantile * test_std
```

## Financial Considerations

- **Volatility clustering**: Prediction intervals should widen during volatile periods
- **Fat tails**: Gaussian intervals underestimate tail risk; use quantile-based or conformal methods
- **Non-stationarity**: Intervals should adapt to changing market conditions
- **Asymmetry**: Downside risk often differs from upside—consider asymmetric intervals

## References

- Romano, Y., et al. (2019). "Conformalized Quantile Regression." NeurIPS.
