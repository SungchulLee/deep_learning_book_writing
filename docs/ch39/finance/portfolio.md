# Portfolio Uncertainty

## Overview

Portfolio construction under model uncertainty requires propagating prediction uncertainty through the optimization. This produces more robust allocations that account for the model's confidence in its forecasts.

## Uncertainty-Aware Portfolio Optimization

Given return predictions $\hat{\mu}_i$ with uncertainty $\hat{\sigma}_i$ for each asset $i$:

### Conservative Allocation

Shrink expected returns toward zero proportionally to uncertainty:

$$\tilde{\mu}_i = \hat{\mu}_i \cdot \frac{\text{conf}_i}{\text{conf}_i + \lambda}$$

where $\text{conf}_i = 1 / \hat{\sigma}_i^2$ is the precision of the prediction.

### Black-Litterman with Model Uncertainty

Treat model predictions as "views" in Black-Litterman, with the uncertainty matrix $\Omega$ derived from the model's epistemic uncertainty.

## Implementation

```python
import torch
import numpy as np


def uncertainty_weighted_portfolio(
    expected_returns: torch.Tensor,
    return_uncertainty: torch.Tensor,
    risk_aversion: float = 1.0,
    max_position: float = 0.2
) -> torch.Tensor:
    """
    Allocate capital inversely proportional to prediction uncertainty.
    """
    # Precision-weighted returns
    precision = 1.0 / (return_uncertainty ** 2 + 1e-8)
    adjusted_returns = expected_returns * precision / (precision + risk_aversion)
    
    # Simple allocation: proportional to adjusted returns
    weights = adjusted_returns / (adjusted_returns.abs().sum() + 1e-8)
    
    # Clip to max position
    weights = weights.clamp(-max_position, max_position)
    
    return weights


def ensemble_portfolio(
    ensemble_returns: torch.Tensor,
    risk_aversion: float = 1.0
) -> dict:
    """
    Portfolio from ensemble predictions with uncertainty analysis.
    
    Args:
        ensemble_returns: (M, n_assets) predicted returns from M members
    """
    mean_returns = ensemble_returns.mean(dim=0)
    uncertainty = ensemble_returns.std(dim=0)
    
    weights = uncertainty_weighted_portfolio(
        mean_returns, uncertainty, risk_aversion
    )
    
    # Portfolio-level uncertainty via delta method
    portfolio_return = (weights * mean_returns).sum()
    portfolio_uncertainty = torch.sqrt(
        (weights ** 2 * uncertainty ** 2).sum()
    )
    
    return {
        'weights': weights,
        'expected_return': portfolio_return.item(),
        'uncertainty': portfolio_uncertainty.item(),
        'sharpe_approx': (portfolio_return / portfolio_uncertainty).item(),
    }
```

## References

- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." Financial Analysts Journal.
