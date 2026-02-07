# Risk Estimation with Model Uncertainty

## Overview

Uncertainty-aware deep learning models provide natural risk estimates for financial applications. By propagating model uncertainty through portfolio calculations, we obtain more honest assessments of tail risk.

## Uncertainty-Aware Value-at-Risk

Standard VaR uses point estimates. Uncertainty-aware VaR integrates over model uncertainty:

$$\text{VaR}_\alpha = \inf\{l : P(L > l | \mathcal{D}) \leq 1 - \alpha\}$$

where the loss distribution $P(L|\mathcal{D})$ is computed by marginalizing over model uncertainty.

## Implementation

```python
import torch
import numpy as np
from typing import Dict


def uncertainty_aware_var(
    model_predictions: torch.Tensor,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute VaR incorporating model uncertainty.
    
    Args:
        model_predictions: (M, T) - M model samples, T time steps
        alpha: VaR confidence level
    """
    # Standard VaR from mean prediction
    mean_pred = model_predictions.mean(dim=0)
    standard_var = np.percentile(mean_pred.numpy(), alpha * 100)
    
    # Uncertainty-aware VaR: sample from predictive distribution
    all_samples = model_predictions.flatten()
    uncertain_var = np.percentile(all_samples.numpy(), alpha * 100)
    
    # Expected Shortfall (CVaR)
    tail_mask = all_samples <= uncertain_var
    if tail_mask.sum() > 0:
        es = all_samples[tail_mask].mean().item()
    else:
        es = uncertain_var
    
    return {
        'var_standard': standard_var,
        'var_uncertain': uncertain_var,
        'expected_shortfall': es,
        'var_gap': uncertain_var - standard_var,
    }


def model_risk_decomposition(
    ensemble_predictions: torch.Tensor,
    y_true: torch.Tensor = None
) -> Dict[str, float]:
    """
    Decompose prediction risk into model uncertainty and market risk.
    """
    mean_pred = ensemble_predictions.mean(dim=0)
    
    # Model risk: uncertainty in the prediction itself
    model_risk = ensemble_predictions.var(dim=0).mean().item()
    
    # If true values available, compute realized vs predicted
    if y_true is not None:
        prediction_error = ((mean_pred - y_true) ** 2).mean().item()
    else:
        prediction_error = None
    
    return {
        'model_risk': model_risk,
        'prediction_error': prediction_error,
        'ensemble_spread': ensemble_predictions.std(dim=0).mean().item(),
    }
```

## Key Insight for Finance

Models that report uncertainty enable **adaptive risk budgeting**: when model uncertainty is high (regime change, novel conditions), automatically reduce exposure. This simple rule significantly improves risk-adjusted returns.

## References

- Zhu, Y., & Laptev, N. (2017). "Deep and Confident Prediction for Time Series at Uber." IEEE ICDM.
