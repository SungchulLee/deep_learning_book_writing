# Coverage Analysis

## Overview

Coverage measures the fraction of true values falling within prediction intervals. For a model producing $\alpha$-level prediction intervals, the empirical coverage should be approximately $\alpha$.

## Definition

For prediction intervals $[l_i, u_i]$ at confidence level $1-\alpha$:

$$\text{Coverage} = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[l_i \leq y_i \leq u_i]$$

Target: Coverage $\approx 1 - \alpha$ (e.g., 95% for $\alpha = 0.05$).

## Implementation

```python
import torch
import numpy as np
from typing import Dict


def compute_coverage(
    y_true: torch.Tensor,
    y_pred_mean: torch.Tensor,
    y_pred_std: torch.Tensor,
    alphas: list = [0.05, 0.10, 0.20]
) -> Dict[str, float]:
    """
    Compute empirical coverage at multiple confidence levels.
    Assumes Gaussian predictive distribution.
    """
    from scipy.stats import norm
    results = {}
    
    for alpha in alphas:
        z = norm.ppf(1 - alpha / 2)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        
        covered = ((y_true >= lower) & (y_true <= upper)).float()
        empirical = covered.mean().item()
        target = 1 - alpha
        
        results[f'coverage_{int((1-alpha)*100)}'] = empirical
        results[f'gap_{int((1-alpha)*100)}'] = empirical - target
    
    return results


def compute_interval_width(
    y_pred_std: torch.Tensor, alpha: float = 0.05
) -> float:
    """Mean prediction interval width."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return (2 * z * y_pred_std).mean().item()
```

## Coverage-Width Trade-off

Wider intervals trivially achieve higher coverage. The **prediction interval normalized average width (PINAW)** evaluates interval quality:

$$\text{PINAW} = \frac{1}{N(y_{\max} - y_{\min})}\sum_{i=1}^N (u_i - l_i)$$

Good models achieve target coverage with minimal width.

## References

- Pearce, T., et al. (2018). "High-Quality Prediction Intervals for Deep Learning." ICML.
