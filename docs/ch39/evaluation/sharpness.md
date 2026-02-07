# Sharpness

## Overview

Sharpness measures how concentrated (peaked) the predictive distributions are. A sharp model makes confident predictions, while a diffuse model hedges broadly. The goal is to be **as sharp as possible while maintaining calibration**—known as the sharpness-calibration trade-off.

## Definition

For classification, sharpness is the negative average entropy:

$$\text{Sharpness} = -\frac{1}{N}\sum_{i=1}^N \mathbb{H}[\hat{p}_i] = \frac{1}{N}\sum_{i=1}^N \sum_c \hat{p}_{ic} \log \hat{p}_{ic}$$

For regression with Gaussian predictions:

$$\text{Sharpness} = \frac{1}{N}\sum_{i=1}^N \hat{\sigma}_i^2$$

Lower variance = sharper predictions.

## Implementation

```python
import torch
import numpy as np


def compute_sharpness_classification(probs: torch.Tensor) -> float:
    """Sharpness = negative mean entropy. Higher = sharper."""
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return -entropy.mean().item()


def compute_sharpness_regression(pred_std: torch.Tensor) -> float:
    """Sharpness = mean predicted variance. Lower = sharper."""
    return (pred_std ** 2).mean().item()
```

## The Sharpness-Calibration Trade-off

A trivially calibrated model reports the base rate for every input—perfectly calibrated but completely useless (zero sharpness). The challenge is to be maximally informative (sharp) while maintaining honest probabilities (calibrated).

Temperature scaling controls this trade-off: higher $T$ sacrifices sharpness for better calibration, while lower $T$ increases sharpness at the risk of overconfidence.

## References

- Gneiting, T., et al. (2007). "Probabilistic Forecasts, Calibration and Sharpness." JRSS-B.
