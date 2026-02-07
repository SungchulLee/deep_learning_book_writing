# Proper Scoring Rules

## Overview

A scoring rule $S(p, y)$ assigns a numerical score to a probabilistic prediction $p$ given the observed outcome $y$. A scoring rule is **proper** if the expected score is maximized when the forecaster reports their true beliefs. A scoring rule is **strictly proper** if the maximum is unique.

## Common Proper Scoring Rules

### Log Score (Negative Log-Likelihood)

$$S_{\text{log}}(p, y) = -\log p(y)$$

The most commonly used proper scoring rule in deep learning. Unbounded penalty for confident wrong predictions.

### Brier Score

$$S_{\text{Brier}}(p, y) = \sum_{c=1}^C (p_c - \mathbb{1}[y=c])^2$$

Bounded in $[0, 2]$, less sensitive to extreme probabilities than log score.

### CRPS (Continuous Ranked Probability Score)

For continuous predictions with CDF $F$:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbb{1}[z \geq y])^2 dz$$

Particularly relevant for regression uncertainty in financial forecasting.

## Brier Score Decomposition

The Brier score decomposes into calibration, refinement (sharpness), and uncertainty:

$$\text{BS} = \underbrace{\frac{1}{N}\sum_m n_m(\bar{p}_m - \bar{o}_m)^2}_{\text{Calibration}} - \underbrace{\frac{1}{N}\sum_m n_m(\bar{o}_m - \bar{o})^2}_{\text{Refinement}} + \underbrace{\bar{o}(1-\bar{o})}_{\text{Uncertainty}}$$

This decomposition reveals whether poor scores come from miscalibration or lack of sharpness.

## Implementation

```python
import torch
import numpy as np


def log_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Negative log-likelihood (lower is better)."""
    return -torch.log(probs[range(len(labels)), labels] + 1e-10).mean().item()


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Brier score (lower is better)."""
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor,
                  y: torch.Tensor) -> float:
    """CRPS for Gaussian predictive distribution."""
    z = (y - mu) / sigma
    phi = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    pdf = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    crps = sigma * (z * (2 * phi - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return crps.mean().item()
```

## Why Proper Scoring Rules Matter

Improper scoring rules can be gamed—a forecaster can improve their score by reporting dishonest probabilities. Proper scoring rules ensure that honest reporting is optimal, which is essential for training well-calibrated models.

## References

- Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." JASA.
- Bröcker, J. (2009). "Reliability, Sufficiency, and the Decomposition of Proper Scores." QJR Meteorol Soc.
