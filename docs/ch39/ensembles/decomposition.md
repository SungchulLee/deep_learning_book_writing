# Uncertainty Decomposition in Deep Ensembles

## Overview

Deep ensembles naturally decompose predictive uncertainty into epistemic and aleatoric components.

## Regression Decomposition

For $M$ members each predicting $(\mu_m, \sigma_m^2)$:

**Predictive mean**: $\bar{\mu} = \frac{1}{M}\sum_m \mu_m$

**Total predictive variance**:

$$\bar{\sigma}^2 = \underbrace{\frac{1}{M}\sum_m \sigma_m^2}_{\text{Aleatoric}} + \underbrace{\frac{1}{M}\sum_m (\mu_m - \bar{\mu})^2}_{\text{Epistemic}}$$

## Classification Decomposition

**Total** (predictive entropy): $\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$

**Aleatoric** (expected entropy): $\frac{1}{M}\sum_m (-\sum_c p_{mc} \log p_{mc})$

**Epistemic** (mutual information): $\text{Total} - \text{Aleatoric}$

## Implementation

```python
import torch
from typing import Dict


def decompose_regression(means, log_vars):
    pred_mean = means.mean(dim=0)
    epistemic = means.var(dim=0)
    aleatoric = log_vars.exp().mean(dim=0)
    return {'mean': pred_mean, 'epistemic': epistemic,
            'aleatoric': aleatoric, 'total': epistemic + aleatoric}


def decompose_classification(all_probs):
    epsilon = 1e-10
    mean_probs = all_probs.mean(dim=0)
    
    total = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)
    member_H = -torch.sum(all_probs * torch.log(all_probs + epsilon), dim=-1)
    aleatoric = member_H.mean(dim=0)
    epistemic = total - aleatoric
    
    return {'total': total, 'epistemic': epistemic,
            'aleatoric': aleatoric, 'probs': mean_probs}
```

## Finance Implications

| Component | High Value | Action |
|-----------|-----------|--------|
| High epistemic | Regime change | Reduce position |
| High aleatoric | Volatile asset | Use in risk budget |
| Both high | Data quality issue | Investigate |

## References

- Depeweg, S., et al. (2018). "Decomposition of Uncertainty in Bayesian Deep Learning." ICML.
