# Diversity Methods for Deep Ensembles

## Overview

Ensemble uncertainty quality depends critically on diversity. Members must disagree on uncertain inputs to capture epistemic uncertainty.

## Sources of Diversity

### Random Initialization

Different initializations explore different loss landscape modes. This is the primary and often sufficient source of diversity.

### Data Diversity

- **Bootstrapping**: Each member trains on a random subset with replacement
- **Feature bagging**: Each member sees a random subset of input features
- **Augmentation diversity**: Different augmentation policies per member

### Architecture Diversity

Different hidden sizes, depths, or activation functions:

```python
import torch.nn as nn


def create_diverse_architectures(input_dim, output_dim, n_members=5):
    configs = [
        [256, 128],
        [512, 256, 128],
        [128, 128, 128, 128],
        [384, 192],
        [256, 256, 128, 64],
    ]
    
    members = nn.ModuleList()
    for m in range(n_members):
        config = configs[m % len(configs)]
        layers = []
        prev = input_dim
        for h in config:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        members.append(nn.Sequential(*layers))
    
    return members
```

## Measuring Diversity

### Jensen-Shannon Divergence

$$\text{JSD}(p_1, \ldots, p_M) = H\left(\frac{1}{M}\sum_m p_m\right) - \frac{1}{M}\sum_m H(p_m)$$

```python
import torch


def compute_ensemble_diversity(all_probs):
    epsilon = 1e-10
    mean_probs = all_probs.mean(dim=0)
    H_mean = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)
    H_individual = -torch.sum(all_probs * torch.log(all_probs + epsilon), dim=-1)
    mean_H = H_individual.mean(dim=0)
    jsd = (H_mean - mean_H).mean().item()
    
    preds = all_probs.argmax(dim=-1)
    M = all_probs.shape[0]
    disagreement = 0
    for i in range(M):
        for j in range(i+1, M):
            disagreement += (preds[i] != preds[j]).float().mean().item()
    disagreement /= (M * (M - 1) / 2)
    
    return {'jsd': jsd, 'disagreement': disagreement}
```

## Regularization for Diversity

### Negative Correlation Learning

Add a penalty encouraging diverse errors among members.

### Repulsive Ensembles

D'Angelo & Fortuin (2021) add a repulsive term in function space to encourage members to cover different posterior modes.

## Key Takeaways

!!! success "Summary"
    1. **Random initialization** is often sufficient for diversity
    2. **Architecture diversity** adds benefit for challenging problems
    3. **JSD and disagreement** quantify diversity
    4. **Diversity without accuracy** is useless—each member must be competent

## References

- Fort, S., et al. (2019). "Deep Ensembles: A Loss Landscape Perspective." arXiv.
- D'Angelo, F., & Fortuin, V. (2021). "Repulsive Deep Ensembles are Bayesian." NeurIPS.
