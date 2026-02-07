# Efficient Ensembles

## Overview

Standard deep ensembles require $M\times$ training and inference compute. This section covers techniques to reduce these costs while retaining uncertainty quality.

## BatchEnsemble

BatchEnsemble (Wen et al., 2020) shares most parameters across members, adding only rank-1 perturbations per member:

$$W_m = W \odot (r_m s_m^T)$$

where $W$ is the shared weight matrix and $r_m, s_m$ are member-specific vectors. This reduces the per-member parameter overhead from $O(d_{\text{in}} \times d_{\text{out}})$ to $O(d_{\text{in}} + d_{\text{out}})$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchEnsembleLinear(nn.Module):
    """Shared weights with rank-1 member perturbations."""
    
    def __init__(self, in_features, out_features, n_members=5):
        super().__init__()
        self.n_members = n_members
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.r = nn.Parameter(torch.ones(n_members, in_features))
        self.s = nn.Parameter(torch.ones(n_members, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, member_idx=None):
        if member_idx is not None:
            x_scaled = x * self.r[member_idx]
            return F.linear(x_scaled, self.weight, self.bias) * self.s[member_idx]
        else:
            outputs = []
            for m in range(self.n_members):
                out = F.linear(x * self.r[m], self.weight, self.bias) * self.s[m]
                outputs.append(out)
            return torch.stack(outputs)
```

## Snapshot Ensembles

Collect multiple models from a single training run using cyclic learning rates. At the end of each cycle, save a snapshot. Costs approximately the same as training one model.

## Stochastic Weight Averaging Gaussian (SWAG)

Fit a low-rank Gaussian to the SGD trajectory. Provides ensemble-like uncertainty from a single training run with minimal memory overhead. See the SWAG section in [Bayesian Methods](../bayesian_methods/fundamentals.md).

## Multi-Input Multi-Output (MIMO)

Train a single network with $M$ input-output heads sharing intermediate layers. Each head sees a different permutation of the training data, providing diversity through the shared representation.

## Hyperparameter Ensembles

Instead of identical architectures, combine models with different hyperparameters (learning rates, weight decay, dropout rates). This is naturally available when doing hyperparameter search.

## Cost Comparison

| Method | Training Cost | Inference Cost | Memory | Uncertainty Quality |
|--------|--------------|---------------|--------|-------------------|
| Standard Ensemble | $M \times$ | $M \times$ | $M \times$ | Best |
| BatchEnsemble | $\sim 1.1 \times$ | $\sim 1.1 \times$ | $\sim 1.05 \times$ | Good |
| Snapshot Ensemble | $1 \times$ | $M \times$ | $M \times$ | Moderate |
| SWAG | $1.1 \times$ | $T \times$ | $\sim 1.2 \times$ | Good |
| MIMO | $\sim 1.2 \times$ | $1 \times$ | $\sim 1.1 \times$ | Moderate |

## Recommendations for Finance

For latency-sensitive trading applications, BatchEnsemble or MIMO provide the best uncertainty-per-FLOP ratio. For offline risk estimation where inference cost is less critical, standard ensembles remain the gold standard.

## References

- Wen, Y., et al. (2020). "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning." ICLR.
- Huang, G., et al. (2017). "Snapshot Ensembles: Train 1, Get M for Free." ICLR.
- Havasi, M., et al. (2021). "Training Independent Subnetworks for Robust Prediction." ICLR.
