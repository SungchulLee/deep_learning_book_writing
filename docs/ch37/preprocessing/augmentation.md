# Data Augmentation for Fairness

## Overview

Data augmentation for fairness creates synthetic samples to balance the joint distribution of protected attributes and labels, reducing bias in the training data without modifying existing samples.

## Strategies

### 1. Oversampling Underrepresented Subgroups

For each $(A, Y)$ combination, oversample minority subgroups to match the majority:

```python
import torch
import numpy as np

def fairness_oversampling(
    X: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
) -> tuple:
    """Oversample underrepresented (A, Y) subgroups."""
    subgroups = {}
    for a in torch.unique(A):
        for yv in torch.unique(y):
            mask = (A == a) & (y == yv)
            subgroups[(a.item(), yv.item())] = mask.nonzero().squeeze(-1)
    
    max_size = max(len(idx) for idx in subgroups.values())
    
    indices = []
    for key, idx in subgroups.items():
        if len(idx) < max_size:
            # Oversample with replacement
            extra = torch.tensor(
                np.random.choice(idx.numpy(), max_size - len(idx), replace=True)
            )
            idx = torch.cat([idx, extra])
        indices.append(idx)
    
    all_idx = torch.cat(indices)
    perm = torch.randperm(len(all_idx))
    all_idx = all_idx[perm]
    
    return X[all_idx], y[all_idx], A[all_idx]
```

### 2. Counterfactual Data Augmentation

Generate counterfactual samples by flipping the protected attribute and adjusting features accordingly:

```python
def counterfactual_augmentation(
    X: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    a_idx: int,
    descendant_indices: list,
) -> tuple:
    """
    Generate counterfactual samples by flipping A and adjusting
    descendant features based on group statistics.
    """
    X_cf = X.clone()
    A_cf = 1 - A  # Flip group
    
    for d_idx in descendant_indices:
        mean_0 = X[A == 0, d_idx].mean()
        mean_1 = X[A == 1, d_idx].mean()
        shift = mean_1 - mean_0
        X_cf[:, d_idx] = torch.where(
            A == 0,
            X[:, d_idx] + shift,
            X[:, d_idx] - shift,
        )
    
    # Combine original and counterfactual
    return (
        torch.cat([X, X_cf]),
        torch.cat([y, y]),
        torch.cat([A, A_cf]),
    )
```

### 3. SMOTE-Fair

Extends SMOTE to generate synthetic minority samples while respecting protected group structure:

```python
def smote_fair(
    X: torch.Tensor,
    y: torch.Tensor,
    A: torch.Tensor,
    k_neighbors: int = 5,
    target_subgroup: tuple = (1, 1),
) -> tuple:
    """
    Generate synthetic samples for an underrepresented (A, Y) subgroup
    using SMOTE-like interpolation.
    """
    a_target, y_target = target_subgroup
    mask = (A == a_target) & (y == y_target)
    X_sub = X[mask]
    n_existing = len(X_sub)
    
    if n_existing < 2:
        return X, y, A
    
    # Find target size (match largest subgroup)
    max_size = max(
        ((A == a) & (y == yv)).sum().item()
        for a in torch.unique(A) for yv in torch.unique(y)
    )
    n_synthetic = max_size - n_existing
    
    if n_synthetic <= 0:
        return X, y, A
    
    # Generate synthetic samples via interpolation
    synthetic = []
    for _ in range(n_synthetic):
        i = np.random.randint(0, n_existing)
        j = np.random.randint(0, n_existing)
        while j == i:
            j = np.random.randint(0, n_existing)
        alpha = torch.rand(1).item()
        synthetic.append(X_sub[i] * alpha + X_sub[j] * (1 - alpha))
    
    X_synth = torch.stack(synthetic)
    y_synth = torch.full((n_synthetic,), y_target, dtype=y.dtype)
    A_synth = torch.full((n_synthetic,), a_target, dtype=A.dtype)
    
    return (
        torch.cat([X, X_synth]),
        torch.cat([y, y_synth]),
        torch.cat([A, A_synth]),
    )
```

## Summary

- **Oversampling** balances subgroup representation without modifying existing data
- **Counterfactual augmentation** generates "what-if" samples to reduce dependence on $A$
- **SMOTE-Fair** creates synthetic in-distribution samples for underrepresented subgroups
- All methods are model-agnostic and composable with other pre-processing techniques

## Next Steps

- [Adversarial Debiasing](../inprocessing/adversarial.md): In-processing mitigation via adversarial networks
