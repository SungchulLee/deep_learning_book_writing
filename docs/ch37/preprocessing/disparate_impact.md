# Disparate Impact Remover

## Overview

The **Disparate Impact Remover** (Feldman et al., 2015) is a pre-processing technique that transforms feature distributions so they are identical across protected groups. The key insight is to modify features to remove their correlation with the protected attribute while preserving within-group ranking.

## Mathematical Formulation

For each feature $X_j$, the transformation maps each group's distribution to a common target (typically the median distribution):

$$\tilde{X}_j^{(a)} = F_{\text{target}}^{-1}\bigl(F_{X_j|A=a}(X_j)\bigr)$$

where $F_{X_j|A=a}$ is the CDF of feature $j$ within group $a$, and $F_{\text{target}}$ is the target CDF.

A **repair level** $\lambda \in [0, 1]$ controls the interpolation between original and fully repaired features:

$$X_j^{\text{repaired}} = (1 - \lambda) \cdot X_j + \lambda \cdot \tilde{X}_j$$

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import List, Optional

class DisparateImpactRemover:
    """
    Remove disparate impact by aligning feature distributions across groups.
    
    For each feature, maps group-specific distributions to a common target
    via quantile matching.
    """
    
    def __init__(self, repair_level: float = 1.0):
        """
        Args:
            repair_level: λ ∈ [0, 1]. 0 = no repair, 1 = full repair.
        """
        self.repair_level = repair_level
        self.fitted = False
        self.group_quantiles = {}
        self.target_quantiles = {}
    
    def fit(
        self,
        X: torch.Tensor,
        sensitive_attr: torch.Tensor,
        n_quantiles: int = 100,
    ) -> 'DisparateImpactRemover':
        """Fit quantile mappings from training data."""
        self.n_features = X.shape[1]
        groups = torch.unique(sensitive_attr).tolist()
        quantile_points = torch.linspace(0, 1, n_quantiles)
        
        for j in range(self.n_features):
            self.group_quantiles[j] = {}
            group_qs = []
            
            for g in groups:
                vals = X[sensitive_attr == g, j].sort().values
                q = torch.quantile(vals.float(), quantile_points)
                self.group_quantiles[j][g] = q
                group_qs.append(q)
            
            # Target: median across group quantiles
            self.target_quantiles[j] = torch.stack(group_qs).median(dim=0).values
        
        self.groups = groups
        self.quantile_points = quantile_points
        self.fitted = True
        return self
    
    def transform(
        self,
        X: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Transform features to remove disparate impact."""
        assert self.fitted, "Must call fit() first"
        X_repaired = X.clone()
        
        for j in range(self.n_features):
            for g in self.groups:
                mask = sensitive_attr == g
                if not mask.any():
                    continue
                
                vals = X[mask, j]
                src_q = self.group_quantiles[j][g]
                tgt_q = self.target_quantiles[j]
                
                # Map through quantiles
                repaired = torch.zeros_like(vals)
                for i, v in enumerate(vals):
                    # Find quantile position in source distribution
                    idx = torch.searchsorted(src_q, v.float()).clamp(0, len(src_q)-1)
                    repaired[i] = tgt_q[idx]
                
                # Interpolate with repair level
                X_repaired[mask, j] = (
                    (1 - self.repair_level) * vals + self.repair_level * repaired
                )
        
        return X_repaired

# Demonstration
def demo():
    torch.manual_seed(42)
    n = 1000
    A = torch.randint(0, 2, (n,))
    
    # Feature with different distributions per group
    X = torch.zeros(n, 2)
    X[A == 0, 0] = torch.randn((A == 0).sum()) * 1.0 + 2.0
    X[A == 1, 0] = torch.randn((A == 1).sum()) * 1.5 + 0.5
    X[:, 1] = torch.randn(n)  # Unbiased feature
    
    dir_remover = DisparateImpactRemover(repair_level=1.0)
    dir_remover.fit(X, A)
    X_repaired = dir_remover.transform(X, A)
    
    print("Disparate Impact Remover")
    print("=" * 50)
    for feat in range(2):
        print(f"\nFeature {feat}:")
        for g in [0, 1]:
            orig = X[A == g, feat].mean().item()
            fixed = X_repaired[A == g, feat].mean().item()
            print(f"  Group {g}: mean {orig:.3f} → {fixed:.3f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Aligns feature distributions across groups via **quantile matching**
- **Repair level** $\lambda$ interpolates between original and fully repaired features
- Preserves **within-group ranking** of individuals
- Model-agnostic: any downstream model can be trained on repaired data

## Next Steps

- [Fair Representation](representation.md): Learn latent representations that are fair by construction
