# Threshold Optimization

## Overview

**Threshold optimization** (Hardt, Price, & Srebro, 2016) is the most widely used post-processing method. Instead of modifying the model, it applies **group-specific classification thresholds** to the model's predicted probabilities, finding thresholds that satisfy a chosen fairness criterion.

## Mathematical Formulation

Given a score function $S(x) \in [0, 1]$, the standard classifier uses a single threshold $t$:

$$\hat{Y} = \mathbf{1}[S(x) \geq t]$$

Threshold optimization uses group-specific thresholds:

$$\hat{Y} = \mathbf{1}[S(x) \geq t_{A}] \quad \text{where } t_A \text{ depends on the individual's group}$$

The thresholds $\{t_a\}$ are chosen to satisfy the desired fairness criterion while maximizing accuracy.

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class ThresholdResult:
    group_thresholds: Dict[int, float]
    group_tpr: Dict[int, float]
    group_fpr: Dict[int, float]
    accuracy: float
    fairness_violation: float

class FairThresholdOptimizer:
    """
    Find group-specific thresholds to satisfy fairness criteria.
    
    Supports: equalized_odds, equal_opportunity, demographic_parity.
    """
    
    def __init__(self, criterion: str = 'equalized_odds', n_grid: int = 100):
        self.criterion = criterion
        self.n_grid = n_grid
    
    def _compute_rates(self, y_true, y_pred):
        """Compute TPR and FPR for given predictions."""
        pos = y_true == 1
        neg = y_true == 0
        tpr = y_pred[pos].float().mean().item() if pos.any() else 0.0
        fpr = y_pred[neg].float().mean().item() if neg.any() else 0.0
        return tpr, fpr
    
    def optimize(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> ThresholdResult:
        """
        Find optimal group-specific thresholds.
        
        Uses grid search over threshold combinations to find the
        pair that minimizes fairness violation while maximizing accuracy.
        """
        groups = torch.unique(sensitive_attr).tolist()
        thresholds = torch.linspace(0.01, 0.99, self.n_grid)
        
        best_result = None
        best_score = float('inf')
        
        # Grid search over threshold pairs
        for t0 in thresholds:
            for t1 in thresholds:
                group_thresh = {groups[0]: t0.item(), groups[1]: t1.item()}
                
                # Apply group-specific thresholds
                y_pred = torch.zeros_like(y_true)
                for g, t in group_thresh.items():
                    mask = sensitive_attr == g
                    y_pred[mask] = (y_prob[mask] >= t).long()
                
                # Compute metrics
                rates = {}
                for g in groups:
                    mask = sensitive_attr == g
                    tpr, fpr = self._compute_rates(y_true[mask], y_pred[mask])
                    rates[g] = {'tpr': tpr, 'fpr': fpr}
                
                # Compute fairness violation based on criterion
                if self.criterion == 'equalized_odds':
                    violation = max(
                        abs(rates[groups[0]]['tpr'] - rates[groups[1]]['tpr']),
                        abs(rates[groups[0]]['fpr'] - rates[groups[1]]['fpr']),
                    )
                elif self.criterion == 'equal_opportunity':
                    violation = abs(rates[groups[0]]['tpr'] - rates[groups[1]]['tpr'])
                elif self.criterion == 'demographic_parity':
                    r0 = y_pred[sensitive_attr == groups[0]].float().mean()
                    r1 = y_pred[sensitive_attr == groups[1]].float().mean()
                    violation = abs(r0 - r1).item()
                
                acc = (y_pred == y_true).float().mean().item()
                
                # Score: fairness violation + small accuracy penalty
                score = violation + 0.1 * (1 - acc)
                
                if score < best_score:
                    best_score = score
                    best_result = ThresholdResult(
                        group_thresholds=group_thresh,
                        group_tpr={g: rates[g]['tpr'] for g in groups},
                        group_fpr={g: rates[g]['fpr'] for g in groups},
                        accuracy=acc,
                        fairness_violation=violation,
                    )
        
        return best_result


# Demonstration
def demo():
    torch.manual_seed(42)
    n = 2000
    A = torch.randint(0, 2, (n,))
    y_true = torch.randint(0, 2, (n,))
    
    # Biased model scores
    noise = torch.randn(n) * 0.3
    bias = torch.where(A == 0, torch.tensor(0.3), torch.tensor(-0.3))
    y_prob = torch.sigmoid(y_true.float() * 2 + bias + noise)
    
    # Single threshold baseline
    y_single = (y_prob >= 0.5).long()
    acc_s = (y_single == y_true).float().mean()
    spd_s = abs(y_single[A==0].float().mean() - y_single[A==1].float().mean())
    
    print("Threshold Optimization")
    print("=" * 55)
    print(f"\nSingle threshold (t=0.5):")
    print(f"  Accuracy: {acc_s:.4f}, SPD: {spd_s:.4f}")
    
    for criterion in ['demographic_parity', 'equal_opportunity', 'equalized_odds']:
        opt = FairThresholdOptimizer(criterion=criterion, n_grid=50)
        result = opt.optimize(y_true, y_prob, A)
        
        print(f"\n{criterion}:")
        print(f"  Thresholds: {result.group_thresholds}")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Violation: {result.fairness_violation:.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- **Group-specific thresholds** are the simplest post-processing fairness method
- Requires only predicted probabilities—no model retraining needed
- Can target any fairness criterion via appropriate violation measurement
- Grid search finds optimal thresholds; convex relaxations available for efficiency

## Next Steps

- [Calibrated Equalized Odds](calibrated_eo.md): Post-processing that preserves calibration
- [Reject Option](reject_option.md): Deferring uncertain decisions for fairness
