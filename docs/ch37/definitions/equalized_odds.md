# Equalized Odds

## Definition

**Equalized Odds** (Hardt, Price, & Srebro, 2016) requires that the prediction $\hat{Y}$ is conditionally independent of the protected attribute $A$ given the true label $Y$. Both the true positive rate (TPR) *and* the false positive rate (FPR) must be equal across groups.

### Mathematical Formulation

A classifier $\hat{Y}$ satisfies equalized odds with respect to protected attribute $A$ if:

$$P(\hat{Y} = 1 \mid Y = y, A = 0) = P(\hat{Y} = 1 \mid Y = y, A = 1) \quad \forall\; y \in \{0, 1\}$$

This is equivalent to:

$$\hat{Y} \perp\!\!\!\perp A \mid Y$$

### Component Constraints

Equalized odds combines two conditions:

1. **Equal TPR** (equal opportunity): $P(\hat{Y}=1 \mid Y=1, A=0) = P(\hat{Y}=1 \mid Y=1, A=1)$
2. **Equal FPR**: $P(\hat{Y}=1 \mid Y=0, A=0) = P(\hat{Y}=1 \mid Y=0, A=1)$

Equal opportunity is therefore a *relaxation* of equalized odds that drops the FPR constraint.

## Motivation

Equalized odds captures a strong notion of fairness: the classifier's error pattern should not depend on group membership. Both types of errors—missing a positive (FN) and falsely flagging a negative (FP)—should be equalized.

**Example.** In a fraud detection system:

- Equal FPR ensures that legitimate transactions from all groups are equally unlikely to be flagged as fraud
- Equal TPR ensures that fraudulent transactions from all groups are equally likely to be detected

## PyTorch Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class EqualizedOddsMetrics:
    """Container for equalized odds metrics."""
    group_tpr: Dict[int, float]
    group_fpr: Dict[int, float]
    tpr_difference: float
    fpr_difference: float
    max_violation: float
    satisfies_equal_opportunity: bool
    satisfies_equalized_odds: bool


class EqualizedOddsCalculator:
    """
    Calculate equalized odds metrics.
    
    Equalized Odds requires both:
        P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (equal TPR)
        P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  (equal FPR)
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def compute(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> EqualizedOddsMetrics:
        """
        Compute equalized odds metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Protected attribute values
            
        Returns:
            EqualizedOddsMetrics
        """
        for arr_name, arr in [('y_true', y_true), ('y_pred', y_pred), ('sensitive_attr', sensitive_attr)]:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr)
        
        y_true = torch.as_tensor(y_true)
        y_pred = torch.as_tensor(y_pred)
        sensitive_attr = torch.as_tensor(sensitive_attr)
        
        groups = torch.unique(sensitive_attr)
        group_tpr, group_fpr = {}, {}
        
        for group in groups:
            gv = group.item()
            
            # TPR: P(Ŷ=1 | Y=1, A=g)
            pos_mask = (sensitive_attr == group) & (y_true == 1)
            if pos_mask.sum() > 0:
                group_tpr[gv] = y_pred[pos_mask].float().mean().item()
            else:
                group_tpr[gv] = 0.0
            
            # FPR: P(Ŷ=1 | Y=0, A=g)
            neg_mask = (sensitive_attr == group) & (y_true == 0)
            if neg_mask.sum() > 0:
                group_fpr[gv] = y_pred[neg_mask].float().mean().item()
            else:
                group_fpr[gv] = 0.0
        
        tpr_vals = list(group_tpr.values())
        fpr_vals = list(group_fpr.values())
        
        tpr_diff = abs(tpr_vals[0] - tpr_vals[1]) if len(tpr_vals) >= 2 else 0.0
        fpr_diff = abs(fpr_vals[0] - fpr_vals[1]) if len(fpr_vals) >= 2 else 0.0
        
        return EqualizedOddsMetrics(
            group_tpr=group_tpr,
            group_fpr=group_fpr,
            tpr_difference=tpr_diff,
            fpr_difference=fpr_diff,
            max_violation=max(tpr_diff, fpr_diff),
            satisfies_equal_opportunity=tpr_diff < self.threshold,
            satisfies_equalized_odds=max(tpr_diff, fpr_diff) < self.threshold,
        )


def equalized_odds_loss(
    y_prob: torch.Tensor,
    y_true: torch.Tensor,
    sensitive_attr: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable equalized odds loss.
    
    L_EO = (E[ŷ|Y=1,A=0] - E[ŷ|Y=1,A=1])² + (E[ŷ|Y=0,A=0] - E[ŷ|Y=0,A=1])²
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        sensitive_attr: Protected attribute (0 or 1)
        
    Returns:
        Scalar equalized odds loss
    """
    loss = torch.tensor(0.0, requires_grad=True)
    
    for y_val in [0, 1]:
        mask_0 = ((sensitive_attr == 0) & (y_true == y_val)).float()
        mask_1 = ((sensitive_attr == 1) & (y_true == y_val)).float()
        
        n_0 = mask_0.sum() + epsilon
        n_1 = mask_1.sum() + epsilon
        
        avg_0 = (y_prob * mask_0).sum() / n_0
        avg_1 = (y_prob * mask_1).sum() / n_1
        
        loss = loss + (avg_0 - avg_1) ** 2
    
    return loss


# --- Demonstration ---

def demonstrate_equalized_odds():
    """Compare equal opportunity vs equalized odds."""
    np.random.seed(42)
    n = 2000
    
    group = torch.tensor(np.random.randint(0, 2, n))
    y_true = torch.tensor(np.random.randint(0, 2, n))
    
    # Model A: Equal TPR but unequal FPR
    y_pred_a = torch.zeros(n, dtype=torch.long)
    for g in [0, 1]:
        pos_mask = (group == g) & (y_true == 1)
        neg_mask = (group == g) & (y_true == 0)
        y_pred_a[pos_mask] = torch.tensor(
            np.random.choice([0, 1], pos_mask.sum().item(), p=[0.2, 0.8])
        )
        # Different FPR by group
        fpr = 0.1 if g == 0 else 0.3
        y_pred_a[neg_mask] = torch.tensor(
            np.random.choice([0, 1], neg_mask.sum().item(), p=[1-fpr, fpr])
        )
    
    calc = EqualizedOddsCalculator()
    m_a = calc.compute(y_true, y_pred_a, group)
    
    print("Model A: Equal TPR, Unequal FPR")
    print(f"  TPR: {m_a.group_tpr}  (diff = {m_a.tpr_difference:.4f})")
    print(f"  FPR: {m_a.group_fpr}  (diff = {m_a.fpr_difference:.4f})")
    print(f"  Equal Opportunity: {'PASS' if m_a.satisfies_equal_opportunity else 'FAIL'}")
    print(f"  Equalized Odds:    {'PASS' if m_a.satisfies_equalized_odds else 'FAIL'}")
    
    # Model B: Equal TPR and equal FPR
    y_pred_b = torch.zeros(n, dtype=torch.long)
    for g in [0, 1]:
        pos_mask = (group == g) & (y_true == 1)
        neg_mask = (group == g) & (y_true == 0)
        y_pred_b[pos_mask] = torch.tensor(
            np.random.choice([0, 1], pos_mask.sum().item(), p=[0.2, 0.8])
        )
        y_pred_b[neg_mask] = torch.tensor(
            np.random.choice([0, 1], neg_mask.sum().item(), p=[0.85, 0.15])
        )
    
    m_b = calc.compute(y_true, y_pred_b, group)
    
    print(f"\nModel B: Equal TPR and Equal FPR")
    print(f"  TPR: {m_b.group_tpr}  (diff = {m_b.tpr_difference:.4f})")
    print(f"  FPR: {m_b.group_fpr}  (diff = {m_b.fpr_difference:.4f})")
    print(f"  Equal Opportunity: {'PASS' if m_b.satisfies_equal_opportunity else 'FAIL'}")
    print(f"  Equalized Odds:    {'PASS' if m_b.satisfies_equalized_odds else 'FAIL'}")

if __name__ == "__main__":
    demonstrate_equalized_odds()
```

## Geometric Interpretation

In ROC space, equalized odds requires that all groups share the same ROC operating point (TPR, FPR). The post-processing method of Hardt et al. (2016) achieves this by finding group-specific thresholds that map each group's ROC curve to a common point (see [Threshold Optimization](../postprocessing/thresholds.md)).

## Relationship to Other Criteria

| Criterion | Constraints | Strength |
|-----------|------------|----------|
| Demographic Parity | Equal $P(\hat{Y}=1 \mid A)$ | Weakest (unconditional) |
| Equal Opportunity | Equal TPR across groups | Medium |
| Equalized Odds | Equal TPR *and* FPR | Strongest (among these three) |

## Summary

- **Equalized Odds** requires equal TPR and equal FPR across groups
- Strictly stronger than equal opportunity (which only requires equal TPR)
- The **differentiable loss** sums squared TPR and FPR differences for gradient-based optimization
- Incompatible with calibration when base rates differ (see [Impossibility Theorems](../impossibility/chouldechova.md))

## Next Steps

- [Calibration](calibration.md): Fairness from the score perspective
- [Chouldechova's Theorem](../impossibility/chouldechova.md): Why equalized odds and calibration cannot coexist
