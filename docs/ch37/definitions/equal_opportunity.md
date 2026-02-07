# Equal Opportunity

## Definition

**Equal Opportunity** (Hardt, Price, & Srebro, 2016) requires that qualified individuals from all groups have an equal chance of receiving a positive prediction. Formally, the true positive rate (TPR) must be equal across protected groups.

### Mathematical Formulation

A classifier $\hat{Y}$ satisfies equal opportunity with respect to protected attribute $A$ if:

$$P(\hat{Y} = 1 \mid Y = 1, A = 0) = P(\hat{Y} = 1 \mid Y = 1, A = 1)$$

Equivalently, the false negative rate (FNR) must be equal:

$$P(\hat{Y} = 0 \mid Y = 1, A = 0) = P(\hat{Y} = 0 \mid Y = 1, A = 1)$$

### Conditional Independence

Equal opportunity requires:

$$\hat{Y} \perp\!\!\!\perp A \mid Y = 1$$

The prediction is independent of group membership *among truly positive individuals*.

## Motivation

Equal opportunity captures the intuition that a fair system should not deny opportunities to qualified individuals based on group membership. Unlike demographic parity, it conditions on the true label, so it does not require equal prediction rates when base rates differ.

**Example.** In a loan approval system, equal opportunity requires that creditworthy applicants from all groups have an equal probability of being approved. It does not require equal approval rates if the proportion of creditworthy applicants differs across groups.

## PyTorch Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class EqualOpportunityMetrics:
    """Container for equal opportunity metrics."""
    group_tpr: Dict[int, float]
    group_fnr: Dict[int, float]
    tpr_difference: float
    is_fair: bool

class EqualOpportunityCalculator:
    """
    Calculate equal opportunity metrics.
    
    Equal Opportunity requires:
        P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Maximum acceptable TPR difference
        """
        self.threshold = threshold
    
    def compute(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> EqualOpportunityMetrics:
        """
        Compute equal opportunity metrics.
        
        Args:
            y_true: True labels, shape (n_samples,)
            y_pred: Predicted labels, shape (n_samples,)
            sensitive_attr: Protected attribute values
            
        Returns:
            EqualOpportunityMetrics
        """
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        if not isinstance(sensitive_attr, torch.Tensor):
            sensitive_attr = torch.tensor(sensitive_attr)
        
        groups = torch.unique(sensitive_attr)
        group_tpr = {}
        group_fnr = {}
        
        for group in groups:
            gv = group.item()
            # Select truly positive individuals in this group
            mask = (sensitive_attr == group) & (y_true == 1)
            if mask.sum() == 0:
                group_tpr[gv] = 0.0
                group_fnr[gv] = 1.0
                continue
            
            tpr = y_pred[mask].float().mean().item()
            group_tpr[gv] = tpr
            group_fnr[gv] = 1.0 - tpr
        
        tpr_vals = list(group_tpr.values())
        tpr_diff = abs(tpr_vals[0] - tpr_vals[1]) if len(tpr_vals) >= 2 else 0.0
        
        return EqualOpportunityMetrics(
            group_tpr=group_tpr,
            group_fnr=group_fnr,
            tpr_difference=tpr_diff,
            is_fair=tpr_diff < self.threshold,
        )


def equal_opportunity_loss(
    y_prob: torch.Tensor,
    y_true: torch.Tensor,
    sensitive_attr: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable equal opportunity loss.
    
    Penalizes differences in average predicted probability among
    truly positive individuals across groups.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels (0 or 1)
        sensitive_attr: Protected attribute (0 or 1)
        epsilon: Numerical stability constant
        
    Returns:
        Scalar loss for equal opportunity violation
    """
    # Masks: truly positive in each group
    pos_group_0 = ((sensitive_attr == 0) & (y_true == 1)).float()
    pos_group_1 = ((sensitive_attr == 1) & (y_true == 1)).float()
    
    n_pos_0 = pos_group_0.sum() + epsilon
    n_pos_1 = pos_group_1.sum() + epsilon
    
    # Average predicted probability for true positives in each group
    avg_prob_pos_0 = (y_prob * pos_group_0).sum() / n_pos_0
    avg_prob_pos_1 = (y_prob * pos_group_1).sum() / n_pos_1
    
    return (avg_prob_pos_0 - avg_prob_pos_1) ** 2


# --- Demonstration ---

def demonstrate_equal_opportunity():
    """Show the difference between DP and EO."""
    np.random.seed(42)
    n = 1000
    
    group = torch.tensor(np.random.randint(0, 2, n))
    
    # Different base rates: P(Y=1|A=0) = 0.6, P(Y=1|A=1) = 0.4
    y_true = torch.where(
        group == 0,
        torch.tensor(np.random.choice([0, 1], n, p=[0.4, 0.6])),
        torch.tensor(np.random.choice([0, 1], n, p=[0.6, 0.4])),
    )
    
    # Model with equal TPR ≈ 0.8 for both groups (satisfies EO)
    y_pred = torch.zeros(n, dtype=torch.long)
    for g in [0, 1]:
        pos_mask = (group == g) & (y_true == 1)
        neg_mask = (group == g) & (y_true == 0)
        # ~80% TPR for positives
        y_pred[pos_mask] = torch.tensor(
            np.random.choice([0, 1], pos_mask.sum().item(), p=[0.2, 0.8])
        )
        # ~20% FPR for negatives
        y_pred[neg_mask] = torch.tensor(
            np.random.choice([0, 1], neg_mask.sum().item(), p=[0.8, 0.2])
        )
    
    calc = EqualOpportunityCalculator()
    metrics = calc.compute(y_true, y_pred, group)
    
    print("Equal Opportunity Demonstration")
    print("=" * 50)
    print(f"Base rates: P(Y=1|A=0) ≈ 0.6, P(Y=1|A=1) ≈ 0.4")
    print(f"\nTPR by group:")
    for g, tpr in metrics.group_tpr.items():
        print(f"  Group {g}: TPR = {tpr:.4f}")
    print(f"\nTPR difference: {metrics.tpr_difference:.4f}")
    print(f"Equal Opportunity: {'PASS' if metrics.is_fair else 'FAIL'}")
    
    # Check demographic parity (will likely fail due to different base rates)
    overall_rate_0 = y_pred[group == 0].float().mean()
    overall_rate_1 = y_pred[group == 1].float().mean()
    print(f"\nPositive prediction rates:")
    print(f"  Group 0: {overall_rate_0:.4f}")
    print(f"  Group 1: {overall_rate_1:.4f}")
    print(f"  SPD: {abs(overall_rate_0 - overall_rate_1):.4f}")
    print(f"  → Demographic Parity is violated (different base rates)")

if __name__ == "__main__":
    demonstrate_equal_opportunity()
```

## Equal Opportunity vs. Demographic Parity

The key distinction: equal opportunity conditions on the true label while demographic parity does not.

When base rates differ ($P(Y=1 \mid A=0) \neq P(Y=1 \mid A=1)$), a model satisfying equal opportunity will generally *not* satisfy demographic parity, and vice versa. Equal opportunity permits unequal prediction rates as long as truly qualified individuals are treated equally.

## Finance Application

In credit scoring, equal opportunity means that truly creditworthy applicants from all demographic groups have an equal chance of being approved. This aligns well with the regulatory goal of preventing discrimination against qualified borrowers while acknowledging that creditworthiness may legitimately vary across populations.

## Summary

- **Equal Opportunity** equalizes TPR (or equivalently FNR) across groups
- Conditions on $Y = 1$: only truly positive individuals matter
- More permissive than demographic parity when base rates differ
- Aligns with the intuition that "qualified individuals deserve equal treatment"
- Does not constrain false positive rates (see [Equalized Odds](equalized_odds.md))

## Next Steps

- [Equalized Odds](equalized_odds.md): Extends equal opportunity by also constraining FPR
- [Calibration](calibration.md): Fairness from the perspective of predicted scores
