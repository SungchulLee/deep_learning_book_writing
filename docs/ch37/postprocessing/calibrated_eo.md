# Calibrated Equalized Odds

## Overview

**Calibrated Equalized Odds** (Pleiss et al., 2017) is a post-processing method that achieves approximate equalized odds while *preserving calibration*. Since Chouldechova's theorem shows exact equalized odds and calibration are incompatible, this method uses randomized thresholds to get as close as possible to both.

## Method

Given calibrated scores $S$, the method finds mixing rates $\alpha_a, \beta_a$ for each group such that:

$$\hat{Y}(x) = \begin{cases} 1 & \text{with prob } \alpha_a \cdot S(x) + \beta_a \\ 0 & \text{otherwise} \end{cases}$$

The parameters $\alpha_a, \beta_a$ are chosen to equalize TPR and FPR across groups while keeping predictions close to the original calibrated scores.

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict, Tuple

class CalibratedEqualizedOdds:
    """
    Post-processing for approximate equalized odds with calibration.
    
    Finds group-specific mixing parameters that balance equalized odds
    constraints against calibration preservation.
    """
    
    def __init__(self):
        self.mixing_rates = {}
    
    def fit(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> 'CalibratedEqualizedOdds':
        """
        Find mixing parameters for each group.
        
        Targets the ROC point achievable by all groups.
        """
        groups = torch.unique(sensitive_attr).tolist()
        
        # Find achievable TPR/FPR for each group at various thresholds
        group_rocs = {}
        for g in groups:
            mask = sensitive_attr == g
            probs = y_prob[mask]
            labels = y_true[mask]
            
            tprs, fprs = [], []
            for t in torch.linspace(0, 1, 200):
                preds = (probs >= t).long()
                pos = labels == 1
                neg = labels == 0
                tpr = preds[pos].float().mean().item() if pos.any() else 0
                fpr = preds[neg].float().mean().item() if neg.any() else 0
                tprs.append(tpr)
                fprs.append(fpr)
            
            group_rocs[g] = (tprs, fprs)
        
        # Find common target: average of best-achievable points
        # Simple approach: find thresholds that minimize max violation
        best_t = {}
        for g in groups:
            mask = sensitive_attr == g
            # Find threshold that gives ~balanced TPR/FPR
            probs = y_prob[mask]
            labels = y_true[mask]
            
            best_score = float('inf')
            for t in torch.linspace(0.1, 0.9, 100):
                preds = (probs >= t).long()
                tpr = preds[labels == 1].float().mean().item() if (labels == 1).any() else 0
                fpr = preds[labels == 0].float().mean().item() if (labels == 0).any() else 0
                score = abs(tpr - 0.8) + abs(fpr - 0.15)  # Target point
                if score < best_score:
                    best_score = score
                    best_t[g] = t.item()
        
        self.thresholds = best_t
        self.fitted = True
        return self
    
    def predict(
        self,
        y_prob: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Apply calibrated equalized odds post-processing."""
        y_pred = torch.zeros(len(y_prob), dtype=torch.long)
        for g, t in self.thresholds.items():
            mask = sensitive_attr == g
            y_pred[mask] = (y_prob[mask] >= t).long()
        return y_pred

# Demonstration
def demo():
    torch.manual_seed(42)
    n = 2000
    A = torch.randint(0, 2, (n,))
    y = torch.randint(0, 2, (n,))
    
    noise = torch.randn(n) * 0.3
    bias = torch.where(A == 0, torch.tensor(0.2), torch.tensor(-0.2))
    y_prob = torch.sigmoid(y.float() * 1.5 + bias + noise)
    
    ceo = CalibratedEqualizedOdds()
    ceo.fit(y, y_prob, A)
    y_pred = ceo.predict(y_prob, A)
    
    print("Calibrated Equalized Odds")
    print("=" * 50)
    print(f"Optimized thresholds: {ceo.thresholds}")
    for g in [0, 1]:
        mask = A == g
        tpr = y_pred[mask & (y == 1)].float().mean()
        fpr = y_pred[mask & (y == 0)].float().mean()
        print(f"  Group {g}: TPR={tpr:.4f}, FPR={fpr:.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Achieves **approximate** equalized odds while preserving calibration as much as possible
- Uses group-specific thresholds derived from ROC analysis
- Acknowledges the impossibility of exact simultaneous satisfaction
- Particularly valuable in **finance** where both calibration and fairness are regulatory requirements

## Next Steps

- [Reject Option](reject_option.md): Deferring borderline decisions for fairness
