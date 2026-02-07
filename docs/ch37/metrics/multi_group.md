# Multi-Group Fairness Metrics

## Overview

Most fairness literature focuses on binary protected attributes, but real-world applications involve multiple groups (e.g., multiple racial/ethnic categories, age brackets). Multi-group metrics extend binary fairness measures to handle $k > 2$ groups.

## Mathematical Extension

For $k$ groups with $A \in \{0, 1, \ldots, k-1\}$, the multi-group analogs are:

**Multi-group SPD** (maximum pairwise difference):

$$\text{SPD}_{\max} = \max_{a, a'} \bigl|P(\hat{Y}=1 \mid A=a) - P(\hat{Y}=1 \mid A=a')\bigr|$$

**Multi-group DIR** (minimum pairwise ratio):

$$\text{DIR}_{\min} = \min_{a, a'} \frac{P(\hat{Y}=1 \mid A=a)}{P(\hat{Y}=1 \mid A=a')}$$

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations
from dataclasses import dataclass

@dataclass
class MultiGroupMetrics:
    """Fairness metrics across multiple groups."""
    group_positive_rates: Dict[int, float]
    max_spd: float
    min_dir: float
    worst_pair: Tuple[int, int]
    pairwise_spd: Dict[Tuple[int, int], float]
    group_tpr: Dict[int, float]
    max_tpr_diff: float

class MultiGroupFairnessCalculator:
    """
    Compute fairness metrics for arbitrary number of groups.
    """
    
    def compute(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> MultiGroupMetrics:
        groups = torch.unique(sensitive_attr).tolist()
        
        # Per-group rates
        pos_rates = {}
        tprs = {}
        for g in groups:
            mask = sensitive_attr == g
            pos_rates[g] = y_pred[mask].float().mean().item()
            pos_mask = mask & (y_true == 1)
            tprs[g] = y_pred[pos_mask].float().mean().item() if pos_mask.any() else 0.0
        
        # Pairwise comparisons
        pairwise_spd = {}
        for a, b in combinations(groups, 2):
            pairwise_spd[(a, b)] = abs(pos_rates[a] - pos_rates[b])
        
        max_spd = max(pairwise_spd.values()) if pairwise_spd else 0.0
        worst_pair = max(pairwise_spd, key=pairwise_spd.get) if pairwise_spd else (0, 0)
        
        rates = list(pos_rates.values())
        min_dir = min(rates) / max(rates) if max(rates) > 0 else 0.0
        
        tpr_vals = list(tprs.values())
        max_tpr_diff = max(tpr_vals) - min(tpr_vals) if len(tpr_vals) >= 2 else 0.0
        
        return MultiGroupMetrics(
            group_positive_rates=pos_rates,
            max_spd=max_spd,
            min_dir=min_dir,
            worst_pair=worst_pair,
            pairwise_spd=pairwise_spd,
            group_tpr=tprs,
            max_tpr_diff=max_tpr_diff,
        )

# Demonstration
def demo():
    torch.manual_seed(42)
    n = 3000
    A = torch.randint(0, 4, (n,))  # 4 groups
    y_true = torch.randint(0, 2, (n,))
    
    # Biased predictions: different rates per group
    bias = torch.tensor([0.0, -0.1, 0.1, -0.2])
    y_pred = (torch.rand(n) + bias[A] > 0.5).long()
    
    calc = MultiGroupFairnessCalculator()
    m = calc.compute(y_true, y_pred, A)
    
    print("Multi-Group Fairness Metrics (4 groups)")
    print("=" * 50)
    for g, r in m.group_positive_rates.items():
        print(f"  Group {g}: positive rate = {r:.4f}")
    print(f"\nMax SPD: {m.max_spd:.4f} (groups {m.worst_pair})")
    print(f"Min DIR: {m.min_dir:.4f}")
    print(f"Max TPR diff: {m.max_tpr_diff:.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Multi-group metrics use **maximum pairwise differences** or **min/max ratios**
- The **worst-case pair** identifies which group comparison drives the fairness violation
- Always report per-group statistics alongside aggregate multi-group metrics

## Next Steps

- [Intersectionality](intersectionality.md): When individuals belong to multiple protected groups simultaneously
