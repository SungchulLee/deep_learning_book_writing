# Intersectionality in Fairness

## Overview

Intersectionality recognizes that individuals belong to multiple protected groups simultaneously (e.g., Black women, elderly disabled individuals). Bias may affect intersectional subgroups even when each individual attribute appears fair in isolation. Kimberlé Crenshaw coined the term to describe how discrimination against Black women could not be understood by examining race and gender separately.

## Mathematical Framework

For protected attributes $A_1, A_2, \ldots, A_m$, define intersectional groups as the Cartesian product:

$$G = A_1 \times A_2 \times \cdots \times A_m$$

A fairness metric $\mathcal{M}$ exhibits **intersectional bias** if it is satisfied for each marginal attribute but violated for some intersectional subgroup.

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict, List, Tuple
from itertools import product
from dataclasses import dataclass

@dataclass
class IntersectionalMetrics:
    """Intersectional fairness analysis results."""
    subgroup_rates: Dict[tuple, float]
    subgroup_counts: Dict[tuple, int]
    max_subgroup_gap: float
    worst_subgroup: tuple
    marginal_fair: Dict[str, bool]
    intersectional_fair: bool

class IntersectionalFairnessCalculator:
    """
    Analyze fairness across intersectional subgroups.
    
    Detects cases where marginal fairness hides intersectional bias.
    """
    
    def __init__(self, min_subgroup_size: int = 30, threshold: float = 0.1):
        self.min_subgroup_size = min_subgroup_size
        self.threshold = threshold
    
    def compute(
        self,
        y_pred: torch.Tensor,
        attributes: Dict[str, torch.Tensor],
    ) -> IntersectionalMetrics:
        """
        Compute intersectional fairness metrics.
        
        Args:
            y_pred: Predicted labels
            attributes: Dict mapping attribute names to values
            
        Returns:
            IntersectionalMetrics
        """
        n = len(y_pred)
        attr_names = list(attributes.keys())
        attr_vals = list(attributes.values())
        
        # Create intersectional group labels
        subgroup_rates = {}
        subgroup_counts = {}
        
        # Get unique values per attribute
        unique_vals = [torch.unique(a).tolist() for a in attr_vals]
        
        for combo in product(*unique_vals):
            mask = torch.ones(n, dtype=torch.bool)
            for attr_tensor, val in zip(attr_vals, combo):
                mask = mask & (attr_tensor == val)
            
            count = mask.sum().item()
            if count >= self.min_subgroup_size:
                rate = y_pred[mask].float().mean().item()
                subgroup_rates[combo] = rate
                subgroup_counts[combo] = count
        
        # Overall positive rate
        overall_rate = y_pred.float().mean().item()
        
        # Max gap from overall rate
        if subgroup_rates:
            gaps = {k: abs(v - overall_rate) for k, v in subgroup_rates.items()}
            worst = max(gaps, key=gaps.get)
            max_gap = gaps[worst]
        else:
            worst, max_gap = (), 0.0
        
        # Check marginal fairness
        marginal_fair = {}
        for name, attr in attributes.items():
            groups = torch.unique(attr)
            rates = [y_pred[attr == g].float().mean().item() for g in groups]
            marginal_fair[name] = (max(rates) - min(rates)) < self.threshold
        
        return IntersectionalMetrics(
            subgroup_rates=subgroup_rates,
            subgroup_counts=subgroup_counts,
            max_subgroup_gap=max_gap,
            worst_subgroup=worst,
            marginal_fair=marginal_fair,
            intersectional_fair=max_gap < self.threshold,
        )

# Demonstration
def demo():
    torch.manual_seed(42)
    n = 4000
    gender = torch.randint(0, 2, (n,))   # 0=M, 1=F
    race = torch.randint(0, 3, (n,))     # 0, 1, 2
    
    # Predictions are fair marginally but biased intersectionally
    base = torch.rand(n)
    bias = torch.zeros(n)
    bias[(gender == 1) & (race == 2)] -= 0.25  # Penalize subgroup (F, race=2)
    bias[(gender == 0) & (race == 0)] += 0.10  # Boost subgroup (M, race=0)
    y_pred = ((base + bias) > 0.5).long()
    
    calc = IntersectionalFairnessCalculator()
    m = calc.compute(y_pred, {'gender': gender, 'race': race})
    
    print("Intersectional Fairness Analysis")
    print("=" * 55)
    print(f"\nMarginal fairness:")
    for name, fair in m.marginal_fair.items():
        print(f"  {name}: {'FAIR' if fair else 'UNFAIR'}")
    
    print(f"\nSubgroup positive rates:")
    for sg, rate in sorted(m.subgroup_rates.items()):
        count = m.subgroup_counts[sg]
        print(f"  (gender={sg[0]}, race={sg[1]}): {rate:.4f}  (n={count})")
    
    print(f"\nMax subgroup gap: {m.max_subgroup_gap:.4f}")
    print(f"Worst subgroup: {m.worst_subgroup}")
    print(f"Intersectional fair: {m.intersectional_fair}")

if __name__ == "__main__":
    demo()
```

## Summary

- **Intersectional analysis** reveals biases hidden by marginal fairness
- Use the **Cartesian product** of protected attributes to define subgroups
- Require **minimum subgroup sizes** to avoid noisy estimates
- Critical in finance where multiple protected attributes (race, gender, age) jointly influence outcomes

## Next Steps

- [Reweighing](../preprocessing/reweighing.md): Pre-processing mitigation for detected biases
