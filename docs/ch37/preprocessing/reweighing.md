# Reweighing

## Overview

**Reweighing** (Kamiran & Calders, 2012) is a pre-processing technique that assigns weights to training samples to remove the correlation between the protected attribute and the label. The idea is to make the training data appear as if it came from a world where the label is independent of the protected attribute.

## Mathematical Formulation

For each combination of group $A = a$ and label $Y = y$, the reweighing weight is:

$$w(a, y) = \frac{P(A = a) \cdot P(Y = y)}{P(A = a, Y = y)}$$

This ensures that in the weighted dataset: $P_w(Y = y \mid A = a) = P(Y = y)$, i.e., the label is independent of the protected attribute.

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict, Tuple

class Reweighing:
    """
    Reweighing pre-processing for bias mitigation.
    
    Computes sample weights that make Y ⊥ A in the weighted distribution.
    """
    
    def compute_weights(
        self,
        y: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reweighing sample weights.
        
        Args:
            y: Labels, shape (n,)
            sensitive_attr: Protected attribute, shape (n,)
            
        Returns:
            Sample weights, shape (n,)
        """
        n = len(y)
        weights = torch.ones(n)
        
        groups = torch.unique(sensitive_attr)
        labels = torch.unique(y)
        
        for a in groups:
            for yv in labels:
                # P(A=a)
                p_a = (sensitive_attr == a).float().mean()
                # P(Y=y)
                p_y = (y == yv).float().mean()
                # P(A=a, Y=y)
                p_ay = ((sensitive_attr == a) & (y == yv)).float().mean()
                
                if p_ay > 0:
                    w = (p_a * p_y) / p_ay
                    mask = (sensitive_attr == a) & (y == yv)
                    weights[mask] = w.item()
        
        # Normalize to preserve effective sample size
        weights = weights * n / weights.sum()
        return weights

# Demonstration
def demo():
    torch.manual_seed(42)
    n = 2000
    A = torch.randint(0, 2, (n,))
    # Biased labels: P(Y=1|A=0) = 0.7, P(Y=1|A=1) = 0.3
    y = torch.where(
        A == 0,
        torch.tensor(np.random.choice([0,1], n, p=[0.3, 0.7])),
        torch.tensor(np.random.choice([0,1], n, p=[0.7, 0.3])),
    )
    
    rw = Reweighing()
    weights = rw.compute_weights(y, A)
    
    print("Reweighing Demonstration")
    print("=" * 50)
    print(f"\nBefore reweighing:")
    for g in [0, 1]:
        print(f"  P(Y=1|A={g}) = {y[A == g].float().mean():.4f}")
    
    print(f"\nAfter reweighing (weighted rates):")
    for g in [0, 1]:
        mask = A == g
        weighted_rate = (y[mask].float() * weights[mask]).sum() / weights[mask].sum()
        print(f"  P_w(Y=1|A={g}) = {weighted_rate:.4f}")
    
    print(f"\nWeight statistics:")
    for g in [0, 1]:
        for yv in [0, 1]:
            mask = (A == g) & (y == yv)
            print(f"  w(A={g}, Y={yv}) = {weights[mask][0]:.4f}")

if __name__ == "__main__":
    demo()
```

## Usage with PyTorch Training

```python
# Use weights in loss computation
criterion = torch.nn.BCELoss(reduction='none')
per_sample_loss = criterion(predictions, labels)
weighted_loss = (per_sample_loss * sample_weights).mean()
weighted_loss.backward()
```

## Summary

- Reweighing assigns sample weights to achieve $Y \perp\!\!\!\perp A$ in the weighted distribution
- Simple, model-agnostic, and can be combined with any training procedure
- Does not modify the data itself—only the importance of each sample

## Next Steps

- [Disparate Impact Remover](disparate_impact.md): Transform features to remove bias
- [Fair Representation](representation.md): Learn fair feature representations
