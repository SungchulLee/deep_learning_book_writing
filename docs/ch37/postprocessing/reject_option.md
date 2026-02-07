# Reject Option Classification

## Overview

**Reject Option Classification** (Kamiran, Karim, & Zhang, 2012) defers predictions for instances near the decision boundary, where the model is least confident. By giving favorable outcomes to disadvantaged group members in the uncertainty region, it improves fairness without significantly impacting overall accuracy.

## Method

Define a **critical region** around the decision boundary where $|S(x) - 0.5| \leq \theta$ for bandwidth $\theta$:

$$\hat{Y}(x) = \begin{cases} 1 & \text{if } S(x) > 0.5 + \theta \\ 0 & \text{if } S(x) < 0.5 - \theta \\ 1 & \text{if } |S(x) - 0.5| \leq \theta \text{ and } A \in \text{disadvantaged} \\ 0 & \text{if } |S(x) - 0.5| \leq \theta \text{ and } A \in \text{advantaged} \end{cases}$$

## PyTorch Implementation

```python
import torch
from typing import Dict

class RejectOptionClassifier:
    """
    Reject option classification for fairness.
    
    In the uncertainty band around the decision boundary, assigns
    favorable outcomes to the disadvantaged group.
    """
    
    def __init__(self, bandwidth: float = 0.1):
        self.bandwidth = bandwidth
        self.disadvantaged_group = None
    
    def fit(self, y_prob: torch.Tensor, sensitive_attr: torch.Tensor):
        """Identify the disadvantaged group (lower positive rate)."""
        groups = torch.unique(sensitive_attr).tolist()
        rates = {g: y_prob[sensitive_attr == g].mean().item() for g in groups}
        self.disadvantaged_group = min(rates, key=rates.get)
        return self
    
    def predict(self, y_prob: torch.Tensor, sensitive_attr: torch.Tensor):
        """Apply reject option classification."""
        y_pred = (y_prob >= 0.5).long()
        
        in_band = (y_prob >= 0.5 - self.bandwidth) & (y_prob <= 0.5 + self.bandwidth)
        disadvantaged = (sensitive_attr == self.disadvantaged_group) & in_band
        advantaged = (sensitive_attr != self.disadvantaged_group) & in_band
        
        y_pred[disadvantaged] = 1  # Favorable for disadvantaged
        y_pred[advantaged] = 0     # Unfavorable for advantaged
        return y_pred

# Demonstration
def demo():
    torch.manual_seed(42)
    n = 2000
    A = torch.randint(0, 2, (n,))
    y = torch.randint(0, 2, (n,))
    
    bias = torch.where(A == 0, torch.tensor(0.15), torch.tensor(-0.15))
    y_prob = torch.sigmoid(y.float() * 1.5 + bias + torch.randn(n) * 0.3)
    
    baseline = (y_prob >= 0.5).long()
    spd_base = abs(baseline[A==0].float().mean() - baseline[A==1].float().mean())
    
    roc = RejectOptionClassifier(bandwidth=0.15)
    roc.fit(y_prob, A)
    y_pred = roc.predict(y_prob, A)
    spd_roc = abs(y_pred[A==0].float().mean() - y_pred[A==1].float().mean())
    
    print("Reject Option Classification")
    print("=" * 50)
    print(f"Disadvantaged group: {roc.disadvantaged_group}")
    print(f"Baseline SPD: {spd_base:.4f}")
    print(f"Reject Option SPD: {spd_roc:.4f}")
    print(f"Baseline accuracy: {(baseline == y).float().mean():.4f}")
    print(f"Reject Option accuracy: {(y_pred == y).float().mean():.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Modifies predictions only in the **uncertainty band** near the decision boundary
- Assigns **favorable outcomes** to the disadvantaged group in that band
- Simple, model-agnostic, requires only predicted probabilities
- Bandwidth $\theta$ controls the fairness–accuracy tradeoff

## Next Steps

- [Fairness Audits](../evaluation/audits.md): Systematic evaluation of deployed models
