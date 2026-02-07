# Fairness Regularization

## Overview

**Fairness regularization** adds a fairness penalty term to the training loss, encouraging the model to produce fair predictions without requiring explicit constraint satisfaction. The total loss becomes:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{R}_{\text{fairness}}$$

where $\lambda$ controls the fairness–accuracy tradeoff.

## Common Regularizers

### Demographic Parity Regularizer

$$\mathcal{R}_{\text{DP}} = \bigl(\mathbb{E}[\hat{Y} \mid A=0] - \mathbb{E}[\hat{Y} \mid A=1]\bigr)^2$$

### Equal Opportunity Regularizer

$$\mathcal{R}_{\text{EO}} = \bigl(\mathbb{E}[\hat{Y} \mid Y=1, A=0] - \mathbb{E}[\hat{Y} \mid Y=1, A=1]\bigr)^2$$

### Equalized Odds Regularizer

$$\mathcal{R}_{\text{EOdds}} = \sum_{y \in \{0,1\}} \bigl(\mathbb{E}[\hat{Y} \mid Y=y, A=0] - \mathbb{E}[\hat{Y} \mid Y=y, A=1]\bigr)^2$$

### Mutual Information Regularizer

$$\mathcal{R}_{\text{MI}} \approx \hat{I}(\hat{Y}; A)$$

estimated via neural mutual information estimators (e.g., MINE).

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Dict

class FairnessRegularizer:
    """
    Collection of differentiable fairness regularizers.
    """
    
    @staticmethod
    def demographic_parity(y_prob, A, eps=1e-8):
        m0 = (A == 0).float()
        m1 = (A == 1).float()
        avg0 = (y_prob * m0).sum() / (m0.sum() + eps)
        avg1 = (y_prob * m1).sum() / (m1.sum() + eps)
        return (avg0 - avg1) ** 2
    
    @staticmethod
    def equal_opportunity(y_prob, y_true, A, eps=1e-8):
        m0 = ((A == 0) & (y_true == 1)).float()
        m1 = ((A == 1) & (y_true == 1)).float()
        avg0 = (y_prob * m0).sum() / (m0.sum() + eps)
        avg1 = (y_prob * m1).sum() / (m1.sum() + eps)
        return (avg0 - avg1) ** 2
    
    @staticmethod
    def equalized_odds(y_prob, y_true, A, eps=1e-8):
        loss = torch.tensor(0.0, requires_grad=True)
        for yv in [0, 1]:
            m0 = ((A == 0) & (y_true == yv)).float()
            m1 = ((A == 1) & (y_true == yv)).float()
            avg0 = (y_prob * m0).sum() / (m0.sum() + eps)
            avg1 = (y_prob * m1).sum() / (m1.sum() + eps)
            loss = loss + (avg0 - avg1) ** 2
        return loss


class RegularizedFairClassifier(nn.Module):
    """Classifier with configurable fairness regularization."""
    
    def __init__(self, input_dim, hidden_dim=64, reg_type='dp', lam=1.0):
        super().__init__()
        self.lam = lam
        self.reg_type = reg_type
        self.reg = FairnessRegularizer()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def compute_loss(self, X, y, A) -> Dict[str, float]:
        pred = self.forward(X)
        task = nn.functional.binary_cross_entropy(pred, y.float())
        
        if self.reg_type == 'dp':
            fair = self.reg.demographic_parity(pred, A)
        elif self.reg_type == 'eo':
            fair = self.reg.equal_opportunity(pred, y, A)
        elif self.reg_type == 'eodds':
            fair = self.reg.equalized_odds(pred, y, A)
        else:
            fair = torch.tensor(0.0)
        
        total = task + self.lam * fair
        return total, {'task': task.item(), 'fair': fair.item(), 'total': total.item()}

# Demonstration: compare regularizer types
def demo():
    torch.manual_seed(42)
    n, d = 1000, 10
    X = torch.randn(n, d)
    A = torch.randint(0, 2, (n,))
    y = torch.bernoulli(torch.sigmoid(X[:, 0] + 0.5 * A.float()))
    
    print("Comparing Fairness Regularizers (λ=5.0, 200 epochs)")
    print("=" * 60)
    
    for reg_type in ['dp', 'eo', 'eodds']:
        model = RegularizedFairClassifier(d, reg_type=reg_type, lam=5.0)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        
        for _ in range(200):
            opt.zero_grad()
            loss, _ = model.compute_loss(X, y, A)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            pred = (model(X) > 0.5).long()
        
        acc = (pred == y.long()).float().mean()
        spd = abs(pred[A==0].float().mean() - pred[A==1].float().mean())
        print(f"  {reg_type:>6s}: accuracy={acc:.4f}, SPD={spd:.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Regularization is the **simplest** in-processing approach: just add a penalty term
- Multiple regularizer types target different fairness criteria
- $\lambda$ controls the tradeoff—larger values sacrifice more accuracy for fairness
- All regularizers are **differentiable** and work with standard gradient-based optimization

## Next Steps

- [Multi-Objective](multi_objective.md): Optimizing for multiple fairness criteria simultaneously
