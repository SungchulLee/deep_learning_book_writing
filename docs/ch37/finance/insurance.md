# Fair Insurance Pricing

## Overview

Insurance pricing models face unique fairness challenges. While actuarial science has historically used demographic variables for risk segmentation, evolving regulations and ethical standards increasingly restrict the use of protected attributes in pricing—even when they are statistically predictive.

## Regulatory Landscape

- **Gender**: EU Gender Directive (2012) prohibits gender-based pricing; many US states restrict it
- **Race/ethnicity**: Universally prohibited as an explicit pricing factor
- **Age**: May be used in some insurance lines but faces increasing scrutiny
- **Geography**: ZIP code pricing may constitute proxy discrimination

## Fair Pricing Framework

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class FairInsurancePricer(nn.Module):
    """
    Insurance pricing model with fairness constraints.
    
    Predicts expected loss while ensuring premium differences
    across protected groups stay within acceptable bounds.
    """
    
    def __init__(self, n_features: int, fairness_weight: float = 1.0):
        super().__init__()
        self.fairness_weight = fairness_weight
        self.network = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus(),  # Positive output for premiums
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def compute_loss(self, X, y_loss, A):
        """Loss = MSE(predicted premium, actual loss) + λ · fairness penalty."""
        premium = self.forward(X)
        mse = nn.functional.mse_loss(premium, y_loss)
        
        # Fairness: equalize average premium across groups
        m0, m1 = (A == 0).float(), (A == 1).float()
        avg0 = (premium * m0).sum() / (m0.sum() + 1e-8)
        avg1 = (premium * m1).sum() / (m1.sum() + 1e-8)
        fair_penalty = (avg0 - avg1) ** 2
        
        return mse + self.fairness_weight * fair_penalty


def demo():
    torch.manual_seed(42)
    n = 3000
    A = torch.randint(0, 2, (n,))  # Gender
    
    # Features: age, vehicle_value, driving_history, region
    X = torch.randn(n, 4) * 0.3 + 0.5
    
    # True loss with some group correlation
    true_loss = torch.relu(0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] +
                           0.1 * A.float() + torch.randn(n) * 0.1)
    
    for name, lam in [("Unconstrained", 0.0), ("Fair (λ=5)", 5.0)]:
        model = FairInsurancePricer(4, fairness_weight=lam)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        for _ in range(300):
            opt.zero_grad()
            loss = model.compute_loss(X, true_loss, A)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            premiums = model(X)
        
        gap = abs(premiums[A==0].mean() - premiums[A==1].mean())
        mse = nn.functional.mse_loss(premiums, true_loss)
        print(f"{name}: avg premium gap = {gap:.4f}, MSE = {mse:.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Insurance fairness balances **actuarial accuracy** with **non-discrimination**
- Premium differences across groups must be justified by legitimate risk factors
- Fairness regularization constrains average premium gaps while preserving risk sensitivity
- Regulatory requirements vary significantly by jurisdiction and insurance line

## Next Steps

- [Algorithmic Trading](trading.md): Fairness in market-making and execution
