# Fair Credit Scoring

## Overview

Credit scoring is one of the most regulated and scrutinized applications of ML in finance. Models must comply with ECOA (Equal Credit Opportunity Act), the Fair Housing Act, and fair lending regulations while maintaining predictive accuracy for risk assessment.

## Regulatory Requirements

- **ECOA** prohibits discrimination based on race, color, religion, national origin, sex, marital status, and age
- **Adverse action notices** must explain why an application was denied
- **Disparate impact** is legally actionable even without discriminatory intent
- **Model risk management** (SR 11-7) requires fairness documentation and validation

## Fair Credit Scoring Pipeline

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class CreditFairnessReport:
    """Credit scoring fairness evaluation."""
    approval_rates: Dict[int, float]
    default_rates: Dict[int, float]
    spd: float
    dir_ratio: float
    tpr_diff: float
    passes_ecoa: bool
    model_accuracy: float

class FairCreditScorer(nn.Module):
    """
    Credit scoring model with fairness constraints.
    
    Combines standard credit features with fairness regularization
    to meet regulatory requirements.
    """
    
    def __init__(self, n_features: int, hidden_dim: int = 64, fairness_weight: float = 2.0):
        super().__init__()
        self.fairness_weight = fairness_weight
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def compute_loss(self, X, y, A):
        pred = self.forward(X)
        bce = nn.functional.binary_cross_entropy(pred, y.float())
        
        # Demographic parity penalty
        m0, m1 = (A == 0).float(), (A == 1).float()
        avg0 = (pred * m0).sum() / (m0.sum() + 1e-8)
        avg1 = (pred * m1).sum() / (m1.sum() + 1e-8)
        dp_loss = (avg0 - avg1) ** 2
        
        return bce + self.fairness_weight * dp_loss


def simulate_credit_data(n: int = 5000):
    """Generate synthetic credit scoring data with bias."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    A = torch.randint(0, 2, (n,))  # Protected attribute
    
    # Features: credit score, income, DTI, employment length, loan amount
    credit_score = torch.randn(n) * 0.15 + 0.65
    income = torch.randn(n) * 0.2 + 0.5
    dti = torch.randn(n) * 0.1 + 0.35
    employment = torch.randn(n) * 0.15 + 0.5
    loan_amount = torch.randn(n) * 0.15 + 0.4
    
    # Historical bias: Group 1 had systematically lower approval
    bias = torch.where(A == 0, torch.tensor(0.1), torch.tensor(-0.1))
    
    # Default probability (true creditworthiness)
    logit = 2 * credit_score + 1.5 * income - 2 * dti + employment - loan_amount + bias
    default_prob = 1 - torch.sigmoid(logit)
    y_default = torch.bernoulli(default_prob)
    
    X = torch.stack([credit_score, income, dti, employment, loan_amount], dim=1)
    return X, y_default, A


def evaluate_credit_model(model, X, y, A) -> CreditFairnessReport:
    """Evaluate credit model for fairness compliance."""
    model.eval()
    with torch.no_grad():
        scores = model(X)
        preds = (scores >= 0.5).long()
    
    # Approval = predicted non-default
    approval = 1 - preds
    
    rates = {}
    defaults = {}
    for g in [0, 1]:
        mask = A == g
        rates[g] = approval[mask].float().mean().item()
        defaults[g] = y[mask].float().mean().item()
    
    spd = abs(rates[0] - rates[1])
    dir_r = min(rates.values()) / max(rates.values()) if max(rates.values()) > 0 else 0
    
    tpr0 = approval[(A == 0) & (y == 0)].float().mean().item()
    tpr1 = approval[(A == 1) & (y == 0)].float().mean().item()
    
    return CreditFairnessReport(
        approval_rates=rates,
        default_rates=defaults,
        spd=spd,
        dir_ratio=dir_r,
        tpr_diff=abs(tpr0 - tpr1),
        passes_ecoa=dir_r >= 0.8,
        model_accuracy=(preds == y.long()).float().mean().item(),
    )


# Demonstration
def demo():
    X, y, A = simulate_credit_data(5000)
    
    # Standard model (no fairness)
    standard = FairCreditScorer(5, fairness_weight=0.0)
    opt = torch.optim.Adam(standard.parameters(), lr=0.005)
    for _ in range(200):
        opt.zero_grad()
        loss = standard.compute_loss(X, y, A)
        loss.backward()
        opt.step()
    
    # Fair model
    fair = FairCreditScorer(5, fairness_weight=5.0)
    opt = torch.optim.Adam(fair.parameters(), lr=0.005)
    for _ in range(200):
        opt.zero_grad()
        loss = fair.compute_loss(X, y, A)
        loss.backward()
        opt.step()
    
    print("Fair Credit Scoring")
    print("=" * 55)
    for name, model in [("Standard", standard), ("Fair", fair)]:
        r = evaluate_credit_model(model, X, y, A)
        print(f"\n{name} Model:")
        print(f"  Approval rates: G0={r.approval_rates[0]:.4f}, G1={r.approval_rates[1]:.4f}")
        print(f"  SPD={r.spd:.4f}, DIR={r.dir_ratio:.4f}")
        print(f"  Accuracy={r.model_accuracy:.4f}")
        print(f"  ECOA compliance: {'✓' if r.passes_ecoa else '✗'}")

if __name__ == "__main__":
    demo()
```

## Summary

- Credit scoring requires simultaneous **predictive accuracy** and **regulatory fairness**
- The **four-fifths rule** (DIR ≥ 0.8) is the primary legal test
- Fairness regularization can bring models into compliance with modest accuracy tradeoffs
- All fairness decisions must be **documented** for regulatory examination

## Next Steps

- [Insurance Pricing](insurance.md): Fairness in actuarial models
