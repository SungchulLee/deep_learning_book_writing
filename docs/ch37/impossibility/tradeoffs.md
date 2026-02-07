# Tradeoff Analysis

## Overview

Given the impossibility theorems of Chouldechova and KMR, practitioners must navigate tradeoffs between competing fairness criteria and between fairness and accuracy. This section provides tools for quantifying these tradeoffs and finding Pareto-optimal solutions.

## The Fairness–Accuracy Pareto Frontier

For a given dataset and model class, the set of achievable (accuracy, fairness violation) pairs forms a Pareto frontier. Points on this frontier represent models where no improvement in fairness is possible without sacrificing accuracy, and vice versa.

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class TradeoffPoint:
    """A single point on the fairness–accuracy frontier."""
    fairness_weight: float
    accuracy: float
    spd: float  # Statistical Parity Difference
    eo_gap: float  # Equal Opportunity gap (TPR difference)
    calibration_gap: float

class FairnessTradeoffAnalyzer:
    """
    Sweep fairness regularization weight to map the Pareto frontier.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def train_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        A: torch.Tensor,
        fairness_weight: float,
        epochs: int = 200,
    ) -> nn.Module:
        """Train a model with given fairness regularization weight."""
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X).squeeze(-1)
            
            bce = nn.functional.binary_cross_entropy(pred, y)
            
            # DP fairness penalty
            mask_0 = (A == 0).float()
            mask_1 = (A == 1).float()
            avg_0 = (pred * mask_0).sum() / (mask_0.sum() + 1e-8)
            avg_1 = (pred * mask_1).sum() / (mask_1.sum() + 1e-8)
            dp_loss = (avg_0 - avg_1) ** 2
            
            loss = bce + fairness_weight * dp_loss
            loss.backward()
            optimizer.step()
        
        return model
    
    def evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        A: torch.Tensor,
        fairness_weight: float,
    ) -> TradeoffPoint:
        """Evaluate a model on accuracy and fairness metrics."""
        model.eval()
        with torch.no_grad():
            prob = model(X).squeeze(-1)
            pred = (prob >= 0.5).long()
        
        accuracy = (pred == y.long()).float().mean().item()
        
        # SPD
        rate_0 = pred[A == 0].float().mean().item()
        rate_1 = pred[A == 1].float().mean().item()
        spd = abs(rate_0 - rate_1)
        
        # EO gap (TPR difference)
        tpr_0 = pred[(A == 0) & (y == 1)].float().mean().item() if ((A == 0) & (y == 1)).any() else 0
        tpr_1 = pred[(A == 1) & (y == 1)].float().mean().item() if ((A == 1) & (y == 1)).any() else 0
        eo_gap = abs(tpr_0 - tpr_1)
        
        # Calibration gap (simplified: PPV difference)
        ppv_0 = y[(A == 0) & (pred == 1)].float().mean().item() if ((A == 0) & (pred == 1)).any() else 0
        ppv_1 = y[(A == 1) & (pred == 1)].float().mean().item() if ((A == 1) & (pred == 1)).any() else 0
        cal_gap = abs(ppv_0 - ppv_1)
        
        return TradeoffPoint(fairness_weight, accuracy, spd, eo_gap, cal_gap)
    
    def sweep(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        A: torch.Tensor,
        weights: List[float],
    ) -> List[TradeoffPoint]:
        """Sweep fairness weights to map the Pareto frontier."""
        results = []
        for w in weights:
            model = self.train_model(X, y, A, w)
            point = self.evaluate(model, X, y, A, w)
            results.append(point)
        return results


def demonstrate_tradeoffs():
    """Map the fairness–accuracy Pareto frontier."""
    torch.manual_seed(42)
    n = 1000
    
    X = torch.randn(n, 5)
    A = torch.randint(0, 2, (n,))
    # Biased labels
    logit = X[:, 0] + X[:, 1] + 0.5 * A.float()
    y = torch.bernoulli(torch.sigmoid(logit))
    
    analyzer = FairnessTradeoffAnalyzer(input_dim=5)
    weights = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = analyzer.sweep(X, y, A, weights)
    
    print("Fairness–Accuracy Pareto Frontier")
    print("=" * 70)
    print(f"{'λ':<8} {'Accuracy':<12} {'SPD':<10} {'EO Gap':<10} {'Cal Gap':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.fairness_weight:<8.1f} {r.accuracy:<12.4f} "
              f"{r.spd:<10.4f} {r.eo_gap:<10.4f} {r.calibration_gap:<10.4f}")
    
    print("\nObservation: As λ increases, SPD decreases (better DP fairness)")
    print("but accuracy decreases and calibration gap may increase.")

if __name__ == "__main__":
    demonstrate_tradeoffs()
```

## Inter-Criteria Tradeoffs

The impossibility theorems reveal that improving one fairness criterion often worsens another:

| Optimizing For | Effect on DP | Effect on EO | Effect on Calibration |
|---------------|-------------|-------------|----------------------|
| Demographic Parity | ✓ Improved | ↕ May worsen | ↕ May worsen |
| Equal Opportunity | ↕ May worsen | ✓ Improved | ↕ May worsen |
| Calibration | ↕ May worsen | ↕ May worsen | ✓ Improved |

## Guidance for Practitioners

1. **Document the choice**: Explicitly state which fairness criterion is prioritized and why
2. **Report multiple metrics**: Even when optimizing for one criterion, report others for transparency
3. **Consider the domain**: Credit scoring may prioritize calibration; hiring may prioritize equal opportunity
4. **Engage stakeholders**: The choice between fairness criteria is fundamentally a value judgment

## Summary

- The **Pareto frontier** visualizes the achievable tradeoff between accuracy and fairness
- **Inter-criteria tradeoffs** mean improving one fairness metric may worsen others
- **Regularization weight** $\lambda$ controls the accuracy–fairness tradeoff
- Practitioners must make **explicit, documented choices** about which criteria to prioritize

## Next Steps

- [Statistical Metrics](../metrics/statistical.md): Comprehensive metrics for evaluating fairness
- [Pre-processing Mitigation](../preprocessing/reweighing.md): The first class of mitigation techniques
