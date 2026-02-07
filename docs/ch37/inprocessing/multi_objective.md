# Multi-Objective Fairness Optimization

## Overview

When multiple fairness criteria must be considered simultaneously—or when accuracy and fairness must be balanced—**multi-objective optimization** finds the Pareto frontier of non-dominated solutions rather than collapsing everything into a single scalar loss.

## Pareto Optimality

A solution $\theta^*$ is **Pareto optimal** if no other $\theta$ improves one objective without worsening another. The set of all Pareto-optimal solutions forms the **Pareto frontier**.

For fairness, typical objectives include:

$$\min_\theta \bigl(\mathcal{L}_{\text{task}}(\theta),\; \mathcal{R}_{\text{DP}}(\theta),\; \mathcal{R}_{\text{EO}}(\theta)\bigr)$$

## Multi-Gradient Descent

Multiple Gradient Descent Algorithm (MGDA) finds a common descent direction for all objectives:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

class MultiObjectiveFairTrainer:
    """
    Train a model with multiple fairness objectives using
    weighted Chebyshev scalarization to explore the Pareto frontier.
    """
    
    def __init__(self, model: nn.Module, objectives: List[str]):
        self.model = model
        self.objectives = objectives
    
    def compute_objectives(
        self, X, y, A, y_prob, eps=1e-8,
    ) -> Dict[str, torch.Tensor]:
        """Compute all objective values."""
        objs = {}
        objs['task'] = nn.functional.binary_cross_entropy(y_prob, y.float())
        
        m0, m1 = (A == 0).float(), (A == 1).float()
        avg0 = (y_prob * m0).sum() / (m0.sum() + eps)
        avg1 = (y_prob * m1).sum() / (m1.sum() + eps)
        objs['dp'] = (avg0 - avg1) ** 2
        
        pm0 = ((A == 0) & (y == 1)).float()
        pm1 = ((A == 1) & (y == 1)).float()
        tavg0 = (y_prob * pm0).sum() / (pm0.sum() + eps)
        tavg1 = (y_prob * pm1).sum() / (pm1.sum() + eps)
        objs['eo'] = (tavg0 - tavg1) ** 2
        
        return objs
    
    def train_with_weights(
        self, X, y, A, weights: Dict[str, float],
        epochs: int = 200, lr: float = 0.005,
    ) -> Dict[str, float]:
        """Train with a specific weight vector on the Pareto frontier."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for _ in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            y_prob = self.model(X).squeeze(-1)
            objs = self.compute_objectives(X, y, A, y_prob)
            
            loss = sum(w * objs[k] for k, w in weights.items() if k in objs)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            y_prob = self.model(X).squeeze(-1)
            final_objs = self.compute_objectives(X, y, A, y_prob)
        
        return {k: v.item() for k, v in final_objs.items()}


def explore_pareto_frontier():
    """Explore the Pareto frontier with different weight combinations."""
    torch.manual_seed(42)
    n, d = 1000, 10
    X = torch.randn(n, d)
    A = torch.randint(0, 2, (n,))
    y = torch.bernoulli(torch.sigmoid(X[:, 0] + 0.5 * A.float()))
    
    weight_configs = [
        {'task': 1.0, 'dp': 0.0, 'eo': 0.0},
        {'task': 1.0, 'dp': 2.0, 'eo': 0.0},
        {'task': 1.0, 'dp': 0.0, 'eo': 2.0},
        {'task': 1.0, 'dp': 5.0, 'eo': 5.0},
        {'task': 1.0, 'dp': 10.0, 'eo': 10.0},
    ]
    
    print("Pareto Frontier Exploration")
    print("=" * 65)
    print(f"{'Weights':<30} {'Task Loss':<12} {'DP':<12} {'EO':<12}")
    print("-" * 65)
    
    for wc in weight_configs:
        model = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        trainer = MultiObjectiveFairTrainer(model, ['task', 'dp', 'eo'])
        results = trainer.train_with_weights(X, y, A, wc)
        
        label = f"task={wc['task']}, dp={wc['dp']}, eo={wc['eo']}"
        print(f"  {label:<28} {results['task']:<12.4f} "
              f"{results['dp']:<12.6f} {results['eo']:<12.6f}")

if __name__ == "__main__":
    explore_pareto_frontier()
```

## Summary

- Multi-objective optimization finds the **Pareto frontier** rather than a single solution
- **Weight scalarization** explores different points on the frontier
- Enables informed tradeoffs between accuracy and multiple fairness criteria
- Critical when stakeholders have different priorities

## Next Steps

- [Threshold Optimization](../postprocessing/thresholds.md): Post-processing approach to fairness
