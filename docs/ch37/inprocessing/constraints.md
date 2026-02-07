# Fairness Constraints

## Overview

**Constrained optimization** for fairness formulates bias mitigation as a constrained optimization problem: minimize prediction loss subject to explicit fairness constraints. Unlike regularization (which adds a penalty term), constraint-based methods enforce hard bounds on fairness violations.

## Mathematical Formulation

$$\min_\theta \; \mathcal{L}_{\text{task}}(\theta) \quad \text{subject to} \quad \mathcal{C}_{\text{fairness}}(\theta) \leq \epsilon$$

For example, with a demographic parity constraint:

$$\min_\theta \; \mathbb{E}[\ell(f_\theta(X), Y)] \quad \text{s.t.} \quad \bigl|P(\hat{Y}=1 \mid A=0) - P(\hat{Y}=1 \mid A=1)\bigr| \leq \epsilon$$

This is typically solved via **Lagrangian relaxation**:

$$\mathcal{L}(\theta, \mu) = \mathcal{L}_{\text{task}}(\theta) + \mu \cdot \bigl(\mathcal{C}_{\text{fairness}}(\theta) - \epsilon\bigr)$$

where $\mu \geq 0$ is the Lagrange multiplier, updated via dual ascent.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple

class ConstrainedFairClassifier(nn.Module):
    """
    Classifier with Lagrangian fairness constraints.
    
    Uses primal-dual optimization: the model parameters are updated
    to minimize loss, while the Lagrange multiplier is updated to
    enforce the fairness constraint.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        epsilon: float = 0.05,
        mu_lr: float = 0.01,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.mu = torch.tensor(0.0, requires_grad=False)
        self.mu_lr = mu_lr
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def compute_constraint(self, y_prob, A):
        """Compute demographic parity violation."""
        mask_0 = (A == 0).float()
        mask_1 = (A == 1).float()
        avg_0 = (y_prob * mask_0).sum() / (mask_0.sum() + 1e-8)
        avg_1 = (y_prob * mask_1).sum() / (mask_1.sum() + 1e-8)
        return (avg_0 - avg_1) ** 2
    
    def train_step(
        self, X, y, A, optimizer,
    ) -> Dict[str, float]:
        """Single primal-dual training step."""
        self.train()
        optimizer.zero_grad()
        
        y_prob = self.forward(X)
        task_loss = nn.functional.binary_cross_entropy(y_prob, y.float())
        constraint_violation = self.compute_constraint(y_prob, A)
        
        # Lagrangian
        lagrangian = task_loss + self.mu * (constraint_violation - self.epsilon)
        lagrangian.backward()
        optimizer.step()
        
        # Dual update (gradient ascent on μ)
        with torch.no_grad():
            self.mu = torch.clamp(
                self.mu + self.mu_lr * (constraint_violation.item() - self.epsilon),
                min=0.0,
            )
        
        return {
            'task_loss': task_loss.item(),
            'constraint': constraint_violation.item(),
            'mu': self.mu.item(),
        }

# Demonstration
def demo():
    torch.manual_seed(42)
    n, d = 1000, 10
    X = torch.randn(n, d)
    A = torch.randint(0, 2, (n,))
    y = torch.bernoulli(torch.sigmoid(X[:, 0] + 0.5 * A.float()))
    
    model = ConstrainedFairClassifier(d, epsilon=0.01)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.005)
    
    print("Constrained Fair Training (ε=0.01)")
    print("=" * 55)
    for epoch in range(300):
        metrics = model.train_step(X, y, A, optimizer)
        if (epoch + 1) % 75 == 0:
            print(f"  Epoch {epoch+1}: loss={metrics['task_loss']:.4f}, "
                  f"constraint={metrics['constraint']:.6f}, μ={metrics['mu']:.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Formulates fairness as an **explicit constraint** rather than a penalty
- **Lagrangian relaxation** with dual ascent enforces the constraint adaptively
- The multiplier $\mu$ automatically adjusts the fairness–accuracy tradeoff
- $\epsilon$ provides a **hard bound** on the acceptable fairness violation

## Next Steps

- [Regularization](regularization.md): Penalty-based fairness approaches
- [Multi-Objective](multi_objective.md): Pareto optimization across fairness criteria
