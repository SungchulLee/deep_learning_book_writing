# Adversarial Debiasing

## Overview

**Adversarial Debiasing** (Zhang, Lemoine, & Mitchell, 2018) trains a predictor and an adversary simultaneously. The predictor tries to predict $Y$ from $X$, while the adversary tries to predict $A$ from the predictor's output. The predictor is trained to fool the adversary—achieving predictions from which the protected attribute cannot be inferred.

## Architecture

```
X ──→ [Predictor] ──→ Ŷ ──→ [Adversary] ──→ Â
         ↑                        ↑
     minimize L_task          maximize L_adv
     maximize L_adv           minimize L_adv
```

The predictor's loss combines task accuracy with adversarial confusion:

$$\mathcal{L}_{\text{predictor}} = \mathcal{L}_{\text{task}}(\hat{Y}, Y) - \lambda \cdot \mathcal{L}_{\text{adv}}(\hat{A}, A)$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch.autograd import Function

class GradientReversalFunction(Function):
    """Reverse gradients during backpropagation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class AdversarialDebiasing(nn.Module):
    """
    Adversarial debiasing model with gradient reversal.
    
    The predictor learns to predict Y while the adversary (with
    reversed gradients) forces the predictor's internal representations
    to be uninformative about A.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        adversary_weight: float = 1.0,
    ):
        super().__init__()
        self.adversary_weight = adversary_weight
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Task head: predict Y
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Adversary head: predict A (with gradient reversal)
        self.adversary = nn.Sequential(
            GradientReversal(alpha=adversary_weight),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        y_pred = self.predictor(h).squeeze(-1)
        a_pred = self.adversary(h).squeeze(-1)
        return y_pred, a_pred
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        y_pred, a_pred = self.forward(x)
        
        task_loss = nn.functional.binary_cross_entropy(y_pred, y.float())
        adv_loss = nn.functional.binary_cross_entropy(a_pred, a.float())
        
        # Gradient reversal handles the sign flip automatically
        total_loss = task_loss + adv_loss
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'adv_loss': adv_loss.item(),
            'total_loss': total_loss.item(),
        }


# Demonstration
def demo():
    torch.manual_seed(42)
    n, d = 1000, 10
    X = torch.randn(n, d)
    A = torch.randint(0, 2, (n,))
    y = torch.bernoulli(torch.sigmoid(X[:, 0] + 0.5 * A.float()))
    
    model = AdversarialDebiasing(d, hidden_dim=32, adversary_weight=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Adversarial Debiasing Training")
    print("=" * 50)
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model.compute_loss(X, y, A)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 75 == 0:
            print(f"  Epoch {epoch+1}: task={metrics['task_loss']:.4f}, "
                  f"adv={metrics['adv_loss']:.4f}")
    
    # Evaluate: adversary should be near chance (0.693 = -log(0.5))
    model.eval()
    with torch.no_grad():
        y_pred, a_pred = model(X)
    
    y_hat = (y_pred > 0.5).long()
    acc = (y_hat == y).float().mean()
    adv_acc = ((a_pred > 0.5).long() == A).float().mean()
    
    spd = abs(y_hat[A==0].float().mean() - y_hat[A==1].float().mean())
    print(f"\nResults: accuracy={acc:.4f}, SPD={spd:.4f}, "
          f"adversary_acc={adv_acc:.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Uses **gradient reversal** to make learned representations uninformative about $A$
- Single unified training procedure with one optimizer
- The **adversary weight** $\lambda$ controls the fairness–accuracy tradeoff
- Adversary accuracy near 50% (chance) indicates successful debiasing

## Next Steps

- [Fairness Constraints](constraints.md): Explicit constraint-based approaches
- [Regularization](regularization.md): Regularization-based fairness
