# Individual Fairness

## Definition

**Individual Fairness** (Dwork et al., 2012) requires that similar individuals receive similar predictions. Unlike group fairness criteria that compare aggregate statistics across groups, individual fairness operates at the level of individual data points.

### Mathematical Formulation

A classifier $f$ satisfies individual fairness if it is Lipschitz-continuous with respect to a task-specific similarity metric $d$:

$$d_{\text{outcome}}(f(x_i), f(x_j)) \leq L \cdot d_{\text{input}}(x_i, x_j) \quad \forall\; x_i, x_j$$

where $d_{\text{input}}$ measures similarity between individuals, $d_{\text{outcome}}$ measures similarity between predictions, and $L$ is the Lipschitz constant.

For probabilistic classifiers producing scores $S(x) \in [0, 1]$:

$$|S(x_i) - S(x_j)| \leq L \cdot d(x_i, x_j)$$

### Key Challenge: Defining the Metric

The central challenge of individual fairness is defining the similarity metric $d_{\text{input}}$. This metric should capture "who should be treated similarly" and inherently encodes normative judgments about which differences between individuals are relevant to the task.

## PyTorch Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Callable, Optional
from dataclasses import dataclass

@dataclass
class IndividualFairnessMetrics:
    """Container for individual fairness metrics."""
    avg_lipschitz_violation: float
    max_lipschitz_violation: float
    fraction_violations: float
    is_fair: bool

class IndividualFairnessCalculator:
    """
    Assess individual fairness via Lipschitz condition.
    
    For all pairs (i, j):
        |f(x_i) - f(x_j)| ≤ L · d(x_i, x_j)
    """
    
    def __init__(
        self,
        distance_fn: Optional[Callable] = None,
        lipschitz_constant: float = 1.0,
        n_pairs: int = 5000,
    ):
        """
        Args:
            distance_fn: Custom similarity metric d(x_i, x_j).
                         Defaults to L2 distance with protected attributes removed.
            lipschitz_constant: Maximum allowed ratio |Δf| / d
            n_pairs: Number of random pairs to sample for evaluation
        """
        self.distance_fn = distance_fn or self._default_distance
        self.L = lipschitz_constant
        self.n_pairs = n_pairs
    
    @staticmethod
    def _default_distance(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Default: Euclidean distance."""
        return torch.norm(x_i - x_j, dim=-1)
    
    def compute(
        self,
        X: torch.Tensor,
        scores: torch.Tensor,
    ) -> IndividualFairnessMetrics:
        """
        Estimate individual fairness by sampling random pairs.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            scores: Predicted scores, shape (n_samples,)
            
        Returns:
            IndividualFairnessMetrics
        """
        n = len(X)
        idx_i = torch.randint(0, n, (self.n_pairs,))
        idx_j = torch.randint(0, n, (self.n_pairs,))
        
        # Input distances
        d_input = self.distance_fn(X[idx_i], X[idx_j])
        
        # Output distances
        d_output = torch.abs(scores[idx_i] - scores[idx_j])
        
        # Lipschitz violations: |f(xi) - f(xj)| > L · d(xi, xj)
        violations = d_output - self.L * d_input
        violation_mask = violations > 0
        
        avg_viol = violations[violation_mask].mean().item() if violation_mask.any() else 0.0
        max_viol = violations.max().item()
        frac_viol = violation_mask.float().mean().item()
        
        return IndividualFairnessMetrics(
            avg_lipschitz_violation=avg_viol,
            max_lipschitz_violation=max(0, max_viol),
            fraction_violations=frac_viol,
            is_fair=frac_viol < 0.05,
        )


def individual_fairness_regularizer(
    model: nn.Module,
    X: torch.Tensor,
    distance_fn: Optional[Callable] = None,
    n_pairs: int = 500,
    lipschitz_constant: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable individual fairness regularizer for training.
    
    Penalizes pairs where |f(x_i) - f(x_j)| > L · d(x_i, x_j).
    
    Loss = (1/K) Σ max(0, |f(x_i) - f(x_j)| - L · d(x_i, x_j))²
    
    Args:
        model: Neural network producing scores
        X: Input batch
        distance_fn: Similarity metric (default: L2)
        n_pairs: Number of pairs to sample
        lipschitz_constant: Lipschitz bound L
        
    Returns:
        Scalar regularization loss
    """
    if distance_fn is None:
        distance_fn = lambda a, b: torch.norm(a - b, dim=-1)
    
    n = len(X)
    idx_i = torch.randint(0, n, (n_pairs,))
    idx_j = torch.randint(0, n, (n_pairs,))
    
    scores = model(X).squeeze(-1)
    
    d_input = distance_fn(X[idx_i], X[idx_j])
    d_output = torch.abs(scores[idx_i] - scores[idx_j])
    
    violations = torch.relu(d_output - lipschitz_constant * d_input)
    
    return (violations ** 2).mean()


# --- Demonstration ---

def demonstrate_individual_fairness():
    """Compare models with and without individual fairness."""
    torch.manual_seed(42)
    n = 500
    
    # Features: first 2 are legitimate, last is a proxy for group
    X = torch.randn(n, 3)
    group = (X[:, 2] > 0).long()
    
    # True score depends only on legitimate features
    true_score = torch.sigmoid(X[:, 0] + X[:, 1])
    
    # Model A: Uses all features (including proxy)
    score_a = torch.sigmoid(X[:, 0] + X[:, 1] + 0.5 * X[:, 2])
    
    # Model B: Uses only legitimate features
    score_b = torch.sigmoid(X[:, 0] + X[:, 1]) + torch.randn(n) * 0.01
    
    # Evaluate using only legitimate features for distance
    calc = IndividualFairnessCalculator(
        distance_fn=lambda a, b: torch.norm(a[:, :2] - b[:, :2], dim=-1),
        lipschitz_constant=1.0,
    )
    
    m_a = calc.compute(X, score_a)
    m_b = calc.compute(X, score_b)
    
    print("Individual Fairness Demonstration")
    print("=" * 50)
    print(f"\nModel A (uses proxy feature):")
    print(f"  Fraction of violations: {m_a.fraction_violations:.4f}")
    print(f"  Avg violation: {m_a.avg_lipschitz_violation:.4f}")
    print(f"  Fair: {'YES' if m_a.is_fair else 'NO'}")
    
    print(f"\nModel B (legitimate features only):")
    print(f"  Fraction of violations: {m_b.fraction_violations:.4f}")
    print(f"  Avg violation: {m_b.avg_lipschitz_violation:.4f}")
    print(f"  Fair: {'YES' if m_b.is_fair else 'NO'}")

if __name__ == "__main__":
    demonstrate_individual_fairness()
```

## Comparison with Group Fairness

| Aspect | Group Fairness | Individual Fairness |
|--------|---------------|-------------------|
| Granularity | Group-level statistics | Individual pairs |
| Metric needed | Group labels | Similarity function $d$ |
| Typical violation | Aggregate rate differences | Similar individuals treated differently |
| Challenge | Choosing which group criterion | Defining the similarity metric |

## Practical Considerations

1. **Metric elicitation**: Defining $d$ often requires domain expertise or can be learned from data (metric learning)
2. **Computational cost**: Evaluating all pairs is $O(n^2)$; sampling approximations are necessary
3. **Tension with group fairness**: Individual fairness does not guarantee group fairness, and vice versa
4. **Feature selection**: The distance metric implicitly defines which features are "legitimate" for decision-making

## Summary

- Individual fairness requires **similar individuals to receive similar predictions**
- Formalized as a **Lipschitz condition** with respect to a task-specific distance metric
- The **metric $d$ encodes normative judgments** about which individual differences matter
- Can be enforced via **regularization** during training
- Complements rather than replaces group fairness criteria

## Next Steps

- [Counterfactual Fairness](counterfactual.md): Causal approach to individual-level fairness
- [Causal Metrics](../metrics/causal.md): Measuring fairness through causal lenses
