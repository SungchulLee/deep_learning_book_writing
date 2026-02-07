# Counterfactual Fairness

## Definition

**Counterfactual Fairness** (Kusner et al., 2017) uses causal reasoning to define fairness at the individual level: a prediction is fair if it would remain the same in a counterfactual world where the individual belonged to a different protected group, holding all else equal.

### Mathematical Formulation

Given a causal model $(U, V, F)$ with latent variables $U$, observed variables $V$, and structural equations $F$, a predictor $\hat{Y}$ is counterfactually fair if for all individuals and all group values $a, a'$:

$$P(\hat{Y}_{A \leftarrow a}(U) = y \mid X = x, A = a) = P(\hat{Y}_{A \leftarrow a'}(U) = y \mid X = x, A = a)$$

where $\hat{Y}_{A \leftarrow a'}(U)$ denotes the prediction in the counterfactual world where $A$ is set to $a'$.

In simpler terms: "What would the model predict for this person if they had been in a different group, but were otherwise the same person?"

### Relationship to Causal Models

Counterfactual fairness requires a **causal graph** specifying:

1. How $A$ influences features $X$ (direct and indirect paths)
2. Which features are *descendants* of $A$ (and thus potentially tainted)
3. Which features are *non-descendants* of $A$ (and thus fair to use)

A sufficient condition for counterfactual fairness: the predictor $\hat{Y}$ depends only on non-descendants of $A$ in the causal graph.

## Causal Graph Example

```
A (protected) ──→ X₁ (proxy) ──→ Y
      │                              ↑
      └──→ X₂ (affected) ──────────┘
                                     ↑
U (latent) ──→ X₃ (legitimate) ────┘
```

- $X_1$: Direct descendant of $A$ (e.g., ZIP code correlated with race)
- $X_2$: Indirect descendant of $A$ (e.g., education affected by socioeconomic factors)
- $X_3$: Non-descendant of $A$ (e.g., innate ability proxied by test scores after debiasing)

A counterfactually fair predictor may only use $X_3$ and the latent $U$.

## PyTorch Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CounterfactualFairnessMetrics:
    """Metrics for counterfactual fairness assessment."""
    avg_counterfactual_gap: float
    max_counterfactual_gap: float
    fraction_unfair: float
    is_fair: bool

class CounterfactualFairnessChecker:
    """
    Assess counterfactual fairness using a structural causal model.
    
    For each individual, computes the prediction under the factual
    and counterfactual group assignment, then measures the gap.
    """
    
    def __init__(
        self,
        causal_model: Optional[nn.Module] = None,
        threshold: float = 0.05,
    ):
        self.causal_model = causal_model
        self.threshold = threshold
    
    def generate_counterfactual(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        descendant_indices: List[int],
        structural_equations: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Generate counterfactual features by flipping A and
        re-computing descendant features.
        
        Args:
            X: Original features
            A: Protected attribute
            descendant_indices: Indices of features that are
                descendants of A in the causal graph
            structural_equations: Optional dict mapping feature index
                to a function of (A, U, X_parents)
                
        Returns:
            Counterfactual feature matrix X_cf
        """
        X_cf = X.clone()
        A_cf = 1 - A  # Flip group membership
        
        if structural_equations is not None:
            for idx, eq in structural_equations.items():
                X_cf[:, idx] = eq(A_cf, X)
        else:
            # Simple linear approximation: remove A's effect on descendants
            for idx in descendant_indices:
                # Estimate A's effect via group mean difference
                mean_0 = X[A == 0, idx].mean()
                mean_1 = X[A == 1, idx].mean()
                effect = mean_1 - mean_0
                
                # Counterfactual: shift by estimated effect
                shift = torch.where(A == 0, effect, -effect)
                X_cf[:, idx] = X[:, idx] + shift
        
        return X_cf
    
    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        A: torch.Tensor,
        descendant_indices: List[int],
    ) -> CounterfactualFairnessMetrics:
        """
        Compute counterfactual fairness metrics.
        
        Args:
            model: Predictor to evaluate
            X: Features
            A: Protected attribute
            descendant_indices: Feature indices descending from A
            
        Returns:
            CounterfactualFairnessMetrics
        """
        model.eval()
        with torch.no_grad():
            # Factual predictions
            scores_factual = model(X).squeeze(-1)
            
            # Counterfactual features
            X_cf = self.generate_counterfactual(X, A, descendant_indices)
            scores_cf = model(X_cf).squeeze(-1)
            
            # Counterfactual gap per individual
            gaps = torch.abs(scores_factual - scores_cf)
        
        return CounterfactualFairnessMetrics(
            avg_counterfactual_gap=gaps.mean().item(),
            max_counterfactual_gap=gaps.max().item(),
            fraction_unfair=(gaps > self.threshold).float().mean().item(),
            is_fair=gaps.mean().item() < self.threshold,
        )


class CounterfactuallyFairPredictor(nn.Module):
    """
    Predictor that uses only non-descendant features of A.
    
    By construction, this predictor ignores all causal pathways
    from A to Ŷ, achieving counterfactual fairness.
    """
    
    def __init__(
        self,
        n_features: int,
        non_descendant_indices: List[int],
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.non_descendant_indices = non_descendant_indices
        input_dim = len(non_descendant_indices)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_fair = X[:, self.non_descendant_indices]
        return self.network(X_fair)


# --- Demonstration ---

def demonstrate_counterfactual_fairness():
    """Compare standard vs. counterfactually fair predictors."""
    torch.manual_seed(42)
    np.random.seed(42)
    n = 1000
    
    # Causal structure:
    # A → X0 (proxy), A → X1 (affected)
    # U → X2 (legitimate), U → X3 (legitimate)
    A = torch.randint(0, 2, (n,)).float()
    U = torch.randn(n)
    
    X0 = 0.8 * A + 0.2 * torch.randn(n)         # Strongly caused by A
    X1 = 0.4 * A + 0.3 * U + 0.3 * torch.randn(n)  # Partially by A
    X2 = 0.7 * U + 0.3 * torch.randn(n)          # Not caused by A
    X3 = 0.5 * U + 0.5 * torch.randn(n)          # Not caused by A
    
    X = torch.stack([X0, X1, X2, X3], dim=1)
    y = (0.3 * X0 + 0.2 * X1 + 0.3 * X2 + 0.2 * X3 > 0.5).float()
    
    # Standard model: uses all features
    standard_model = nn.Sequential(
        nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
    )
    
    # Fair model: uses only non-descendants (X2, X3)
    fair_model = CounterfactuallyFairPredictor(
        n_features=4,
        non_descendant_indices=[2, 3],
    )
    
    # Train both
    for name, model, features in [
        ("Standard", standard_model, X),
        ("CF-Fair", fair_model, X),
    ]:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            pred = model(features).squeeze(-1)
            loss = nn.functional.binary_cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
    
    # Evaluate counterfactual fairness
    checker = CounterfactualFairnessChecker(threshold=0.05)
    
    m_std = checker.compute(standard_model, X, A.long(), descendant_indices=[0, 1])
    m_fair = checker.compute(fair_model, X, A.long(), descendant_indices=[0, 1])
    
    print("Counterfactual Fairness Demonstration")
    print("=" * 55)
    print(f"\nStandard model (uses all features):")
    print(f"  Avg CF gap: {m_std.avg_counterfactual_gap:.4f}")
    print(f"  Fraction unfair: {m_std.fraction_unfair:.4f}")
    print(f"  Fair: {'YES' if m_std.is_fair else 'NO'}")
    
    print(f"\nCF-Fair model (non-descendants only):")
    print(f"  Avg CF gap: {m_fair.avg_counterfactual_gap:.4f}")
    print(f"  Fraction unfair: {m_fair.fraction_unfair:.4f}")
    print(f"  Fair: {'YES' if m_fair.is_fair else 'NO'}")

if __name__ == "__main__":
    demonstrate_counterfactual_fairness()
```

## Advantages and Limitations

**Advantages:**

- Provides individual-level fairness guarantees rooted in causal reasoning
- Accounts for complex causal pathways from protected attributes to outcomes
- Does not require defining an explicit similarity metric (unlike Dwork-style individual fairness)

**Limitations:**

- Requires a causal graph, which may be unknown or contested
- Counterfactual reasoning depends on untestable assumptions about latent variables
- May be overly conservative: excludes *all* descendant features, including those with legitimate variation

## Summary

- **Counterfactual fairness** asks: "Would the prediction change if the individual's group were different?"
- Requires a **causal model** specifying how $A$ influences features
- Achieved by building predictors that use only **non-descendants** of $A$
- Provides strong individual-level guarantees but requires causal knowledge
- Complements statistical fairness criteria with causal reasoning

## Next Steps

- [Chouldechova's Theorem](../impossibility/chouldechova.md): Why statistical criteria conflict
- [Causal Metrics](../metrics/causal.md): Practical causal fairness measurement
