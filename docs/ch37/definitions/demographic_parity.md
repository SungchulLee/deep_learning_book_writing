# Demographic Parity

## Definition

**Demographic Parity** (also called **Statistical Parity** or **Group Fairness**) requires that the probability of receiving a positive prediction is equal across all protected groups.

### Mathematical Formulation

A classifier $\hat{Y}$ satisfies demographic parity with respect to protected attribute $A$ if:

$$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$

More generally, for multiple groups $A \in \{0, 1, \ldots, k\}$:

$$P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = a') \quad \forall\; a, a' \in \{0, 1, \ldots, k\}$$

### Equivalent Formulation: Independence

Demographic parity is equivalent to requiring that the prediction $\hat{Y}$ is **independent** of the protected attribute $A$:

$$\hat{Y} \perp\!\!\!\perp A$$

This means knowing the group membership should provide no information about the prediction.

## Measuring Demographic Parity

### Statistical Parity Difference (SPD)

The most common measure is the difference in positive prediction rates:

$$\text{SPD} = \bigl|P(\hat{Y} = 1 \mid A = 0) - P(\hat{Y} = 1 \mid A = 1)\bigr|$$

- **Perfect fairness**: $\text{SPD} = 0$
- **Acceptable range**: $\text{SPD} < 0.1$ (commonly used threshold)
- **Problematic**: $\text{SPD} > 0.2$

### Disparate Impact Ratio (DIR)

The ratio of positive rates, often used in legal contexts (the "80% rule"):

$$\text{DIR} = \frac{\min\bigl(P(\hat{Y}=1 \mid A=0),\; P(\hat{Y}=1 \mid A=1)\bigr)}{\max\bigl(P(\hat{Y}=1 \mid A=0),\; P(\hat{Y}=1 \mid A=1)\bigr)}$$

- **Perfect fairness**: $\text{DIR} = 1.0$
- **Legal threshold**: $\text{DIR} \geq 0.8$ (EEOC four-fifths rule)
- **Problematic**: $\text{DIR} < 0.8$

## PyTorch Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DemographicParityMetrics:
    """Container for demographic parity metrics."""
    group_positive_rates: Dict[int, float]
    statistical_parity_difference: float
    disparate_impact_ratio: float
    is_fair_spd: bool   # SPD < threshold
    is_fair_dir: bool   # DIR >= 0.8


class DemographicParityCalculator:
    """
    Calculate demographic parity metrics for binary classification.
    
    Demographic Parity requires equal positive prediction rates across groups:
        P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
    """
    
    def __init__(
        self,
        spd_threshold: float = 0.1,
        dir_threshold: float = 0.8,
    ):
        self.spd_threshold = spd_threshold
        self.dir_threshold = dir_threshold
    
    def compute_positive_rate(
        self, y_pred: torch.Tensor, mask: torch.Tensor,
    ) -> float:
        """Compute positive prediction rate for a single group."""
        if mask.sum() == 0:
            return 0.0
        return y_pred[mask].float().mean().item()
    
    def compute(
        self,
        y_pred: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> DemographicParityMetrics:
        """
        Compute all demographic parity metrics.
        
        Args:
            y_pred: Predicted labels, shape (n_samples,)
            sensitive_attr: Protected attribute values, shape (n_samples,)
            
        Returns:
            DemographicParityMetrics with all computed metrics
        """
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        if not isinstance(sensitive_attr, torch.Tensor):
            sensitive_attr = torch.tensor(sensitive_attr)
        
        groups = torch.unique(sensitive_attr)
        positive_rates = {}
        for group in groups:
            gv = group.item()
            mask = sensitive_attr == group
            positive_rates[gv] = self.compute_positive_rate(y_pred, mask)
        
        rates = list(positive_rates.values())
        if len(rates) >= 2:
            spd = abs(rates[0] - rates[1])
            max_rate, min_rate = max(rates), min(rates)
            dir_ratio = min_rate / max_rate if max_rate > 0 else 0.0
        else:
            spd = 0.0
            dir_ratio = 1.0
        
        return DemographicParityMetrics(
            group_positive_rates=positive_rates,
            statistical_parity_difference=spd,
            disparate_impact_ratio=dir_ratio,
            is_fair_spd=spd < self.spd_threshold,
            is_fair_dir=dir_ratio >= self.dir_threshold,
        )
    
    def compute_from_probabilities(
        self,
        y_prob: torch.Tensor,
        sensitive_attr: torch.Tensor,
        threshold: float = 0.5,
    ) -> DemographicParityMetrics:
        """Compute metrics from predicted probabilities."""
        y_pred = (y_prob >= threshold).long()
        return self.compute(y_pred, sensitive_attr)


def demographic_parity_loss(
    y_prob: torch.Tensor,
    sensitive_attr: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable demographic parity loss for training.
    
    Penalizes differences in average predicted probabilities across
    protected groups, enabling end-to-end fairness optimization.
    
    Loss = (E[ŷ | A=0] - E[ŷ | A=1])²
    
    Args:
        y_prob: Predicted probabilities, shape (n_samples,)
        sensitive_attr: Protected attribute values (0 or 1)
        epsilon: Small constant for numerical stability
        
    Returns:
        Scalar loss representing demographic parity violation
    """
    if not y_prob.requires_grad:
        y_prob = y_prob.requires_grad_(True)
    
    group_0_mask = (sensitive_attr == 0).float()
    group_1_mask = (sensitive_attr == 1).float()
    
    n_group_0 = group_0_mask.sum() + epsilon
    n_group_1 = group_1_mask.sum() + epsilon
    
    avg_prob_0 = (y_prob * group_0_mask).sum() / n_group_0
    avg_prob_1 = (y_prob * group_1_mask).sum() / n_group_1
    
    return (avg_prob_0 - avg_prob_1) ** 2


class FairClassifier(nn.Module):
    """
    Neural network classifier with demographic parity regularization.
    
    Total loss: L = L_BCE + λ · L_DP
    
    where L_DP = (E[ŷ|A=0] - E[ŷ|A=1])²
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        fairness_weight: float = 1.0,
    ):
        super().__init__()
        self.fairness_weight = fairness_weight
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined classification + fairness loss.
        
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        y_prob = self.forward(x)
        
        bce_loss = nn.functional.binary_cross_entropy(
            y_prob, y_true.float(), reduction='mean'
        )
        dp_loss = demographic_parity_loss(y_prob, sensitive_attr)
        total_loss = bce_loss + self.fairness_weight * dp_loss
        
        return total_loss, {
            'bce_loss': bce_loss.item(),
            'dp_loss': dp_loss.item(),
            'total_loss': total_loss.item(),
        }


# --- Demonstration ---

def demonstrate_demographic_parity():
    """Comprehensive demonstration of demographic parity."""
    np.random.seed(42)
    n_samples = 1000
    
    sensitive_attr = torch.tensor(np.random.randint(0, 2, n_samples))
    calculator = DemographicParityCalculator()
    
    # Scenario 1: Biased model
    print("=" * 65)
    print("Scenario 1: BIASED Model")
    print("=" * 65)
    y_pred_biased = torch.where(
        sensitive_attr == 0,
        torch.tensor(np.random.choice([0, 1], n_samples, p=[0.3, 0.7])),
        torch.tensor(np.random.choice([0, 1], n_samples, p=[0.6, 0.4])),
    )
    m = calculator.compute(y_pred_biased, sensitive_attr)
    for g, r in m.group_positive_rates.items():
        print(f"  Group {g} positive rate: {r:.4f}")
    print(f"  SPD: {m.statistical_parity_difference:.4f}  "
          f"({'PASS' if m.is_fair_spd else 'FAIL'})")
    print(f"  DIR: {m.disparate_impact_ratio:.4f}  "
          f"({'PASS' if m.is_fair_dir else 'FAIL'})")
    
    # Scenario 2: Fair model
    print(f"\n{'=' * 65}")
    print("Scenario 2: FAIR Model")
    print("=" * 65)
    y_pred_fair = torch.tensor(np.random.choice([0, 1], n_samples, p=[0.45, 0.55]))
    m = calculator.compute(y_pred_fair, sensitive_attr)
    for g, r in m.group_positive_rates.items():
        print(f"  Group {g} positive rate: {r:.4f}")
    print(f"  SPD: {m.statistical_parity_difference:.4f}  "
          f"({'PASS' if m.is_fair_spd else 'FAIL'})")
    print(f"  DIR: {m.disparate_impact_ratio:.4f}  "
          f"({'PASS' if m.is_fair_dir else 'FAIL'})")
    
    # Scenario 3: Training with DP regularization
    print(f"\n{'=' * 65}")
    print("Scenario 3: Training with DP Regularization (λ = 5.0)")
    print("=" * 65)
    torch.manual_seed(42)
    n_train, input_dim = 500, 10
    X = torch.randn(n_train, input_dim)
    sensitive = torch.randint(0, 2, (n_train,))
    base_prob = torch.sigmoid(X[:, 0] + X[:, 1])
    bias_factor = torch.where(sensitive == 0, torch.tensor(0.2), torch.tensor(-0.2))
    y = (base_prob + bias_factor > 0.5).float()
    
    model = FairClassifier(input_dim, hidden_dim=32, fairness_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model.compute_loss(X, y, sensitive)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}: BCE={metrics['bce_loss']:.4f}, "
                  f"DP={metrics['dp_loss']:.6f}, Total={metrics['total_loss']:.4f}")
    
    model.eval()
    with torch.no_grad():
        y_pred_final = (model(X) >= 0.5).long()
    m = calculator.compute(y_pred_final, sensitive)
    print(f"\n  Final SPD: {m.statistical_parity_difference:.4f}")
    print(f"  Final DIR: {m.disparate_impact_ratio:.4f}")
    print(f"  Accuracy:  {(y_pred_final == y.long()).float().mean():.4f}")

if __name__ == "__main__":
    demonstrate_demographic_parity()
```

## When to Use Demographic Parity

### Appropriate Use Cases

1. **Equal access requirements**: When the goal is equal access to resources regardless of qualifications
2. **Affirmative action**: When actively correcting historical underrepresentation
3. **No reliable ground truth**: When true labels may themselves be biased
4. **Outcome equality**: When equal outcomes are legally or ethically required

### Limitations

1. **May reduce utility**: Ignores whether individuals are actually qualified
2. **Laziness problem**: Random prediction trivially satisfies DP
3. **Ignores base rates**: Groups with different qualification rates are treated identically
4. **Accuracy tradeoff**: Strict enforcement may significantly reduce prediction quality

## Relationship to Other Fairness Criteria

| Criterion | Condition | Conditioning Variable |
|-----------|-----------|----------------------|
| Demographic Parity | $P(\hat{Y}=1 \mid A=0) = P(\hat{Y}=1 \mid A=1)$ | None |
| Equal Opportunity | $P(\hat{Y}=1 \mid Y=1, A=0) = P(\hat{Y}=1 \mid Y=1, A=1)$ | $Y = 1$ |
| Equalized Odds | $P(\hat{Y}=1 \mid Y=y, A=0) = P(\hat{Y}=1 \mid Y=y, A=1)$ | $Y = y$ |

### Mathematical Tradeoff

If base rates differ, $P(Y=1 \mid A=0) \neq P(Y=1 \mid A=1)$, then demographic parity and accuracy-optimal classification are generally incompatible.

**Theorem.** For a classifier $\hat{Y}$ with base rate difference $\delta = |P(Y=1 \mid A=0) - P(Y=1 \mid A=1)| > 0$, satisfying exact demographic parity requires suboptimal accuracy.

## Summary

- **Demographic Parity** requires equal positive prediction rates across groups
- **SPD** and **DIR** are the primary metrics, with thresholds of 0.1 and 0.8 respectively
- **Differentiable losses** enable end-to-end fairness optimization via regularization
- Use when **equal outcomes** are the primary concern
- Be aware of **tradeoffs** with accuracy and incompatibility with other fairness definitions

## Next Steps

- [Equal Opportunity](equal_opportunity.md): Fairness conditioned on the true positive class
- [Equalized Odds](equalized_odds.md): Combining TPR and FPR fairness
