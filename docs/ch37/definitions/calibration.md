# Calibration

## Definition

**Calibration** (also called **predictive parity** or **test fairness**) requires that predicted probabilities accurately reflect true outcome rates within each group. If a model assigns a risk score $S$, then among all individuals with score $S = s$, the fraction who are truly positive should be the same regardless of group membership.

### Mathematical Formulation

A score function $S$ satisfies calibration with respect to protected attribute $A$ if:

$$P(Y = 1 \mid S = s, A = 0) = P(Y = 1 \mid S = s, A = 1) \quad \forall\; s$$

Equivalently:

$$Y \perp\!\!\!\perp A \mid S$$

### Sufficiency

Calibration is equivalent to the **sufficiency** criterion: the score $S$ is *sufficient* for predicting $Y$ in the sense that conditioning on $S$ renders $A$ uninformative about $Y$.

## Motivation

Calibration ensures that predicted scores have the same meaning across groups. If a model assigns a 70% probability of default to both a Group 0 and a Group 1 applicant, calibration requires that both actually default at approximately 70%. This is critical for decision-making: stakeholders rely on predicted probabilities to set thresholds and allocate resources.

**Finance context.** In credit scoring, calibration means that a predicted default probability of 5% corresponds to a 5% actual default rate regardless of the borrower's demographic group. This is essential for pricing and capital allocation.

## PyTorch Implementation

```python
import numpy as np
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CalibrationMetrics:
    """Container for calibration fairness metrics."""
    group_calibration_errors: Dict[int, float]
    max_calibration_gap: float
    avg_calibration_gap: float
    is_fair: bool

class CalibrationFairnessCalculator:
    """
    Assess calibration fairness across protected groups.
    
    Calibration requires: P(Y=1 | S=s, A=0) = P(Y=1 | S=s, A=1)
    
    Measured by comparing calibration curves across groups.
    """
    
    def __init__(self, n_bins: int = 10, threshold: float = 0.05):
        """
        Args:
            n_bins: Number of bins for calibration curve
            threshold: Maximum acceptable calibration gap
        """
        self.n_bins = n_bins
        self.threshold = threshold
    
    def compute_calibration_curve(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Compute calibration curve (reliability diagram data).
        
        Returns:
            Tuple of (bin_centers, bin_accuracies, bin_counts)
        """
        bin_edges = torch.linspace(0, 1, self.n_bins + 1)
        bin_centers, bin_accs, bin_counts = [], [], []
        
        for i in range(self.n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == self.n_bins - 1:  # Include right edge in last bin
                mask = mask | (y_prob == bin_edges[i + 1])
            
            count = mask.sum().item()
            if count > 0:
                acc = y_true[mask].float().mean().item()
                center = (bin_edges[i] + bin_edges[i + 1]).item() / 2
                bin_centers.append(center)
                bin_accs.append(acc)
                bin_counts.append(count)
        
        return bin_centers, bin_accs, bin_counts
    
    def compute_ece(
        self, y_true: torch.Tensor, y_prob: torch.Tensor,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ_b (n_b / N) |acc(b) - conf(b)|
        """
        bin_edges = torch.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n_total = len(y_true)
        
        for i in range(self.n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = mask | (y_prob == bin_edges[i + 1])
            
            n_bin = mask.sum().item()
            if n_bin > 0:
                acc = y_true[mask].float().mean().item()
                conf = y_prob[mask].mean().item()
                ece += (n_bin / n_total) * abs(acc - conf)
        
        return ece
    
    def compute(
        self,
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> CalibrationMetrics:
        """
        Compute calibration fairness metrics.
        
        Compares ECE across groups and measures the maximum gap
        between group calibration curves.
        """
        y_true = torch.as_tensor(y_true).float()
        y_prob = torch.as_tensor(y_prob).float()
        sensitive_attr = torch.as_tensor(sensitive_attr)
        
        groups = torch.unique(sensitive_attr)
        group_ece = {}
        group_curves = {}
        
        for group in groups:
            gv = group.item()
            mask = sensitive_attr == group
            ece = self.compute_ece(y_true[mask], y_prob[mask])
            group_ece[gv] = ece
            group_curves[gv] = self.compute_calibration_curve(
                y_true[mask], y_prob[mask]
            )
        
        # Max gap between any two groups' ECE
        ece_vals = list(group_ece.values())
        max_gap = max(ece_vals) - min(ece_vals) if len(ece_vals) >= 2 else 0.0
        avg_gap = np.mean(ece_vals)
        
        return CalibrationMetrics(
            group_calibration_errors=group_ece,
            max_calibration_gap=max_gap,
            avg_calibration_gap=avg_gap,
            is_fair=max_gap < self.threshold,
        )


# --- Demonstration ---

def demonstrate_calibration_fairness():
    """Show calibrated vs. miscalibrated predictions across groups."""
    np.random.seed(42)
    n = 2000
    
    group = torch.tensor(np.random.randint(0, 2, n))
    
    # True probabilities
    true_prob = torch.rand(n) * 0.8 + 0.1  # Range [0.1, 0.9]
    y_true = torch.bernoulli(true_prob)
    
    # Well-calibrated model: predicted prob ≈ true prob for both groups
    y_prob_good = true_prob + torch.randn(n) * 0.05
    y_prob_good = y_prob_good.clamp(0, 1)
    
    # Miscalibrated model: overconfident for Group 1
    y_prob_bad = y_prob_good.clone()
    g1_mask = group == 1
    y_prob_bad[g1_mask] = (y_prob_bad[g1_mask] * 1.3).clamp(0, 1)
    
    calc = CalibrationFairnessCalculator()
    
    m_good = calc.compute(y_true, y_prob_good, group)
    m_bad = calc.compute(y_true, y_prob_bad, group)
    
    print("Calibration Fairness Demonstration")
    print("=" * 50)
    print("\nWell-calibrated model:")
    for g, ece in m_good.group_calibration_errors.items():
        print(f"  Group {g} ECE: {ece:.4f}")
    print(f"  Max gap: {m_good.max_calibration_gap:.4f}  "
          f"({'FAIR' if m_good.is_fair else 'UNFAIR'})")
    
    print("\nMiscalibrated model:")
    for g, ece in m_bad.group_calibration_errors.items():
        print(f"  Group {g} ECE: {ece:.4f}")
    print(f"  Max gap: {m_bad.max_calibration_gap:.4f}  "
          f"({'FAIR' if m_bad.is_fair else 'UNFAIR'})")

if __name__ == "__main__":
    demonstrate_calibration_fairness()
```

## Calibration vs. Error Rate Fairness

Calibration and error rate fairness (equalized odds) represent fundamentally different perspectives:

- **Calibration** asks: "Among people I scored as $s$, is the true positive rate the same across groups?"
- **Equalized odds** asks: "Among truly positive (or negative) people, is my prediction rate the same across groups?"

These correspond to conditioning in opposite directions, and they cannot be simultaneously satisfied when base rates differ (see [Chouldechova's Theorem](../impossibility/chouldechova.md)).

## Summary

- **Calibration** ensures predicted probabilities mean the same thing across groups
- Measured via **Expected Calibration Error (ECE)** per group and the gap between groups
- Critical in finance where predicted probabilities drive pricing and capital allocation
- Incompatible with equalized odds when base rates differ
- Represents the **sufficiency** criterion: $Y \perp\!\!\!\perp A \mid S$

## Next Steps

- [Individual Fairness](individual_fairness.md): Moving beyond group-level criteria
- [Chouldechova's Theorem](../impossibility/chouldechova.md): The impossibility of calibration + equal error rates
