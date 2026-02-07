# Disparate Impact Testing

## Overview

**Disparate impact testing** applies statistical hypothesis testing to determine whether observed fairness violations are statistically significant or could arise from sampling variability. This is critical for regulatory compliance, where a fairness violation must be demonstrated with statistical confidence.

## Statistical Tests

### Two-Proportion Z-Test for SPD

Test whether positive prediction rates differ significantly between groups:

$$H_0: P(\hat{Y}=1 \mid A=0) = P(\hat{Y}=1 \mid A=1)$$

$$Z = \frac{\hat{p}_0 - \hat{p}_1}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_0} + \frac{1}{n_1}\right)}}$$

where $\hat{p}$ is the pooled positive rate.

### Permutation Test

A nonparametric alternative that makes no distributional assumptions:

1. Compute the observed SPD
2. Repeatedly shuffle group labels and recompute SPD
3. The p-value is the fraction of permutations with SPD ≥ observed

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict
from dataclasses import dataclass
from scipy import stats as scipy_stats

@dataclass
class DisparateImpactTestResult:
    """Statistical test results for disparate impact."""
    observed_spd: float
    observed_dir: float
    z_statistic: float
    p_value_parametric: float
    p_value_permutation: float
    significant_at_005: bool
    passes_four_fifths: bool

class DisparateImpactTester:
    """Statistical testing for disparate impact."""
    
    def __init__(self, n_permutations: int = 10000):
        self.n_permutations = n_permutations
    
    def two_proportion_z_test(
        self, y_pred: torch.Tensor, A: torch.Tensor,
    ) -> tuple:
        """Two-proportion z-test for difference in positive rates."""
        n0 = (A == 0).sum().item()
        n1 = (A == 1).sum().item()
        p0 = y_pred[A == 0].float().mean().item()
        p1 = y_pred[A == 1].float().mean().item()
        
        p_pooled = y_pred.float().mean().item()
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n0 + 1/n1))
        
        z = (p0 - p1) / se if se > 0 else 0.0
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def permutation_test(
        self, y_pred: torch.Tensor, A: torch.Tensor,
    ) -> float:
        """Permutation test for SPD significance."""
        observed_spd = abs(
            y_pred[A == 0].float().mean() - y_pred[A == 1].float().mean()
        ).item()
        
        count = 0
        for _ in range(self.n_permutations):
            perm = torch.randperm(len(A))
            A_perm = A[perm]
            perm_spd = abs(
                y_pred[A_perm == 0].float().mean() -
                y_pred[A_perm == 1].float().mean()
            ).item()
            if perm_spd >= observed_spd:
                count += 1
        
        return count / self.n_permutations
    
    def test(
        self, y_pred: torch.Tensor, A: torch.Tensor,
    ) -> DisparateImpactTestResult:
        """Run full disparate impact test battery."""
        p0 = y_pred[A == 0].float().mean().item()
        p1 = y_pred[A == 1].float().mean().item()
        spd = abs(p0 - p1)
        dir_ratio = min(p0, p1) / max(p0, p1) if max(p0, p1) > 0 else 0
        
        z, p_param = self.two_proportion_z_test(y_pred, A)
        p_perm = self.permutation_test(y_pred, A)
        
        return DisparateImpactTestResult(
            observed_spd=spd,
            observed_dir=dir_ratio,
            z_statistic=z,
            p_value_parametric=p_param,
            p_value_permutation=p_perm,
            significant_at_005=p_param < 0.05,
            passes_four_fifths=dir_ratio >= 0.8,
        )

# Demonstration
def demo():
    torch.manual_seed(42)
    n = 2000
    A = torch.randint(0, 2, (n,))
    y_pred = torch.where(A == 0,
        torch.tensor(np.random.choice([0,1], n, p=[0.35, 0.65])),
        torch.tensor(np.random.choice([0,1], n, p=[0.50, 0.50])))
    
    tester = DisparateImpactTester(n_permutations=5000)
    result = tester.test(y_pred, A)
    
    print("Disparate Impact Testing")
    print("=" * 50)
    print(f"Observed SPD: {result.observed_spd:.4f}")
    print(f"Observed DIR: {result.observed_dir:.4f}")
    print(f"Z-statistic: {result.z_statistic:.4f}")
    print(f"P-value (parametric): {result.p_value_parametric:.4f}")
    print(f"P-value (permutation): {result.p_value_permutation:.4f}")
    print(f"Significant at α=0.05: {result.significant_at_005}")
    print(f"Passes 4/5 rule: {result.passes_four_fifths}")

if __name__ == "__main__":
    demo()
```

## Summary

- **Two-proportion z-test** provides a parametric test for rate differences
- **Permutation test** provides a nonparametric alternative with no distributional assumptions
- Both are needed for **regulatory submissions** where statistical significance matters
- The **four-fifths rule** provides a simple practical threshold ($\text{DIR} \geq 0.8$)

## Next Steps

- [Longitudinal Analysis](longitudinal.md): Tracking fairness over time in production
