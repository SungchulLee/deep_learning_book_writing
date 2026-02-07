# KMR Impossibility Theorem

## Overview

The Kleinberg–Mullainathan–Raghavan (KMR) impossibility theorem (2016) provides a complementary impossibility result to Chouldechova's theorem. It shows that three natural fairness conditions—calibration within groups, balance for the positive class, and balance for the negative class—cannot be simultaneously satisfied when base rates differ.

## Statement of the Theorem

**Theorem (Kleinberg, Mullainathan, & Raghavan, 2016).** Consider a risk score $S \in [0, 1]$ applied to two groups $A \in \{0, 1\}$. The following three conditions cannot all hold when $P(Y=1 \mid A=0) \neq P(Y=1 \mid A=1)$:

1. **Calibration within groups**: $\mathbb{E}[Y \mid S=s, A=a] = s \quad \forall s, a$
2. **Balance for the positive class**: $\mathbb{E}[S \mid Y=1, A=0] = \mathbb{E}[S \mid Y=1, A=1]$
3. **Balance for the negative class**: $\mathbb{E}[S \mid Y=0, A=0] = \mathbb{E}[S \mid Y=0, A=1]$

Conditions 2 and 3 are score-based analogs of equal opportunity and equal FPR, respectively.

## Intuition

Calibration means the score faithfully reflects true risk within each group. Balance means the average score assigned to truly positive (or negative) individuals is the same across groups. When one group has a higher base rate, its truly positive individuals are "easier" to identify (higher prior), so a calibrated score will naturally assign them higher average scores—violating balance.

## Proof Sketch

By calibration: $\mathbb{E}[Y \mid S=s, A=a] = s$. Therefore:

$$\mathbb{E}[S \mid Y=1, A=a] = \int s \cdot P(S=s \mid Y=1, A=a) \, ds$$

Using Bayes' theorem:

$$P(S=s \mid Y=1, A=a) = \frac{P(Y=1 \mid S=s, A=a) P(S=s \mid A=a)}{P(Y=1 \mid A=a)} = \frac{s \cdot P(S=s \mid A=a)}{\pi_a}$$

Thus:

$$\mathbb{E}[S \mid Y=1, A=a] = \frac{1}{\pi_a} \int s^2 \cdot P(S=s \mid A=a) \, ds = \frac{\mathbb{E}[S^2 \mid A=a]}{\pi_a}$$

For balance: $\mathbb{E}[S \mid Y=1, A=0] = \mathbb{E}[S \mid Y=1, A=1]$ requires $\frac{\mathbb{E}[S^2 \mid A=0]}{\pi_0} = \frac{\mathbb{E}[S^2 \mid A=1]}{\pi_1}$. Combined with the negative-class balance condition and calibration, this forces $\pi_0 = \pi_1$. $\square$

```python
import numpy as np

def verify_kmr(pi_0: float, pi_1: float, n: int = 100000):
    """
    Verify KMR impossibility numerically with calibrated scores.
    """
    np.random.seed(42)
    
    results = {}
    for name, pi in [('Group 0', pi_0), ('Group 1', pi_1)]:
        # Generate calibrated scores: S ~ Beta distribution
        # with E[S] = pi (so calibration approximately holds)
        alpha = pi * 5
        beta = (1 - pi) * 5
        S = np.random.beta(alpha, beta, n)
        Y = np.random.binomial(1, S)  # Y|S ~ Bernoulli(S) → calibrated
        
        avg_score_pos = S[Y == 1].mean()
        avg_score_neg = S[Y == 0].mean()
        
        results[name] = {
            'base_rate': Y.mean(),
            'E[S|Y=1]': avg_score_pos,
            'E[S|Y=0]': avg_score_neg,
        }
    
    print("KMR Impossibility — Numerical Verification")
    print("=" * 55)
    for name, r in results.items():
        print(f"\n{name} (π ≈ {r['base_rate']:.3f}):")
        print(f"  E[S | Y=1] = {r['E[S|Y=1]']:.4f}")
        print(f"  E[S | Y=0] = {r['E[S|Y=0]']:.4f}")
    
    gap_pos = abs(results['Group 0']['E[S|Y=1]'] - results['Group 1']['E[S|Y=1]'])
    gap_neg = abs(results['Group 0']['E[S|Y=0]'] - results['Group 1']['E[S|Y=0]'])
    print(f"\nBalance gaps (with calibrated scores):")
    print(f"  Positive class gap: {gap_pos:.4f}")
    print(f"  Negative class gap: {gap_neg:.4f}")
    print(f"  → Balance is violated when base rates differ.")

verify_kmr(pi_0=0.6, pi_1=0.3)
```

## Relationship to Chouldechova

| Aspect | Chouldechova | KMR |
|--------|-------------|-----|
| Score type | Binary $\hat{Y}$ | Continuous $S \in [0,1]$ |
| Fairness criteria | Calibration + equal FPR/FNR | Calibration + balance |
| Conclusion | Same: incompatible when $\pi_0 \neq \pi_1$ | Same |
| Emphasis | Error rates | Expected scores |

Both theorems arrive at the same fundamental conclusion from different angles: **calibration and error rate equalization are incompatible when base rates differ**.

## Practical Implications

1. **Score design**: In credit scoring, a calibrated risk model will assign different average scores to groups with different default rates—this is mathematically unavoidable
2. **Regulatory tension**: Regulators may demand both calibration (for pricing accuracy) and equal treatment (for fairness), creating an inherent conflict
3. **Informed tradeoffs**: Practitioners must choose which criterion to prioritize and document the rationale

## Summary

- KMR proves that calibration, positive-class balance, and negative-class balance cannot coexist with unequal base rates
- Complements Chouldechova by analyzing continuous scores rather than binary predictions
- Both results force practitioners to make **explicit choices** among competing fairness criteria

## Next Steps

- [Tradeoff Analysis](tradeoffs.md): Quantifying the Pareto frontier of fairness–accuracy tradeoffs
