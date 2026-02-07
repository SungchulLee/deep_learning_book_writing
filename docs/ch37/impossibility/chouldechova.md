# Chouldechova's Theorem

## Overview

Chouldechova's Theorem (2017) proves that when base rates differ across groups, a classifier cannot simultaneously achieve calibration *and* equal false positive/false negative rates. This result formalized the mathematical core of the COMPAS recidivism debate.

## Statement of the Theorem

**Theorem (Chouldechova, 2017).** If the base rates differ, $P(Y=1 \mid A=0) \neq P(Y=1 \mid A=1)$, then the following three conditions **cannot** hold simultaneously (except in degenerate cases):

1. **Calibration**: $P(Y=1 \mid S=s, A=0) = P(Y=1 \mid S=s, A=1) \;\; \forall s$
2. **Equal FPR**: $P(\hat{Y}=1 \mid Y=0, A=0) = P(\hat{Y}=1 \mid Y=0, A=1)$
3. **Equal FNR**: $P(\hat{Y}=0 \mid Y=1, A=0) = P(\hat{Y}=0 \mid Y=1, A=1)$

## Proof Sketch

Define for each group $a$: base rate $\pi_a = P(Y=1 \mid A=a)$. By Bayes' theorem, the positive predictive value is:

$$\text{PPV}_a = \frac{\pi_a (1 - \text{FNR}_a)}{\pi_a (1 - \text{FNR}_a) + (1 - \pi_a) \text{FPR}_a}$$

If $\text{FPR}_0 = \text{FPR}_1$ and $\text{FNR}_0 = \text{FNR}_1$ but $\pi_0 \neq \pi_1$, substituting yields $\text{PPV}_0 \neq \text{PPV}_1$, violating calibration. $\square$

## Numerical Verification

```python
import numpy as np
from typing import Dict

def verify_chouldechova(
    pi_0: float, pi_1: float, fpr: float, fnr: float,
) -> Dict[str, float]:
    """
    Verify Chouldechova's theorem numerically.
    
    Given equal FPR and FNR but different base rates,
    compute the resulting PPV for each group.
    """
    tpr = 1 - fnr
    ppv_0 = (pi_0 * tpr) / (pi_0 * tpr + (1 - pi_0) * fpr)
    ppv_1 = (pi_1 * tpr) / (pi_1 * tpr + (1 - pi_1) * fpr)
    
    return {
        'ppv_0': ppv_0, 'ppv_1': ppv_1,
        'ppv_gap': abs(ppv_0 - ppv_1),
        'calibration_violated': abs(ppv_0 - ppv_1) > 1e-10,
    }

# Different base rates → calibration violated
print("Case 1: π₀=0.6, π₁=0.3, equal FPR=0.15, FNR=0.20")
r = verify_chouldechova(0.6, 0.3, 0.15, 0.20)
print(f"  PPV₀ = {r['ppv_0']:.4f}, PPV₁ = {r['ppv_1']:.4f}, gap = {r['ppv_gap']:.4f}")
print(f"  Calibration violated: {r['calibration_violated']}")

# Equal base rates → no conflict
print("\nCase 2: π₀=0.5, π₁=0.5, equal FPR=0.15, FNR=0.20")
r = verify_chouldechova(0.5, 0.5, 0.15, 0.20)
print(f"  PPV₀ = {r['ppv_0']:.4f}, PPV₁ = {r['ppv_1']:.4f}, gap = {r['ppv_gap']:.4f}")
print(f"  Calibration violated: {r['calibration_violated']}")

# Sweep base rate difference
print("\nSweep: π₀=0.5, FPR=0.15, FNR=0.20, varying π₁")
for pi_1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
    r = verify_chouldechova(0.5, pi_1, 0.15, 0.20)
    print(f"  π₁={pi_1:.1f}  →  PPV gap = {r['ppv_gap']:.4f}")
```

## Implications

1. **The COMPAS debate was inevitable**: ProPublica measured error rates; Northpointe measured calibration. Both were right about their own metrics—but the metrics cannot simultaneously be equalized.
2. **Practitioners must choose**: When base rates differ, you must decide which fairness criterion takes priority.
3. **No "unfair" party**: The impossibility is mathematical, not a failure of either side.

## Summary

- When base rates differ, **calibration and equalized odds are incompatible**
- The conflict is **algebraic**, not an artifact of imperfect models
- Forces explicit choices about which fairness criterion to prioritize

## Next Steps

- [KMR Impossibility](kmr.md): A complementary impossibility result
- [Tradeoff Analysis](tradeoffs.md): Quantifying the cost of choosing one criterion over another
