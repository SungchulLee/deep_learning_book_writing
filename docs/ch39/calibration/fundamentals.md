# Calibration Fundamentals

## Overview

A model is **calibrated** if its predicted probabilities reflect true outcome frequencies. Among all predictions where a model states "80% confidence," the model should be correct approximately 80% of the time. Calibration is orthogonal to accuracy—a model can be accurate but poorly calibrated (overconfident), or well-calibrated but inaccurate.

## Mathematical Definition

**Perfect calibration**:

$$\mathbb{P}(Y = y \mid \hat{p}(Y = y) = p) = p, \quad \forall p \in [0, 1]$$

## Why Calibration Matters in Finance

| Application | Without Calibration | With Calibration |
|-------------|---------------------|-----------------|
| VaR estimation | Understated tail risk | Accurate risk bounds |
| Position sizing | Overconfident sizing | Risk-adjusted allocation |
| Credit scoring | Regulatory violations | Compliant probability estimates |
| Option pricing | Mispriced derivatives | Consistent implied probabilities |

## The Calibration Gap

Modern neural networks are systematically **overconfident**: predicted confidence exceeds actual accuracy. This gap grows with model depth and capacity. Post-hoc calibration methods (temperature scaling, Platt scaling, isotonic regression) address this without retraining.

## Calibration Pipeline

```python
# Standard calibration workflow
# 1. Train model on training set
# 2. Extract logits on held-out validation set
# 3. Fit calibration method (e.g., temperature) on validation logits
# 4. Apply calibrated model to test set
# 5. Evaluate with calibration metrics (ECE, reliability diagram)
```

The following sections cover specific calibration methods (temperature scaling, isotonic regression, Platt scaling, focal loss) and calibration metrics in detail.

## References

- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
- Naeini, M. P., et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." AAAI.
