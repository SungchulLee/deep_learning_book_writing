# 35.5.4 Overfitting

## Learning Objectives

- Recognize and prevent overfitting in financial RL
- Implement cross-validation strategies for time-series data
- Design regularization techniques specific to financial RL
- Distinguish between true alpha and data-mined patterns

## Introduction

Overfitting is the primary failure mode of financial RL. With enough parameters and training time, an RL agent can learn to perfectly trade the training data by memorizing specific market patterns that do not generalize. The result: spectacular backtesting performance that collapses in live trading.

## Sources of Overfitting in Financial RL

| Source | Description | Mitigation |
|--------|-------------|-----------|
| Look-ahead bias | Using future information in state | Strict temporal ordering |
| Survivorship bias | Only using assets that survived | Include delisted assets |
| Selection bias | Cherry-picking training periods | Random period sampling |
| Multiple testing | Testing many strategies | Bonferroni correction |
| Complexity | Too many parameters | Regularization, pruning |
| Training length | Over-training on fixed data | Early stopping |

## Time-Series Cross-Validation

Standard k-fold CV is invalid for time series (leaks future information). Use temporal splits:

### Walk-Forward Validation

```
Train: [------]
Test:         [--]
     Train: [--------]
     Test:           [--]
          Train: [----------]
          Test:              [--]
```

### Purged Cross-Validation

Add a gap between train and test to prevent information leakage from overlapping windows.

### Combinatorial Purged Cross-Validation (CPCV)

Tests all possible train/test split combinations, providing a distribution of out-of-sample performance.

## Regularization Techniques

### 1. Dropout and Weight Decay

Standard neural network regularization.

### 2. Action Smoothing

Penalize large changes in position: $\mathcal{L}_{\text{smooth}} = \lambda \|\mathbf{w}_t - \mathbf{w}_{t-1}\|^2$

### 3. Entropy Regularization

Encourage exploration and prevent premature convergence: $\mathcal{L} = \mathcal{L}_{\text{policy}} - \beta \mathcal{H}[\pi]$

### 4. Noise Injection

Add noise to states, actions, or rewards during training.

### 5. Deflated Sharpe Ratio

Adjust the Sharpe ratio for multiple testing:

$$\text{DSR} = \text{SR} - \sqrt{\frac{V[\hat{\text{SR}}]}{1}} \cdot \Phi^{-1}\left(1 - \frac{1}{N_{\text{trials}}}\right)$$

## Statistical Tests for Overfitting

- **Probability of Backtest Overfitting (PBO)**: Probability that the best in-sample strategy underperforms out-of-sample
- **Minimum Backtest Length**: Minimum data needed to achieve a given confidence level
- **Deflated Sharpe Ratio**: Adjusts for selection bias in strategy development

## Summary

Overfitting prevention requires disciplined methodology: strict temporal validation, regularization at multiple levels, and statistical tests for strategy significance. The key principle is that out-of-sample performance is the only reliable measure.

## References

- Bailey, D.H. & López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management.
- Bailey, D.H., et al. (2017). The Probability of Backtest Overfitting. Journal of Computational Finance.
- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
