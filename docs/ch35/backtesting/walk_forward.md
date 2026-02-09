# 35.6.2 Walk-Forward Analysis

## Learning Objectives

- Implement walk-forward analysis for RL strategy evaluation
- Design expanding and rolling window training procedures
- Handle model retraining schedules and parameter stability
- Combine walk-forward results into aggregate performance estimates

## Introduction

Walk-forward analysis is the gold standard for evaluating time-series strategies. The data is divided into sequential train-test blocks. The model is trained on each training block and evaluated on the subsequent test block.

## Walk-Forward Variants

### Anchored (Expanding Window)

Training window grows over time, retaining all historical data.

### Rolling Window

Fixed-size training window slides forward, adapting to recent conditions.

### Purged Walk-Forward

Add a gap (embargo period) between train and test to prevent information leakage from overlapping features.

## Retraining Schedule

| Frequency | Pros | Cons |
|-----------|------|------|
| Daily | Maximum adaptation | High computational cost |
| Weekly | Good balance | Moderate cost |
| Monthly | Low cost | Slower adaptation |
| Triggered | Adapts when needed | Complex logic |

## Summary

Walk-forward analysis provides the most realistic assessment of strategy performance by simulating actual deployment of periodic retraining and out-of-sample evaluation.

## References

- Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies. Wiley.
- Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management.
