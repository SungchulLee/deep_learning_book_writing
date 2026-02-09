# 35.6.4 Statistical Significance

## Learning Objectives

- Test whether RL strategy performance is statistically significant
- Implement bootstrap hypothesis tests for Sharpe ratios
- Apply multiple testing corrections for strategy selection
- Compute confidence intervals for performance metrics

## Introduction

A backtest showing a Sharpe ratio of 1.5 means little without statistical context. Was this result due to skill or luck? Statistical significance tests help answer this question by computing the probability of observing the results by chance.

## Hypothesis Testing

**Null hypothesis**: The strategy has zero expected return (no skill).

**Test statistic**: Sharpe ratio or t-statistic of mean returns.

## Tests for Sharpe Ratio

### t-Test for Mean Return

The t-statistic is related to the Sharpe ratio:

$$t = \text{SR} \times \sqrt{T}$$

For a Sharpe ratio of 1.0 over 252 days: $t = 1.0 \times \sqrt{252} \approx 15.9$ — clearly significant. But for daily Sharpe 0.03, $t = 0.03 \times \sqrt{252} \approx 0.48$ — not significant.

### Bootstrap Test

1. Resample returns (with replacement) B times
2. Compute Sharpe ratio for each bootstrap sample
3. The p-value is the fraction of bootstrap samples with Sharpe <= 0

### Permutation Test

Randomly shuffle the strategy's position timing relative to returns. If the strategy has genuine timing skill, shuffled versions should perform worse.

## Multiple Testing Corrections

When comparing $M$ strategies, the probability of finding at least one significant result by chance increases dramatically.

| Correction | Adjusted p-value | Conservative? |
|-----------|-----------------|--------------|
| Bonferroni | $p_i \times M$ | Very |
| Holm-Bonferroni | Step-down procedure | Moderate |
| BHY (FDR) | Controls false discovery rate | Less |

## Minimum Backtest Length

The minimum number of observations needed for a given Sharpe ratio to be significant:

$$T_{\min} \approx \left(\frac{z_\alpha}{\text{SR}}\right)^2$$

For SR=0.5 and 95% confidence: $T_{\min} \approx (1.96/0.5)^2 \approx 15.4$ years of daily data.

## Summary

Statistical significance testing prevents deploying strategies that appear profitable due to chance. Bootstrap and permutation tests provide distribution-free alternatives to parametric tests. Multiple testing corrections are essential when selecting among many candidate strategies.

## References

- Lo, A. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal.
- Bailey, D.H. & López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management.
- Harvey, C., Liu, Y., & Zhu, H. (2016). ...and the Cross-Section of Expected Returns. Review of Financial Studies.
