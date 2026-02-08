# 33.6.3 Statistical Testing

## Why Statistical Tests Matter

RL experiments have high variance. Without proper statistical testing, apparent differences between algorithms may be due to random seed selection rather than genuine algorithmic improvements. Reliable conclusions require hypothesis tests and confidence intervals.

## Recommended Tests

### 1. Welch's t-test (Two Algorithms)
For comparing mean performance of two algorithms across seeds:
- **Null hypothesis**: $H_0: \mu_A = \mu_B$
- **Assumes**: Normal distribution (approximately holds by CLT for aggregated returns)
- **Does not assume**: Equal variance (Welch's correction)
- **Use**: $p < 0.05$ to reject null

### 2. Mann-Whitney U Test (Non-parametric)
When normality is questionable:
- Tests whether one distribution stochastically dominates the other
- No distributional assumptions
- Robust to outliers

### 3. Bootstrap Confidence Intervals
Model-free approach to uncertainty quantification:
1. Resample returns with replacement ($B = 10{,}000$ times)
2. Compute statistic (mean, median) for each bootstrap sample
3. Report 2.5th and 97.5th percentiles as 95% CI
4. If CIs don't overlap, difference is significant

### 4. Stratified Bootstrap (Recommended)
For comparing two algorithms:
1. Compute paired differences: $\delta_i = R^A_i - R^B_i$ per seed
2. Bootstrap the differences
3. Check if 95% CI excludes zero

## Multiple Comparisons

When comparing $k > 2$ algorithms:
- **Bonferroni correction**: $p_\text{adj} = p \times k(k-1)/2$
- **Holm-Bonferroni**: Less conservative step-down procedure
- **False Discovery Rate (FDR)**: Benjamini-Hochberg for many comparisons

## Effect Size

Statistical significance alone is insufficient. Report effect size:
- **Cohen's d**: $d = \frac{\mu_A - \mu_B}{\sigma_\text{pooled}}$; small (0.2), medium (0.5), large (0.8)
- **Relative improvement**: $\frac{\mu_A - \mu_B}{\mu_B} \times 100\%$

## Reporting Guidelines

1. Report mean ± standard error (not standard deviation) for uncertainty
2. Include number of seeds, episodes per evaluation, and environment version
3. Plot learning curves with shaded confidence regions
4. Report both statistical significance ($p$-value) and practical significance (effect size)
5. Acknowledge when differences are not statistically significant
