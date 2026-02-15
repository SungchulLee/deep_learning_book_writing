# Statistical Inference for Model Evaluation

## 1. Introduction: The Difference Between Signal and Noise

When evaluating a deep learning model for quantitative trading, a seemingly impressive 2% improvement in Sharpe ratio could be genuine alpha or merely statistical noise. Statistical inference provides the rigorous framework to distinguish between real improvements and random fluctuations.

!!! warning "The Multiplicity Problem"
    Testing dozens of model architectures against historical data virtually guarantees finding apparent improvements by chance alone. This is data snooping—the silent killer of quantitative strategies.

## 2. Estimator Properties

Before comparing models, we need properties that guarantee our estimates are trustworthy:

**Bias**: An estimator $\hat{\theta}$ is unbiased if $E[\hat{\theta}] = \theta$
$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

**Variance**: Measures estimator stability across different samples
$$\text{Var}(\hat{\theta}) = E[(\hat{\theta} - E[\hat{\theta}])^2]$$

**Mean Squared Error**: Combines bias and variance
$$\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$$

**Consistency**: An estimator is consistent if $\hat{\theta} \xrightarrow{p} \theta$ as $n \to \infty$

For the sample mean $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$:
- Unbiased: $E[\bar{X}_n] = \mu$
- Variance: $\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}$ (decreases with sample size)
- Consistent: By law of large numbers

Sample variance $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is unbiased but has higher variance.

## 3. Confidence Intervals

A 95% confidence interval for a model metric provides a range where the true parameter lies with 95% probability.

**For Accuracy or Average Return:**
Using the normal approximation (for large $n$):
$$\hat{p} \pm z_{1-\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**For Sharpe Ratio:**
The Sharpe ratio $\hat{S} = \frac{\bar{R} - r_f}{\hat{\sigma}_R}$ has non-normal sampling distribution.

!!! tip "Bootstrap Confidence Intervals"
    For non-standard metrics (maximum drawdown, Calmar ratio), bootstrap provides distribution-free CIs:
    1. Resample returns with replacement 10,000 times
    2. Compute metric on each bootstrap sample
    3. Use empirical 2.5% and 97.5% quantiles as CI bounds

## 4. Hypothesis Testing

**Comparing Two Models (Paired t-test):**

Given paired performance metrics $(Y_{1i}, Y_{2i})$ for each test period $i$:
$$t = \frac{\bar{D}}{\frac{S_D}{\sqrt{n}}} \sim t_{n-1}$$

where $D_i = Y_{1i} - Y_{2i}$ is the performance difference.

**Null Hypothesis**: $H_0: \mu_D = 0$ (no difference in performance)

**p-value**: Probability of observing this (or more extreme) test statistic under $H_0$

!!! danger "Multiple Comparison Problem"
    If you test 100 models against a benchmark, expect ~5 false positives at $\alpha=0.05$ significance level. This is why corrections are essential.

## 5. Multiple Testing Correction

**Bonferroni Correction:**

For $m$ independent tests, control family-wise error rate by adjusting individual significance level:
$$\alpha_{adjusted} = \frac{\alpha}{m}$$

Very conservative—rejects many true signals. For $m=100$ tests: $\alpha_{adjusted} = 0.0005$

**False Discovery Rate (FDR) Control:**

Benjamini-Hochberg procedure is less conservative:
1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest $i$ where $p_{(i)} \leq \frac{i}{m} \alpha$
3. Reject all hypotheses $1, ..., i$

FDR limits the proportion of false positives among rejections (typically 10% in quant).

## 6. Bootstrap Methods

**Non-parametric Bootstrap Algorithm:**

```
for b = 1 to B:
    Sample with replacement from original data: X*_b
    Compute estimator: θ̂*_b = f(X*_b)
end
Estimated variance: Var(θ̂) ≈ Var(θ̂*_1, ..., θ̂*_B)
```

**Advantages:**
- No distributional assumptions
- Works for any metric (Sharpe, max drawdown, etc.)
- Estimates standard errors and confidence intervals directly

**Example**: Testing if strategy Sharpe ratio > 1.0
- Bootstrap resample returns 10,000 times
- Compute Sharpe ratio each time
- If 95% CI excludes 1.0, reject $H_0: S \leq 1.0$

## 7. Cross-Validation as Statistical Procedure

K-fold cross-validation estimates generalization error $\mathcal{L}$:
$$\widehat{\text{CV}} = \frac{1}{k} \sum_{i=1}^k L_i$$

where $L_i$ is loss on $i$-th holdout fold.

**Key Insight**: CV estimates are not independent. Folds share training data, creating correlation.

**Variance of CV Estimator:**
$$\text{Var}(\widehat{\text{CV}}) = \frac{\sigma^2_{CV}}{k} + \text{correlation effect}$$

Standard error typically 10-30% higher than if folds were independent. Always use this larger estimate when constructing CIs!

!!! warning "Time Series CV"
    Standard k-fold inappropriate for time series. Use forward-chaining validation:
    - Train on years 1-5, test on year 6
    - Train on years 1-6, test on year 7
    - No look-ahead bias, no data leakage

## 8. Practical Guidelines for Quant Practitioners

**Avoid Data Snooping:**
- Specify strategy *before* looking at data
- Use out-of-sample testing with proper time-series CV
- Report all backtests, not just winners
- Use Bonferroni or FDR correction when comparing multiple strategies

**Rigorous Performance Testing:**

1. **Point Estimate**: Report Sharpe ratio, max drawdown, Calmar ratio
2. **Uncertainty**: Bootstrap 95% CI around each metric
3. **Hypothesis Test**: Test if Sharpe > 1.0 at 5% significance
4. **Multiple Correction**: If testing multiple variants, apply FDR
5. **Walk-Forward**: Test on recent data not in training set

**Statistical Power:**

To reliably detect strategy with true Sharpe = 1.2 vs. benchmark 1.0, need roughly 250 daily observations (1 year). With 10 years of data, power exceeds 99%.

!!! success "Best Practice"
    Implement trading strategy on paper for 3+ months before live trading. Treat paper results as *new* data for hypothesis testing. Only deploy if significance tests pass on this fresh data.

## References

- Efron & Tibshirani (1993): *An Introduction to the Bootstrap*
- White (2000): "A Reality Check for Data Snooping" (*Econometric Reviews*)
- Benjamini & Hochberg (1995): "Controlling the False Discovery Rate" (*JASA*)
- De Prado (2018): *Advances in Financial Machine Learning*, Chapters 4-6
