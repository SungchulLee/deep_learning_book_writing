# Reconstruction Quality Analysis for Autoencoders

## Introduction

Assessing and analyzing reconstruction quality is fundamental to understanding autoencoder performance and identifying when the learned representation adequately captures input structure. Reconstruction error—the difference between input and autoencoder output—provides a primary quality metric, but meaningful evaluation requires multiple perspectives: aggregate error statistics, per-sample error distributions, feature-wise reconstruction patterns, and task-specific performance metrics.

In financial applications, reconstruction quality analysis reveals whether autoencoders successfully compress market microstructure (high-frequency trading), capture systematic risk factors (equity returns), or preserve essential information for downstream tasks like anomaly detection or portfolio optimization. Poor reconstruction may indicate inadequate bottleneck capacity, inappropriate architecture, or presence of non-stationary patterns that challenge the autoencoder's generalization capability.

This section develops comprehensive frameworks for reconstruction quality analysis, establishes baseline comparisons, and demonstrates how to diagnose autoencoder deficiencies through detailed error investigation.

## Key Concepts

### Reconstruction Metrics
- **Mean Squared Error (MSE)**: Average squared difference between input and output
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Relative Error**: Normalized by input variance or magnitude
- **Correlation**: Preservation of statistical relationships in reconstruction

### Error Analysis Dimensions
- **Temporal Patterns**: How reconstruction error varies over time
- **Feature Analysis**: Which inputs/features have largest reconstruction errors
- **Distributional Properties**: Shape of error distribution (normal vs heavy-tailed)
- **Stability**: How error varies across different data regimes

## Mathematical Framework

### Reconstruction Error Metrics

For input $x \in \mathbb{R}^d$ and reconstruction $\hat{x} = f_{\text{dec}}(f_{\text{enc}}(x))$:

$$\text{MSE} = \frac{1}{d}\sum_{i=1}^{d} (x_i - \hat{x}_i)^2$$

$$\text{MAE} = \frac{1}{d}\sum_{i=1}^{d} |x_i - \hat{x}_i|$$

$$\text{MAPE} = \frac{1}{d}\sum_{i=1}^{d} \frac{|x_i - \hat{x}_i|}{|x_i| + \epsilon}$$

For non-negative inputs, MAPE provides scale-invariant error independent of magnitude.

### Relative Error Normalization

Normalize reconstruction error by input variance:

$$\text{RelError} = \frac{\text{MSE}}{\text{Var}(x)}$$

Interpretation: error as percentage of input variability. Value < 0.1 indicates excellent reconstruction (10% of variance unexplained).

### Correlation-Based Quality

Measure preservation of statistical relationships:

$$\rho = \text{Corr}(x, \hat{x}) = \frac{\text{Cov}(x, \hat{x})}{\sqrt{\text{Var}(x)\text{Var}(\hat{x})}}$$

High correlation (ρ > 0.95) indicates structure preservation despite some noise.

## Per-Sample Error Analysis

### Error Distribution Characterization

Examine empirical distribution of reconstruction errors:

$$\mathcal{E} = \{e_n : e_n = \|x_n - \hat{x}_n\|^2, n = 1,\ldots,N\}$$

Compute quantiles and moments:

$$\text{Mean Error} = \mathbb{E}[e_n]$$
$$\text{Std Error} = \sqrt{\text{Var}(e_n)}$$
$$\text{Skewness} = \frac{\mathbb{E}[(e_n - \mu)^3]}{\sigma^3}$$

Heavy right tail (positive skewness) indicates occasional large errors.

### Percentile Analysis

Examine error concentration:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Error_{90\%} | 90th percentile | 90% of samples below this error |
| Error_{99\%} | 99th percentile | Worst 1% of samples |
| Error_{max} | Maximum error | Largest individual reconstruction error |

Financial data often exhibits high 99th percentile errors during market stress.

### Anomalous Sample Identification

Samples with reconstruction error beyond threshold indicate either:

1. **Outliers/Anomalies**: Unusual market conditions not well represented in training
2. **Capacity Deficiency**: Autoencoder architecture insufficient for full data complexity
3. **Non-Stationarity**: Temporal shift in data distribution since training

Investigate high-error samples separately; they often coincide with market events, regime changes, or data quality issues.

## Feature-Wise Reconstruction Analysis

### Per-Feature Error Breakdown

For multivariate input with d features:

$$\text{MSE}_j = \frac{1}{n}\sum_{i=1}^{n} (x_{ij} - \hat{x}_{ij})^2$$

Identify features with consistently high reconstruction error:

- **Easy Features**: Low error, well captured by bottleneck
- **Hard Features**: High error, potentially noisy or nonlinear
- **Asymmetric Error**: Bias toward over/under estimation

### Correlation-Adjusted Error

For feature j with variance $\sigma_j^2$:

$$\text{RelError}_j = \frac{\text{MSE}_j}{\sigma_j^2}$$

Reveals which features lose most information (relative to their inherent variability).

### Reconstruction Bias

Check for systematic bias (not random noise):

$$\text{Bias}_j = \frac{1}{n}\sum_{i=1}^{n} (\hat{x}_{ij} - x_{ij})$$

Non-zero bias indicates autoencoder systematically over/under estimates feature.

## Temporal Analysis of Reconstruction Error

### Time-Series Error Patterns

For financial time series with T observations:

$$\text{Error}_t = \|x_t - \hat{x}_t\|^2$$

Track error evolution to identify regime-dependent reconstruction quality:

- **Stable Periods**: Error remains low and constant
- **Transition Periods**: Error spikes during regime changes
- **Stress Events**: Error elevated during market anomalies

### Rolling Window Error Statistics

Compute error statistics in sliding windows:

$$\text{Error}_{[t-W, t]} = \text{Mean}(\text{Error}_\tau), \quad \tau \in [t-W, t]$$

Reveals whether autoencoder reconstruction degrades over time (non-stationarity) or maintains consistent quality.

### Autocorrelation of Error

High autocorrelation in reconstruction error suggests:

$$\rho_k = \text{Corr}(\text{Error}_t, \text{Error}_{t+k})$$

- **ρ_k > 0.5**: Strong temporal clustering of errors
- **Interpretation**: Some types of price movements harder to reconstruct than others

## Task-Specific Reconstruction Quality

### Anomaly Detection Metrics

For anomaly detection application, evaluate reconstruction error's discrimination:

$$\text{AUC} = \frac{\# \text{(normal samples with error < threshold)}}{N_{\text{normal}}}$$

Threshold chosen such that false positive rate acceptable for application.

### Dimensionality Reduction Quality

For downstream clustering/classification, assess preservation of distances:

$$\text{Correlation of Distances} = \text{Corr}(\|x_i - x_j\|, \|\hat{x}_i - \hat{x}_j\|)$$

High correlation indicates reconstruction preserves neighborhood structure.

### Factor Extraction Fidelity

For learning market factors via autoencoder, analyze whether factor structure preserved:

$$\text{PCA Alignment} = \text{Corr}(\text{PC}_{\text{original}}, \text{PC}_{\text{reconstructed}})$$

Measures whether principal modes of variation reconstructed accurately.

## Diagnostic Tools

### Residual Analysis

Examine residuals $r = x - \hat{x}$:

1. **Residual Mean**: Should be near zero (indicates unbiased reconstruction)
2. **Residual Variance**: Should be white noise (unpredictable from inputs)
3. **Residual Autocorrelation**: Should decay quickly (temporal independence)
4. **Residual Q-Q Plot**: Compare to standard normal distribution

Systematic residual patterns indicate missing structure in learned representation.

### Error Visualization

Create heatmaps of reconstruction error:

1. **Feature × Sample Matrix**: Shows error patterns across inputs and samples
2. **Error by Time and Feature**: Temporal dynamics of per-feature error
3. **Error Distribution Histogram**: Empirical error distribution with fitted model

Visual inspection often reveals error patterns missed by summary statistics.

### Comparison to Baselines

Compare reconstruction error to baseline methods:

| Method | MSE | RelError | Advantages |
|--------|-----|----------|------------|
| Mean Imputation | σ² | 1.0 | Simplest baseline |
| PCA (K factors) | 0.3σ² | 0.3 | Linear compression |
| Autoencoder | 0.15σ² | 0.15 | **Nonlinear + optimal |

Autoencoder should outperform baselines; if not, investigate architecture or training.

## Practical Recommendations

### Reconstruction Quality Targets

For financial applications:

- **MSE < 0.1 × Var(x)**: Excellent reconstruction
- **MSE ∈ [0.1, 0.3] × Var(x)**: Good reconstruction, acceptable for most tasks
- **MSE > 0.3 × Var(x)**: Poor reconstruction, review architecture

### Diagnostic Procedure

When reconstruction quality inadequate:

1. **Check Data Quality**: Verify training data contains expected variation
2. **Validate Train/Test Split**: Ensure test error not inflated by distribution mismatch
3. **Increase Bottleneck**: Expand latent dimension to improve capacity
4. **Deepen Network**: Add layers to capture nonlinear relationships
5. **Regularization Tuning**: Reduce weight decay if overly constraining

!!! warning "Overfitting Considerations"
    Autoencoders can overfit to training data, achieving perfect reconstruction while failing to generalize. Always evaluate on held-out test set; large train-test gap indicates overfitting requiring regularization increase.

