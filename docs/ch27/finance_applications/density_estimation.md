# Density Estimation Using EBMs for Financial Applications

## Introduction

Accurate density estimation of financial return distributions is fundamental to quantitative finance, enabling pricing, risk management, and portfolio optimization. Financial returns exhibit non-Gaussian characteristics including heavy tails, skewness, and kurtosis that standard parametric models fail to capture. Energy-Based Models provide a flexible, non-parametric approach to learning complex return distributions directly from data without imposing restrictive parametric assumptions.

Unlike kernel density estimation (KDE) that suffers curse of dimensionality, or mixture models requiring manual specification of component count, EBMs learn flexible densities through neural network parameterization. The ability to evaluate the learned density at arbitrary points enables applications ranging from quantile estimation and Value-at-Risk calculation to derivatives pricing and portfolio optimization.

This section develops EBM-based density estimation methods for financial applications, addresses computational and statistical challenges in high dimensions, and demonstrates practical implementations for risk measurement.

## Key Concepts

### Density Estimation Objectives
- **Characterize Return Distribution**: Learn p(r) where r are financial returns
- **Capture Non-Gaussianity**: Account for heavy tails, skewness, excess kurtosis
- **Tail Risk Estimation**: Accurate extreme quantiles for VaR and Expected Shortfall
- **Conditional Densities**: Model p(r|x) where x are conditioning variables

### Energy-Based Density Framework
- **Energy Function**: E(r; θ) quantifies distribution plausibility
- **Probabilistic Model**: $p(r) = \exp(-E(r; \theta)) / Z$
- **Flexibility**: Neural network enables arbitrary density shapes
- **Sample Generation**: Langevin dynamics generate samples from learned p(r)

## Mathematical Framework

### Energy-Based Density Estimation

Learn density p(r) by fitting energy function E(r; θ):

$$p(r; \theta) = \frac{\exp(-E(r; \theta))}{Z(\theta)}$$

Partition function (often intractable):

$$Z(\theta) = \int_{\mathbb{R}^d} \exp(-E(r; \theta)) dr$$

Training maximizes likelihood on observed returns R = {r₁, ..., r_N}.

### Contrastive Divergence Training

Approximate maximum likelihood through contrastive divergence:

$$\nabla_\theta \log p(r; \theta) \approx \mathbb{E}_{r \sim p_{\text{data}}}[\nabla_\theta E(r)] - \mathbb{E}_{r \sim p_\theta}[\nabla_\theta E(r)]$$

First term (positive phase) pulls energy down on observed data. Second term (negative phase) pushes energy up on model-sampled returns, requiring Langevin sampling.

### Langevin Sampling for Density Model

Generate samples from learned density using Langevin dynamics:

$$r^{(t+1)} = r^{(t)} - \frac{\eta}{2}\nabla_r E(r^{(t)}) + \sqrt{\eta} \xi_t$$

where η is step size and ξ_t ~ N(0,I) is noise. As t→∞, r^{(t)} samples from p_θ.

## Density Components and Structure

### Marginal Density Learning

For univariate returns, learn marginal density directly:

$$p(r) = \frac{\exp(-E(r))}{Z}$$

Neural network maps scalar return to scalar energy:

$$E(r) = f_\theta(r), \quad f_\theta: \mathbb{R} \to \mathbb{R}$$

Captures non-Gaussian shapes: heavy tails, skewness, multimodality.

### Multivariate Density Learning

For joint distribution of multiple assets {r₁, ..., r_d}:

$$p(r_1, \ldots, r_d) = \frac{\exp(-E(r_1, \ldots, r_d))}{Z}$$

Energy function encodes cross-asset dependencies through neural network with multivariate input.

### Conditional Density Estimation

Learn p(r|x) where x are conditioning variables (volatility, interest rates, etc.):

$$p(r|x) = \frac{\exp(-E(r; x, \theta))}{Z(x)}$$

Energy function depends on both r and x; enables risk forecasting conditional on market state.

## Financial Applications

### Value-at-Risk and Expected Shortfall

From learned density p(r; θ), compute risk measures:

$$\text{VaR}_\alpha = \inf\{r : \int_{-\infty}^{r} p(s; \theta) ds \geq \alpha\}$$

$$\text{ES}_\alpha = \mathbb{E}[r | r \leq \text{VaR}_\alpha] = \frac{1}{\alpha} \int_{-\infty}^{\text{VaR}_\alpha} r \cdot p(r; \theta) dr$$

More accurate tail risk measurement than parametric models when EBM captures heavy tails.

### Portfolio Return Distribution

For portfolio with weights w, return density:

$$p(r_p) = p\left(\sum_i w_i r_i\right)$$

Learn marginal distribution of portfolio return directly. Enables:

1. **Optimal Quantile Estimation**: 1%, 5%, 95%, 99% quantiles for risk limits
2. **Shortfall Analysis**: Tail expectation computation
3. **Probability Statements**: P(r_p < -5%) for risk disclosure

### Regime-Conditional Density

Model changing density across market regimes:

$$p(r | \text{regime}) = \frac{\exp(-E(r; \text{regime}))}{Z_{\text{regime}}}$$

Train separate EBMs for:
- **Normal Markets**: Low volatility regime (majority of data)
- **Elevated Volatility**: Moderate stress (10-20% of data)
- **Crisis**: Tail events (<5% of data)

Regime detection via hidden Markov model or smoothly transitions probabilities.

### Option Pricing from Risk-Neutral Density

Under risk-neutral measure, call option price:

$$C(K, T) = \int_K^\infty (S_T - K) p^Q(S_T) dS_T$$

where p^Q is risk-neutral density. Estimate p^Q from options market; compare to risk-neutral density derived from EBM trained on physical density (require risk premium specification).

## Handling Financial Data Characteristics

### Heavy Tails and Outliers

Financial returns have fat tails; standard Gaussian ill-suited. Energy function must accommodate:

$$E(r) = \sum_j \alpha_j |r|^j$$

Polynomial energy grows sublinearly in |r| enabling heavy tails. Alternative: mixture formulation:

$$p(r) = \pi_{\text{normal}} \mathcal{N}(r) + (1-\pi) \text{EBM}_{\text{tail}}(r)$$

### Skewness and Asymmetry

Returns skewed; EBM learns through asymmetric energy:

$$E(r) = (1 + \text{sign}(r) \cdot s) \cdot |r|^\gamma$$

where s ∈ [-1, 1] controls skewness. Positive s: left-skewed (negative outliers); negative s: right-skewed.

### Time-Varying Distributions

Return distribution non-stationary. Remedies:

1. **Rolling Windows**: Retrain EBM on 252-day rolling windows
2. **Time-Dependent Energy**: $E(r,t)$ explicitly models time variation
3. **GARCH Integration**: Combine EBM for standardized return shape with GARCH for volatility

## Computational Considerations

### Scalability to High Dimensions

For multivariate densities with d assets:

**Challenge**: Energy function computation O(d) per evaluation; Langevin sampling requires many evaluations.

**Solutions**:

1. **Dimensionality Reduction**: First apply PCA to reduce d to 5-10 factors; learn density in factor space
2. **Factorized Approximation**: Assume independence/correlation structure; learn conditional densities
3. **GPU Acceleration**: Parallelize Langevin sampling across multiple chains

### Numerical Stability

Neural network energy can become unbounded or saturated. Remedies:

1. **Energy Regularization**: Penalize |E(r; θ)| to keep energy in reasonable range
2. **Batch Normalization**: Normalize hidden activations to avoid extreme values
3. **Careful Initialization**: Initialize weights to ensure stable energy landscape

## Evaluation and Validation

### Goodness-of-Fit Tests

Assess whether learned density matches data:

$$K_s = \max_{r} \left| \hat{F}_{\text{empirical}}(r) - F_{\text{EBM}}(r) \right|$$

Kolmogorov-Smirnov test compares empirical and model CDFs.

### Quantile Analysis

Compare empirical quantiles to EBM-estimated quantiles:

$$\text{Error}_\alpha = |\hat{q}_\alpha^{\text{empirical}} - \hat{q}_\alpha^{\text{EBM}}|$$

Focus on extreme quantiles (1%, 5%, 95%, 99%) critical for risk management.

### Likelihood-Based Validation

On held-out test set, evaluate model log-likelihood:

$$\hat{\mathcal{L}} = \frac{1}{N_{\text{test}}} \sum_{n=1}^{N_{\text{test}}} \log p(r_n; \theta^*)$$

Higher likelihood indicates better density fit.

## Practical Implementation Guidelines

### Data Preparation

1. **Standardization**: Center and scale returns to unit variance
2. **Outlier Capping**: Clip extreme values (>4σ) to prevent gradient explosion
3. **Train/Test Split**: 80/20 split for validation; use rolling windows for time series

### Architecture Design

- **Input**: Standardized returns (real-valued scalars or vectors)
- **Hidden**: 2-3 hidden layers, 64-256 units with ELU activation
- **Output**: Single scalar energy (no activation)

### Training Details

- **Optimizer**: Adam with learning rate 1e-3 to 1e-4
- **Batch Size**: 256-512
- **Epochs**: Train until validation log-likelihood plateaus
- **Langevin Steps**: 50-100 steps per negative phase sample

!!! note "EBM Density Estimation"
    EBMs provide flexible, non-parametric density estimation suitable for complex financial distributions. Key advantages: capture non-Gaussianity, scale to moderate dimensions (d=10-100), enable direct likelihood evaluation. Limitations: computational cost (Langevin sampling), difficulty in very high dimensions (d>1000).

