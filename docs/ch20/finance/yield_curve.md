# Yield Curve Decomposition Using PCA

## Introduction

The yield curve—the relationship between bond yields and maturity—is one of the most important financial instruments, encoding market expectations about future interest rates, economic growth, and inflation. However, observing yield curve dynamics requires monitoring dozens of maturities (3 months, 6 months, 1 year, 2 years, ..., 30 years), creating a high-dimensional estimation problem.

Principal Component Analysis provides an elegant solution, revealing that yield curve movements can be parsimoniously described by just three principal components that correspond to economically interpretable factors: the level (parallel shift in all rates), the slope (change in the spread between long and short rates), and the curvature (non-parallel changes in intermediate maturities). This decomposition enables portfolio managers to characterize yield curve exposure, estimate duration and convexity, and implement efficient hedging strategies.

This section develops the theoretical and practical foundations of PCA-based yield curve analysis, demonstrates component interpretation, and explores applications to fixed income portfolio management.

## Key Concepts

### Yield Curve Characteristics
- **Level**: Overall height of curve; reflects average interest rate across maturities
- **Slope**: Difference between long and short rates; captures term premium
- **Curvature**: Humped or inverted middle section; reflects rate expectations
- **Volatility**: Standard deviation of yield changes; maturity-dependent

### Maturity Representation
- **Short End**: 3M, 6M, 1Y rates; most sensitive to policy rates
- **Intermediate**: 2Y, 5Y, 7Y rates; reflect expectations of future rates
- **Long End**: 10Y, 20Y, 30Y rates; embody inflation expectations and term premium

## Mathematical Framework

### Yield Curve as Vector

Represent yield curve at time t as vector of yields across K maturities:

$$y_t = [y_t(m_1), y_t(m_2), \ldots, y_t(m_K)]^T$$

where $y_t(m)$ is the yield at time t for maturity m. Typically K = 8-13 standard maturities.

### Change Vector and Covariance

Define yield changes:

$$\Delta y_t = y_t - y_{t-1}$$

The covariance matrix of yield changes:

$$\Sigma = \text{Cov}(\Delta y) = \frac{1}{T}\sum_{t=1}^{T} \Delta y_t \Delta y_t^T$$

captures the correlation structure across maturities.

### PCA Decomposition

Eigendecomposition yields:

$$\Sigma = V \Lambda V^T$$

Principal components represent yield curve movements:

$$\text{PC}_k(t) = V_{:,k}^T \Delta y_t$$

The yield curve evolves as:

$$\Delta y_t \approx \sum_{k=1}^{3} \text{PC}_k(t) \cdot V_{:,k}$$

### Variance Explained

Typical decomposition for U.S. Treasuries:

- **PC1 (Level)**: 80-90% of variance
- **PC2 (Slope)**: 8-12% of variance  
- **PC3 (Curvature)**: 2-4% of variance

Three components together explain 95%+ of yield curve variance.

## Principal Component Interpretation

### First Component: Level (Parallel Shift)

PC1 has approximately equal loadings across all maturities:

$$V_{:,1} \approx \frac{1}{\sqrt{K}}[1, 1, 1, \ldots, 1]^T$$

Represents parallel shift in yield curve; driven by:
- Monetary policy changes
- Risk appetite shifts
- Inflation expectations

### Second Component: Slope (Twist)

PC2 has negative loading at short end, positive at long end:

$$V_{:,2} \approx \text{sign}(m - m_{\text{pivot}})} \cdot |m - m_{\text{pivot}}|^{1/2}$$

Captures steepening/flattening movements; driven by:
- Expected rate path changes
- Term premium fluctuations
- Recession probability shifts

### Third Component: Curvature (Butterfly)

PC3 exhibits hump shape at intermediate maturities:

$$V_{:,3} \approx \begin{cases}
+ & \text{if } m \text{ near 2-5Y} \\
- & \text{if } m \text{ short or long}
\end{cases}$$

Represents changes in curve shape; driven by:
- Supply/demand imbalances at specific maturities
- Forward rate expectations
- Optionality effects in bond pricing

## Dynamics of Yield Curve Factors

### Univariate Models for Factor Dynamics

Each principal component typically follows AR(1) process:

$$\text{PC}_k(t) = \rho_k \text{PC}_k(t-1) + \epsilon_{k,t}$$

with characteristics:

| Factor | Mean | Std Dev | AR(1) | Half-Life |
|--------|------|---------|-------|-----------|
| PC1 (Level) | 0 | 0.60% | 0.98 | 35 days |
| PC2 (Slope) | 0 | 0.45% | 0.94 | 12 days |
| PC3 (Curvature) | 0 | 0.30% | 0.85 | 5 days |

PC1 is most persistent; PC3 mean-reverts rapidly.

### Correlation Between Components

Typically low correlation between components:

$$\text{Corr}(\text{PC}_1, \text{PC}_2) \approx 0.1-0.2$$
$$\text{Corr}(\text{PC}_2, \text{PC}_3) \approx -0.15$$

Enables independent modeling and hedging of different yield curve movements.

## Applications to Fixed Income Portfolio Management

### Portfolio Decomposition

Express portfolio duration and convexity in terms of principal components:

$$\text{Duration} = \sum_{k=1}^{3} \beta_k \cdot \text{Duration}_k$$

where $\beta_k$ measures exposure to factor k, enabling targeted risk management.

### Factor Hedging

Isolate and hedge specific yield curve exposures:

1. Long bonds with positive level exposure, short bonds with negative exposure
2. Create butterfly trades to isolate curvature movements
3. Match short/long positions to control duration

### Relative Value Identification

Compare empirical PCA loadings to theoretical expectations:

- Bonds with loading mismatch represent relative value opportunities
- Anomalous curvature factor loading suggests mispricing
- Level factor exposure identifies "rich" vs "cheap" curve segments

### Factor-Based Bond Indexing

Construct bond portfolio using only principal components:

$$\min_w \left\|w - w_{\text{index}}\right\|^2 \text{ subject to } |w_i| \leq c$$

Dramatically reduces number of holdings while tracking index factor exposures.

## Time-Varying Yield Curve Factor Loadings

### Rolling PCA

Update principal components with rolling window (e.g., 252 trading days):

1. Compute covariance of yield changes in each window
2. Extract eigenvalues and eigenvectors
3. Track changes in factor loadings over time

Reveals whether factor structure remains stable across interest rate regimes.

### Eigenvalue Stability

Monitor dominant eigenvalue ratio:

$$\text{Ratio} = \frac{\lambda_1}{\lambda_1 + \lambda_2 + \lambda_3}$$

High ratio indicates stable level factor dominance; decreasing ratio suggests increasing complexity in yield curve movements.

## Real-World Considerations

### Data Preparation

1. Use liquid yields (on-the-run Treasuries)
2. Transform non-uniformly-spaced maturities to standardized grid via interpolation
3. Remove outliers (settlements, option-adjusted spreads)
4. Use yield changes (not levels) for stationarity

### Parameter Estimation

For robust estimation:

- Minimum 5 years of daily data (1,260 observations)
- Quarterly reestimation to capture regime changes
- Cross-validation on hold-out data to verify stability

!!! warning "Non-Stationarity"
    Yield curve factor structure exhibits regime-dependent behavior. During policy rate near-zero lower bounds or abnormal market conditions (e.g., COVID), traditional three-factor decomposition may require augmentation or regime-specific estimation.

