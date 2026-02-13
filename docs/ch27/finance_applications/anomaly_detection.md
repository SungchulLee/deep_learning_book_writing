# Anomaly Detection Using Energy-Based Models in Finance

## Introduction

Anomaly detection in financial markets—identifying unusual trading patterns, rare market microstructure events, or unexpected correlations—is critical for surveillance, fraud prevention, and risk management. Energy-Based Models (EBMs) offer a principled probabilistic framework for anomaly detection by learning an energy function that assigns lower energy (higher probability) to normal market conditions and higher energy to anomalous configurations.

Unlike supervised classification approaches that require labeled anomalies, EBMs learn the distribution of normal market behavior unsupervised, enabling detection of novel anomalies not present in training data. The energy function naturally provides anomaly scores through its ability to distinguish between normal-in-distribution and out-of-distribution samples, making EBMs particularly suited for detecting regime-breaking market events and rare trading phenomena.

This section develops practical EBM applications for financial anomaly detection, demonstrates energy-based scoring mechanisms, and explores integration with other deep learning methods for robust anomaly identification.

## Key Concepts

### Energy-Based Perspective
- **Energy Function**: E(x) assigns scalar energy to each market state x
- **Probability Model**: $p(x) \propto \exp(-E(x))$
- **Anomaly Score**: E(x) - higher energy indicates anomalous configuration
- **Distribution Learning**: Learns p(x) from normal market data

### Anomaly Detection Applications
- **Order Flow Anomalies**: Unusual trading volumes, latency patterns
- **Market Microstructure**: Extreme bid-ask spreads, quote stuffing
- **Portfolio Risk**: Unusual factor exposure combinations
- **Correlation Breakdown**: Assets deviating from historical correlation structure

## Mathematical Framework

### Energy-Based Model Formulation

Define energy function E(x; θ) where x is market observation:

$$p(x; \theta) = \frac{\exp(-E(x; \theta))}{Z(\theta)}$$

Partition function:

$$Z(\theta) = \int \exp(-E(x; \theta)) dx$$

Models p(x) without explicit density, enabling flexible function approximation.

### Training via Contrastive Divergence

Maximize log-likelihood of observed normal data:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}[\log p(x; \theta)] - \mathbb{E}_{x \sim p_\theta}[\log p(x; \theta)]$$

Contrastive Divergence approximates gradient:

$$\nabla_\theta \mathcal{L} \approx \mathbb{E}_{x \sim p_{\text{data}}}[\nabla_\theta E(x)] - \mathbb{E}_{x \sim p_\theta}[\nabla_\theta E(x)]$$

First term pulls energy down on data; second term pushes up on model samples.

### Anomaly Scoring

Given trained model, compute anomaly score:

$$\text{AnomalyScore}(x) = E(x; \theta^*) - \text{median}(E(x) | x \in D_{\text{train}})$$

Positive scores indicate higher-than-normal energy (anomalous). Threshold at percentile α (e.g., 95th) determines alerts.

## Energy-Based Architectures for Finance

### Neural Network Energy Function

Parametrize energy using neural network:

$$E(x; \theta) = \text{NN}_\theta(x) \in \mathbb{R}$$

Maps high-dimensional market observations to scalar energy. Network architecture typically:
- Input layer: d_in = number of market features
- Hidden layers: 128-256 units with ReLU activation
- Output layer: 1 unit (energy scalar)

### Structured Energy Functions

For financial problems, design energy to capture domain knowledge:

$$E(x) = E_{\text{normal}}(x) + E_{\text{constraint}}(x)$$

Example for market microstructure:

$$E_{\text{constraint}}(x) = \lambda \cdot \text{Indicator}(\text{spread} > \text{max\_spread})$$

Hard constraints ensure physically impossible states have infinite energy.

## Applications to Financial Anomalies

### Market Microstructure Anomalies

For order book snapshots with bid/ask/volume data:

1. **Train on Normal Data**: Use pre-crisis trading data to learn normal microstructure
2. **Energy Function**: Captures typical bid-ask spreads, depth distribution, volatility patterns
3. **Anomaly Detection**: Unusual spreads, volume concentrations, or volatility spikes detected as high-energy states

Effective for detecting:
- Flash crashes (extreme price movements)
- Quote stuffing (excessive cancellations)
- Spoofing (fake orders to manipulate prices)

### Portfolio Risk Anomalies

Monitor portfolio exposure for unusual factor combinations:

$$x = [r_{\text{portfolio}}, \beta_{\text{market}}, \beta_{\text{size}}, \beta_{\text{value}}, \sigma_p]$$

EBM learns typical relationships between returns, betas, and volatility. Anomalies indicate:
- Unexpected leverage changes
- Correlated bet concentrations
- Volatility-return mismatches

### Correlation Breakdown Detection

Market regimes exhibit typical correlation structures. EBM trained on stable-period correlations detects when correlation matrix deviates:

$$x = \text{vec}(\rho)$$

Energy increases when correlation matrix exhibits unusual eigenvalue spectrum or cross-asset relationships.

## Comparative Analysis: EBM vs Alternatives

### EBM vs Isolation Forest

| Aspect | EBM | Isolation Forest |
|--------|-----|-----------------|
| **Density Modeling** | Explicit p(x) | Implicit density |
| **High Dimensions** | Requires careful design | Scales well |
| **Interpretability** | Energy provides measure | Path length opaque |
| **Theoretical Guarantee** | Probabilistic framework | Heuristic |

EBMs provide principled probabilistic interpretation; Isolation Forest more scalable.

### EBM vs Autoencoders

Both unsupervised anomaly detection methods:

- **EBM**: Direct density model; anomaly = low probability
- **AE**: Reconstruction error; anomaly = high error

EBMs tend to better capture tail behavior; autoencoders may fail on very rare anomalies.

### EBM vs LSTM for Sequential Anomalies

For time-series anomalies:

- **EBM**: Evaluates individual states; no temporal context
- **LSTM**: Learns temporal patterns; detects breaks in sequence

Hybrid approach: EBM on LSTM latent states captures both statistical anomaly and sequence irregularity.

## Training Considerations for Financial Data

### Data Preparation

Normalize features for energy computation:

1. **Standardization**: $(x - \mu) / \sigma$ ensures features on comparable scale
2. **Outlier Handling**: Cap extreme values (preserve but prevent gradient explosion)
3. **Feature Engineering**: Include domain-relevant features (volatility, spreads, volumes)

### Handling Non-Stationarity

Financial data exhibits changing statistics. Remedies:

1. **Rolling Training**: Retrain EBM quarterly with recent 2-3 years of data
2. **Adaptive Thresholds**: Adjust anomaly threshold by regime
3. **Ensemble Models**: Train multiple EBMs on different periods; alert when majority flag anomaly

### Imbalanced Anomalies

Most training data is normal; few anomalies. Approaches:

1. **Cost-Weighted Loss**: Assign higher weight to rare anomaly examples
2. **Synthetic Anomalies**: Generate artificial out-of-distribution samples
3. **One-Class Learning**: Explicitly optimize for one-class (normal) detection

## Practical Implementation

### Anomaly Detection Pipeline

1. **Preprocessing**: Standardize features, remove missing data
2. **Train EBM**: On normal market data (1-3 years of trading)
3. **Calibrate Threshold**: Choose percentile (90th-99th) for anomaly alerts
4. **Monitor**: Track energy scores of new data; alert on threshold exceedance
5. **Validate**: Compare anomaly flags to known market events

### Interpretability and Feedback

When anomaly detected, explain through:

$$\text{Feature Contribution} = \frac{\partial E(x)}{\partial x_i} \cdot (x_i - \mu_i)$$

Shows which features drive anomaly score; aids human trader judgment.

## Integration with Other Methods

### EBM + Autoencoder Hybrid

Combine both approaches:

1. Use AE to compress market observations to latent space
2. Train EBM on AE latent representation
3. Hybrid anomaly score: $E_{\text{hybrid}} = E(z) + \|x - \hat{x}\|^2$

Captures both density estimation (EBM) and reconstruction error (AE).

### EBM + Time Series Models

For sequential markets:

1. Train EBM on statistical features of returns (variance, skewness, kurtosis)
2. Train ARIMA/GARCH on returns
3. Flag anomaly if either model unusual

Ensemble approach improves robustness.

!!! note "Practical Guidance"
    EBMs work well for detecting anomalies in reasonably high-dimensional settings (tens of features) where normal market behavior can be learned from data. For extremely high dimensions (>1000), combine with dimensionality reduction (PCA) first to avoid curse of dimensionality.

