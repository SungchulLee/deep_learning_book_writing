# 35.1.2 State Representations

## Learning Objectives

- Design effective state representations for financial RL agents
- Understand which features carry predictive signal
- Implement feature normalization and preprocessing for RL
- Handle multi-asset, multi-timeframe state spaces

## Introduction

The state representation is arguably the most critical design decision in a financial RL system. It determines what information the agent can observe and, consequently, what patterns it can learn. Unlike game environments where the full state is often observable, financial markets are partially observable—the agent sees a noisy projection of a vastly more complex underlying system.

## Categories of State Features

### 1. Price-Based Features

Raw prices are non-stationary and should be transformed into stationary features:

| Feature | Formula | Interpretation |
|---------|---------|---------------|
| Log returns | $r_t = \ln(p_t / p_{t-1})$ | Percentage price change |
| Normalized price | $(p_t - \mu_{w}) / \sigma_{w}$ | Z-scored over window $w$ |
| Return momentum | $\sum_{i=0}^{k-1} r_{t-i}$ | Cumulative return over $k$ periods |
| Realized volatility | $\sqrt{\sum_{i=0}^{k-1} r_{t-i}^2}$ | Price variability |

### 2. Technical Indicators

Computed from price and volume data:

| Indicator | Window | Signal |
|-----------|--------|--------|
| RSI (Relative Strength Index) | 14 | Overbought/oversold |
| MACD | 12/26/9 | Trend direction and momentum |
| Bollinger Band position | 20 | Mean reversion signal |
| ATR (Average True Range) | 14 | Volatility measure |
| OBV (On-Balance Volume) | — | Volume-price trend |

### 3. Portfolio State

The agent must know its current positions:

- Current portfolio weights $w_t \in \mathbb{R}^N$
- Unrealized P&L per position
- Time since last trade per asset
- Available capital / buying power
- Current leverage ratio

### 4. Market Microstructure (for high-frequency)

- Bid-ask spread
- Order book imbalance
- Trade flow imbalance
- Quote arrival rate

### 5. Cross-Asset Features

- Correlation matrix (rolling window)
- Sector/industry exposures
- Market regime indicators (VIX level, yield curve slope)

## State Space Design

### Flat Vector Representation

The simplest approach concatenates all features into a single vector:

$$s_t = [f_t^{\text{price}}, f_t^{\text{tech}}, f_t^{\text{portfolio}}, f_t^{\text{market}}] \in \mathbb{R}^D$$

This works well with MLP-based policies but doesn't capture temporal structure.

### Temporal Tensor Representation

For sequence models (LSTM, Transformer), organize features as a 2D tensor:

$$S_t = \begin{bmatrix} f_{t-L+1}^{(1)} & \cdots & f_{t-L+1}^{(F)} \\ \vdots & \ddots & \vdots \\ f_t^{(1)} & \cdots & f_t^{(F)} \end{bmatrix} \in \mathbb{R}^{L \times F}$$

where $L$ is the lookback window and $F$ is the number of features.

### Multi-Channel Representation

For CNN-based policies, stack different feature types as channels:

$$S_t \in \mathbb{R}^{C \times L \times N}$$

where $C$ = feature channels, $L$ = lookback, $N$ = number of assets.

## Normalization Strategies

Normalization is critical for RL training stability:

### Rolling Z-Score

$$\hat{f}_t = \frac{f_t - \mu_t^{(w)}}{\sigma_t^{(w)} + \epsilon}$$

where $\mu_t^{(w)}$ and $\sigma_t^{(w)}$ are rolling mean and standard deviation over window $w$.

### Rank Normalization

Transform features to uniform $[0, 1]$ using their rank within the cross-section:

$$\hat{f}_{t,i} = \frac{\text{rank}(f_{t,i})}{N}$$

This is robust to outliers and preserves relative ordering.

### Adaptive Normalization

Maintain exponential moving statistics:

$$\mu_t = \alpha f_t + (1 - \alpha) \mu_{t-1}$$
$$\sigma_t^2 = \alpha (f_t - \mu_t)^2 + (1 - \alpha) \sigma_{t-1}^2$$

## Handling Missing Data

Financial data often has missing values (holidays, halts, delistings):

1. **Forward fill**: Carry last known value (most common for prices)
2. **Masking**: Include a binary mask indicating data availability
3. **Imputation**: Use cross-sectional or model-based imputation
4. **Sentinel values**: Use a special value (e.g., 0) with an indicator feature

## Implementation Considerations

### Observation Space Definition

```python
# Flat observation
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(num_features,),
    dtype=np.float32
)

# Temporal observation with Dict space
observation_space = spaces.Dict({
    'market': spaces.Box(-np.inf, np.inf, (lookback, num_assets, num_features)),
    'portfolio': spaces.Box(-1, 1, (num_assets,)),
    'account': spaces.Box(-np.inf, np.inf, (3,)),  # cash, equity, leverage
})
```

### Feature Computation Pipeline

```python
def _get_obs(self):
    window = self.data_feeder.get_window()
    
    # Price features
    returns = np.diff(np.log(window['prices']), axis=0)
    volatility = returns.std(axis=0)
    momentum = returns.sum(axis=0)
    
    # Technical indicators
    rsi = self._compute_rsi(window['prices'])
    
    # Portfolio state
    weights = self.portfolio.get_weights()
    
    # Normalize
    market_features = self._normalize(
        np.column_stack([returns[-1], volatility, momentum, rsi])
    )
    
    return {
        'market': market_features.astype(np.float32),
        'portfolio': weights.astype(np.float32),
    }
```

## Common Pitfalls

1. **Look-ahead bias**: Never include future information in the state. Even centering/scaling must use only past data.
2. **Non-stationarity**: Raw prices, volumes, or dollar values as features cause training instability.
3. **Feature explosion**: Too many features increase sample complexity. Start minimal and add features based on ablation studies.
4. **Ignoring portfolio state**: The agent must know its current positions to make informed decisions.

## Summary

Effective state representations combine price-derived features (returns, volatility, momentum), technical indicators, portfolio state, and optionally market microstructure data. All features must be normalized using only historically available information to prevent look-ahead bias. The choice between flat, temporal, and multi-channel representations depends on the policy architecture.

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. Review of Financial Studies
- Kolm, P., & Ritter, G. (2019). Modern Perspectives on Reinforcement Learning in Finance
