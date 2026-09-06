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

## Exercises

**Exercise 1.**
Design a Gymnasium-compatible environment for the financial problem described in this section. Specify the observation space, action space, reward function, and episode termination conditions.

??? success "Solution to Exercise 1"
    Observation space: a vector containing recent returns, current position, portfolio value, and relevant market features (e.g., volatility, volume). Action space: depends on the problem (discrete for buy/hold/sell, continuous for position sizing). Reward: risk-adjusted return per step (e.g., log return minus penalty for risk). Episode terminates after a fixed horizon (e.g., one trading year) or if portfolio value drops below a threshold (margin call). The environment must handle transaction costs, slippage, and market impact realistically. $\square$

---

**Exercise 2.**
Analyze the reward shaping trade-offs for this financial RL problem. Compare at least three candidate reward functions and discuss which properties of the optimal policy each preserves.

??? success "Solution to Exercise 2"
    Candidate rewards: (1) raw PnL -- simple but high variance and delayed; (2) Sharpe-based differential reward $D_t = \frac{\Delta A_t B_{t-1} - \frac{1}{2}\Delta B_t A_{t-1}}{(B_{t-1} - A_{t-1}^2)^{3/2}}$ -- directly optimizes the Sharpe ratio but complex; (3) log return with drawdown penalty $r_t = \log(V_t/V_{t-1}) - \lambda \max(0, DD_t - \tau)$ -- balances return with risk control. The potential-based shaping theorem guarantees that adding $\gamma\Phi(s') - \Phi(s)$ preserves the optimal policy. The Sharpe-based reward changes the optimization objective (may alter optimal policy), while pure PnL preserves it but learns slowly. $\square$

---

**Exercise 3.**
Discuss the non-stationarity challenge specific to this financial application. Propose a concrete strategy for adapting the RL agent to regime changes.

??? success "Solution to Exercise 3"
    Financial markets exhibit regime changes (bull/bear, high/low volatility) that violate the MDP stationarity assumption. A concrete adaptation strategy: (1) include a regime indicator as part of the state (e.g., HMM-estimated regime probabilities); (2) use a meta-learning approach where the agent maintains multiple policies and selects based on detected regime; (3) implement continual learning with an expanding replay buffer weighted toward recent experience (exponential decay weights). The agent should also monitor its own performance and reduce position sizes when out-of-distribution inputs are detected. $\square$

---

**Exercise 4.**
Compare the backtesting results one would expect from this RL approach versus a simple heuristic baseline. What statistical tests should be used to determine if the RL agent genuinely outperforms?

??? success "Solution to Exercise 4"
    Baselines: buy-and-hold, equal-weight, or momentum strategy. The RL agent should be evaluated on walk-forward out-of-sample periods (never on training data). Statistical tests: (1) paired t-test on daily returns for mean difference; (2) bootstrap confidence interval on the Sharpe ratio difference; (3) multiple hypothesis testing correction (e.g., Bonferroni or Holm) if comparing multiple strategies. A common pitfall is p-hacking through hyperparameter tuning on the test set. The evaluation must use a hold-out period that was never used for any model selection. Report both statistical significance and economic significance (transaction costs, capacity). $\square$
