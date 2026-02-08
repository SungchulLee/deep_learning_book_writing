# 35.1.5 Market Simulation

## Learning Objectives

- Build realistic market simulators for RL training
- Model transaction costs, slippage, and market impact
- Implement order execution with realistic fill assumptions
- Generate synthetic market data for environment augmentation

## Introduction

The market simulator determines how the agent's actions translate into actual portfolio changes. A naive simulator that fills all orders at the closing price creates unrealistic expectations. Realistic simulation must account for transaction costs, slippage, partial fills, and market impact—especially for strategies that trade significant volume relative to available liquidity.

## Order Execution Models

### 1. Simple Fill Model

All orders execute at the next period's opening price (or closing price):

$$\text{fill\_price}_i = p_{t+1}^{\text{open}} \cdot (1 + \text{slippage}_i)$$

where slippage is a random or deterministic spread penalty.

### 2. VWAP Execution Model

For orders executed over a period, approximate fill price using volume-weighted average price:

$$\text{fill\_price}_i = \text{VWAP}_{[t, t+\Delta t]} + \text{market\_impact}_i$$

### 3. Limit Order Book Simulation

For high-frequency strategies, simulate the limit order book:

- Model bid/ask spread dynamics
- Queue position for limit orders
- Market impact from consuming liquidity

## Transaction Cost Models

### Proportional Costs

$$\text{TC}_t = c \cdot \sum_i |q_{t,i}| \cdot p_{t,i}$$

where $c$ is the cost rate (typically 5-20 basis points for equities), $q_{t,i}$ is trade quantity.

### Spread-Based Costs

$$\text{TC}_t = \frac{1}{2} \sum_i |q_{t,i}| \cdot \text{spread}_{t,i}$$

The half-spread model: buying at the ask, selling at the bid.

### Tiered Commission Model

```python
def compute_commission(self, trade_value):
    if trade_value < 10_000:
        return max(1.0, trade_value * 0.005)  # $1 min, 50 bps
    elif trade_value < 100_000:
        return trade_value * 0.001              # 10 bps
    else:
        return trade_value * 0.0005             # 5 bps
```

## Market Impact Models

### Linear Impact

$$\Delta p = \lambda \cdot \frac{q}{V}$$

where $q$ is order quantity, $V$ is average daily volume, and $\lambda$ is the impact coefficient.

### Square-Root Impact (Almgren-Chriss)

$$\Delta p = \sigma \cdot \eta \cdot \text{sgn}(q) \cdot \sqrt{\frac{|q|}{V}}$$

where $\sigma$ is volatility and $\eta$ is a calibrated parameter. This model better captures the concave relationship between trade size and impact.

### Temporary vs. Permanent Impact

- **Temporary impact**: Price reverts after the trade; cost is paid only by the trader
- **Permanent impact**: Price moves permanently; affects all subsequent valuations

$$p_{t+1} = p_t + \underbrace{\gamma \cdot \frac{q}{V}}_{\text{permanent}} + \underbrace{\eta \cdot \text{sgn}(q) \cdot \sqrt{\frac{|q|}{V}}}_{\text{temporary}} + \epsilon_t$$

## Slippage Modeling

Slippage captures the difference between expected and actual fill price:

### Deterministic Slippage

$$\text{slippage} = \text{sign}(q) \cdot s$$

where $s$ is a fixed slippage amount (e.g., 1 basis point).

### Stochastic Slippage

$$\text{slippage} \sim \mathcal{N}(\mu_s, \sigma_s^2)$$

Calibrate $\mu_s$ and $\sigma_s$ from historical execution data.

### Volume-Dependent Slippage

$$\text{slippage} = s_0 + s_1 \cdot \frac{|q|}{V_t}$$

Larger orders relative to volume incur more slippage.

## Synthetic Data Generation

Training on limited historical data causes overfitting. Synthetic data augmentation helps:

### Bootstrap Resampling

Resample historical returns with replacement to create new price paths:

$$R_t^{\text{synthetic}} = R_{\pi(t)}^{\text{historical}}$$

where $\pi$ is a random permutation (block bootstrap preserves autocorrelation).

### Generative Models

Use trained generative models to produce realistic synthetic data:

- **GARCH models**: Capture volatility clustering
- **Regime-switching models**: Capture market regime dynamics
- **GANs/VAEs**: Learn complex distributional features

### Noise Injection

Add calibrated noise to historical data:

$$p_t^{\text{aug}} = p_t \cdot e^{\epsilon_t}, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{aug}}^2)$$

## Domain Randomization

Vary environment parameters during training to improve robustness:

```python
def randomize_params(self):
    """Randomize environment parameters for each episode."""
    self.transaction_cost = np.random.uniform(0.0005, 0.002)
    self.slippage_mean = np.random.uniform(0.0, 0.001)
    self.market_impact_coeff = np.random.uniform(0.05, 0.2)
    self.initial_capital *= np.random.uniform(0.8, 1.2)
```

## Multi-Asset Correlation

For multi-asset environments, properly model cross-asset dynamics:

- **Historical correlation**: Use rolling correlation matrices
- **Copula models**: Capture non-linear dependencies
- **Factor models**: Decompose returns into common factors and idiosyncratic components

## Implementation: Market Simulator Class

```python
class MarketSimulator:
    def __init__(self, config):
        self.cost_rate = config.get('transaction_cost', 0.001)
        self.slippage_std = config.get('slippage_std', 0.0005)
        self.impact_coeff = config.get('impact_coeff', 0.1)
        self.volume_data = config.get('volume')
    
    def execute(self, target_weights, portfolio, current_prices):
        current_weights = portfolio.get_weights()
        trades = target_weights - current_weights
        
        # Compute fill prices with slippage and impact
        fill_prices = self._compute_fill_prices(
            current_prices, trades, portfolio.total_value
        )
        
        # Compute transaction costs
        trade_values = np.abs(trades) * portfolio.total_value
        costs = self._compute_costs(trade_values)
        
        return {
            'fill_prices': fill_prices,
            'costs': costs,
            'trades': trades,
        }
    
    def _compute_fill_prices(self, prices, trades, portfolio_value):
        # Slippage
        slippage = np.random.normal(0, self.slippage_std, len(prices))
        
        # Market impact (square-root model)
        if self.volume_data is not None:
            participation = np.abs(trades) * portfolio_value / (prices * self.volume_data)
            impact = self.impact_coeff * np.sign(trades) * np.sqrt(participation)
        else:
            impact = 0
        
        fill_prices = prices * (1 + np.sign(trades) * slippage + impact)
        return fill_prices
    
    def _compute_costs(self, trade_values):
        return self.cost_rate * trade_values.sum()
```

## Summary

Realistic market simulation is essential for training RL agents that transfer to live trading. Key elements include appropriate transaction cost models, market impact estimation (especially the square-root model), stochastic slippage, and synthetic data generation for training robustness. Domain randomization over simulation parameters builds policies that are robust to real-world variability.

## References

- Almgren, R., & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. Journal of Risk
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). Algorithmic and High-Frequency Trading
