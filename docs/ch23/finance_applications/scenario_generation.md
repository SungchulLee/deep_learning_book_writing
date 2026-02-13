# Scenario Generation

## Overview

Normalizing flows generate realistic, multi-dimensional financial scenarios for risk management, stress testing, and simulation-based pricing.

## Monte Carlo Simulation

Traditional approach: assume a parametric model and sample from it. Flow-based approach: learn the distribution from historical data and sample from the flow.

```python
def generate_scenarios(flow, n_scenarios, horizon_days):
    daily_returns = flow.sample(n_scenarios * horizon_days)
    daily_returns = daily_returns.reshape(n_scenarios, horizon_days, -1)
    
    cumulative_returns = (1 + daily_returns).cumprod(dim=1) - 1
    return cumulative_returns
```

## Conditional Scenario Generation

Generate scenarios conditioned on specific market conditions:

$$x_{\text{future}} \sim P(x_{\text{future}} \mid x_{\text{observed}})$$

Conditional flows or conditional sampling (fixing some dimensions and sampling the rest) enable this.

## Advantages Over Historical Simulation

| Aspect | Historical Simulation | Flow-Based |
|--------|---------------------|------------|
| Sample diversity | Limited to observed history | Infinite new samples |
| Tail coverage | Limited by sample size | Can extrapolate |
| Conditional scenarios | Difficult | Natural |
| Non-stationarity | Lookback window choice | Retrain or condition |

## Validation

Generated scenarios must be validated against historical data:

- Marginal distributions (QQ-plots per asset)
- Correlation structure (compare correlation matrices)
- Temporal properties (autocorrelation of returns and squared returns)
- Stylized facts (volatility clustering, leverage effect, fat tails)
