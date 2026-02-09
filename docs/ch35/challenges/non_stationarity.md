# 35.5.1 Non-Stationarity

## Learning Objectives

- Understand why financial markets are non-stationary and how this impacts RL
- Implement adaptive RL methods that handle distribution shift
- Design training procedures robust to non-stationarity
- Use online learning and continual adaptation techniques

## Introduction

Financial markets are fundamentally non-stationary: the data-generating process changes over time due to evolving market structure, changing participants, regulatory shifts, and macroeconomic cycles. An RL agent trained on historical data faces the challenge that the environment it encounters in production may differ substantially from its training environment.

## Sources of Non-Stationarity

| Source | Example | Impact on RL |
|--------|---------|-------------|
| Market microstructure | Decimalization, electronic trading | Changes execution dynamics |
| Monetary policy | Interest rate regimes | Alters asset correlations |
| Regulation | Short-selling bans, circuit breakers | Changes action space |
| Technology | HFT proliferation | Changes signal decay rates |
| Participant behavior | Crowded trades, factor rotation | Changes reward distribution |

## Detection Methods

### 1. Statistical Tests

- **CUSUM test**: Detects cumulative deviations from the mean
- **Page-Hinkley test**: Sequential change detection
- **Kolmogorov-Smirnov test**: Compares recent vs. historical return distributions

### 2. Rolling Statistics

Monitor rolling means, variances, and correlations. Alert when they deviate significantly from training-period values.

### 3. Model Performance Monitoring

Track the agent's prediction accuracy, reward, and portfolio metrics on a rolling basis. Degradation signals distribution shift.

## Adaptation Strategies

### 1. Rolling Window Training

Retrain on the most recent $W$ days, discarding older data:

$$\theta_t = \arg\max_\theta \mathbb{E}_{s \sim \mathcal{D}_{t-W:t}} [V^\pi(s)]$$

### 2. Exponential Weighting

Weight recent experience more heavily:

$$\mathcal{L} = \sum_{i} \lambda^{t-i} \ell_i$$

### 3. Online Fine-Tuning

Continuously update the policy with small learning rate using live data.

### 4. Meta-Learning (MAML)

Train a policy that can quickly adapt to new market conditions with few gradient steps.

### 5. Domain Randomization

Train on augmented data with randomized parameters (drift, volatility, correlation) to learn robust policies.

## Summary

Non-stationarity is the fundamental challenge of financial RL. No single technique fully solves it, but combining rolling window training, continual adaptation, and robust training procedures significantly improves policy durability.

## References

- Lim, B. & Zohren, S. (2021). Time-Series Forecasting With Deep Learning: A Survey. Philosophical Transactions A.
- Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation. ICML.
