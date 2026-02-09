# 35.3.3 Statistical Arbitrage

## Learning Objectives

- Formulate statistical arbitrage as a reinforcement learning problem
- Implement pairs trading with RL-based entry/exit signals
- Understand mean-reversion strategies and cointegration
- Design multi-pair arbitrage with portfolio-level risk management

## Introduction

Statistical arbitrage (stat arb) exploits temporary mispricings between related securities. The classic example is pairs trading: when two historically correlated assets diverge, the strategy bets on convergence. RL enhances stat arb by learning optimal entry/exit thresholds, position sizing, and multi-pair coordination from data.

## Pairs Trading Framework

### Spread Construction

For a pair of assets $(A, B)$, the spread is:

$$z_t = \log(P_t^A) - \beta \log(P_t^B)$$

where $\beta$ is the hedge ratio, typically estimated via:
- **OLS regression**: $\beta = \text{Cov}(A, B) / \text{Var}(B)$
- **Kalman filter**: time-varying $\beta_t$
- **Johansen test**: cointegration-based estimation

### Mean Reversion

If the spread is cointegrated (stationary), it has a mean-reverting property:

$$dz_t = \theta(\mu - z_t)dt + \sigma dW_t$$

The Ornstein-Uhlenbeck (OU) process parameters:
- $\theta$: speed of mean reversion (higher = faster)
- $\mu$: long-run mean of the spread
- $\sigma$: volatility of the spread

### Z-Score Signal

$$\text{z-score}_t = \frac{z_t - \mu_t^{\text{rolling}}}{\sigma_t^{\text{rolling}}}$$

Traditional rules: enter when $|\text{z-score}| > 2$, exit when $|\text{z-score}| < 0.5$.

## MDP Formulation

### State

$$s_t = \left(z_t, \text{zscore}_t, \theta_t, \sigma_t, \text{position}_t, \text{PnL}_t, \text{holding\_time}_t\right)$$

### Action

Discrete: $a_t \in \{$long spread, short spread, close, hold$\}$

Or continuous: $a_t \in [-1, 1]$ representing position size and direction.

### Reward

$$r_t = \text{position}_t \cdot \Delta z_t - c \cdot |\Delta \text{position}_t| - \lambda \cdot \text{holding\_cost}_t$$

## RL Advantages for Stat Arb

| Aspect | Traditional | RL-Based |
|--------|-------------|----------|
| Entry/exit | Fixed thresholds | Adaptive, state-dependent |
| Position sizing | Fixed or linear | Non-linear, volatility-adjusted |
| Holding period | Time-based | Return-based |
| Multi-pair | Independent | Joint optimization |
| Regime awareness | Rolling window | Learned regime detection |

## Hedge Ratio Estimation

### Static Hedge Ratio

$$\beta = \frac{\text{Cov}(r_A, r_B)}{\text{Var}(r_B)}$$

### Dynamic Hedge Ratio (Kalman Filter)

The Kalman filter provides a time-varying estimate:

$$\beta_t = \beta_{t-1} + K_t (P_t^A - \beta_{t-1} P_t^B)$$

where $K_t$ is the Kalman gain.

## Multi-Pair Portfolio

For $K$ pairs, the agent manages a portfolio of spreads:

$$\mathbf{a}_t = (a_t^1, a_t^2, \ldots, a_t^K)$$

Subject to constraints:
- Total exposure limit: $\sum |a_t^k| \leq L_{\max}$
- Per-pair limit: $|a_t^k| \leq a_{\max}$
- Sector neutrality: $\sum_{\text{sector}} a_t^k \approx 0$

## Summary

Statistical arbitrage via RL learns adaptive trading rules that outperform fixed-threshold approaches, especially during regime changes. The key innovation is joint optimization of entry/exit timing, position sizing, and multi-pair allocation. Challenges include non-stationarity of cointegration relationships and the need for robust hedge ratio estimation.

## References

- Gatev, E., Goetzmann, W., & Rouwenhorst, K. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. Review of Financial Studies.
- Krauss, C. (2017). Statistical Arbitrage Pairs Trading Strategies: Review and Outlook. Journal of Economic Surveys.
- Kim, G., Shin, D., & Oh, H.S. (2023). Reinforcement Learning for Statistical Arbitrage. arXiv.
