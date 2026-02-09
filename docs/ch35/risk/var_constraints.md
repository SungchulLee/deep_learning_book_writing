# 35.4.3 VaR Constraints

## Learning Objectives

- Implement Value-at-Risk constraints in RL portfolio optimization
- Understand parametric, historical, and Monte Carlo VaR estimation
- Design constrained RL formulations with VaR limits
- Handle VaR constraints via Lagrangian relaxation in policy optimization

## Introduction

Value-at-Risk (VaR) quantifies the maximum expected loss over a given time horizon at a specified confidence level. For example, a 1-day 95% VaR of $1M means there is a 5% chance of losing more than $1M in a single day. Regulatory requirements (Basel III) mandate VaR-based capital reserves, making VaR constraints essential in production systems.

## VaR Definition

$$\text{VaR}_\alpha = -\inf\{x : P(R \leq x) > \alpha\}$$

At confidence level $(1-\alpha)$: the loss that is exceeded with probability $\alpha$.

## VaR Estimation Methods

### 1. Parametric (Variance-Covariance)

Assumes Gaussian returns:

$$\text{VaR}_\alpha = -(\mu_p - z_\alpha \cdot \sigma_p) \cdot V$$

where $z_\alpha$ is the standard normal quantile, $\mu_p$ and $\sigma_p$ are portfolio return mean and standard deviation.

### 2. Historical Simulation

Use the empirical distribution of past returns:

$$\text{VaR}_\alpha = -\text{Quantile}_\alpha(\{R_1, R_2, \ldots, R_T\}) \cdot V$$

### 3. Monte Carlo

Simulate future returns from a fitted model and compute the quantile.

## Constrained RL with VaR

### Lagrangian Relaxation

$$\mathcal{L}(\theta, \lambda) = \mathbb{E}_\pi\left[\sum r_t\right] + \lambda \cdot \left(\text{VaR}_\alpha^{\text{limit}} - \text{VaR}_\alpha(\pi)\right)$$

Dual update: $\lambda \leftarrow \max(0, \lambda + \eta (\text{VaR}_\alpha(\pi) - \text{VaR}_\alpha^{\text{limit}}))$

### Reward Penalty

$$r_t^{\text{constrained}} = r_t - \lambda_{\text{var}} \cdot \max(0, \hat{\text{VaR}}_t - \text{VaR}_{\text{limit}})$$

### Action Masking

Before executing an action, check if the resulting portfolio VaR exceeds the limit. If so, scale down the action.

## Summary

VaR constraints ensure RL policies operate within regulatory and risk management bounds. The combination of Lagrangian relaxation for soft constraints and action masking for hard constraints provides robust VaR control.

## References

- Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk. McGraw-Hill.
- Chow, Y., et al. (2017). Risk-Constrained Reinforcement Learning with Percentile Risk Criteria. JMLR.
- Tamar, A., et al. (2015). Optimizing the CVaR via Sampling. AAAI.
