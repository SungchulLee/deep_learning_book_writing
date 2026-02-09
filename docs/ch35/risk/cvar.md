# 35.4.4 CVaR Optimization

## Learning Objectives

- Understand Conditional Value-at-Risk (CVaR) and its advantages over VaR
- Implement CVaR estimation and optimization in RL frameworks
- Design CVaR-constrained policy optimization algorithms
- Apply distributional RL for tail-risk management

## Introduction

Conditional Value-at-Risk (CVaR), also known as Expected Shortfall, measures the expected loss in the worst $\alpha\%$ of scenarios. Unlike VaR, which only identifies a threshold, CVaR captures the severity of tail losses, making it a coherent risk measure suitable for optimization.

## CVaR Definition

$$\text{CVaR}_\alpha = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha] = \frac{1}{\alpha} \int_0^\alpha \text{VaR}_u \, du$$

For a portfolio with return $R$:

$$\text{CVaR}_\alpha = -\frac{1}{\alpha} \mathbb{E}\left[R \cdot \mathbb{1}_{R \leq -\text{VaR}_\alpha}\right]$$

## CVaR vs VaR

| Property | VaR | CVaR |
|----------|-----|------|
| Coherent risk measure | No | Yes |
| Sub-additivity | Not guaranteed | Yes (diversification always helps) |
| Tail sensitivity | Ignores tail shape | Captures expected tail loss |
| Optimization | Non-convex | Convex |
| Regulatory use | Basel III required | Basel III recommended |

## Rockafellar-Uryasev Formulation

CVaR can be computed as an optimization:

$$\text{CVaR}_\alpha = \min_{\nu} \left\{ \nu + \frac{1}{\alpha} \mathbb{E}\left[\max(0, -R - \nu)\right] \right\}$$

where $\nu$ is an auxiliary variable (the VaR at optimality).

## CVaR in RL

### CVaR-Constrained Objective

$$\max_\theta \mathbb{E}_\pi\left[\sum r_t\right] \quad \text{s.t.} \quad \text{CVaR}_\alpha(R_\pi) \leq c$$

### Distributional RL for CVaR

Distributional RL (QR-DQN, IQN) learns the full return distribution, enabling direct CVaR computation:

$$\text{CVaR}_\alpha \approx \frac{1}{\lceil \alpha N \rceil} \sum_{i=1}^{\lceil \alpha N \rceil} \hat{q}_i$$

where $\hat{q}_1 \leq \hat{q}_2 \leq \ldots$ are the sorted quantile estimates.

### CVaR Policy Gradient

$$\nabla_\theta \text{CVaR}_\alpha \approx \frac{1}{\alpha N} \sum_{i: R_i \leq \text{VaR}_\alpha} \nabla_\theta \log \pi_\theta(a_i | s_i) \cdot R_i$$

## Summary

CVaR provides a principled framework for tail-risk management in RL. Its convexity and coherence make it superior to VaR for optimization. Distributional RL methods naturally support CVaR estimation and optimization, enabling risk-aware policies that explicitly manage worst-case outcomes.

## References

- Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. Journal of Risk.
- Tamar, A., et al. (2015). Optimizing the CVaR via Sampling. AAAI.
- Dabney, W., et al. (2018). Distributional Reinforcement Learning with Quantile Regression. AAAI.
