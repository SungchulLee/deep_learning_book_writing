# 32.5.4 Importance Sampling

## Concept

**Importance sampling** (IS) is a statistical technique for estimating expectations under one distribution using samples from another. In RL, it enables off-policy learning by reweighting returns.

## Mathematical Foundation

To estimate $\mathbb{E}_\pi[f(X)]$ using samples from $b$:

$$\mathbb{E}_\pi[f(X)] = \mathbb{E}_b\left[\frac{\pi(X)}{b(X)} f(X)\right]$$

The ratio $\rho = \frac{\pi(X)}{b(X)}$ is the **importance sampling ratio**.

## Importance Sampling in RL

For a trajectory from time $t$ to the end of episode $T$:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}$$

Note: transition probabilities cancel because $P(s'|s,a)$ is the same regardless of which policy selected $a$.

## Ordinary Importance Sampling (OIS)

$$V_\pi(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}$$

where $\mathcal{T}(s)$ is the set of time steps when state $s$ is visited.

**Properties**:
- **Unbiased**: $\mathbb{E}[\hat{V}] = V_\pi(s)$
- **High variance**: Can be extreme when $\rho$ is large
- **Infinite variance possible**: When $\pi/b$ is unbounded

## Weighted Importance Sampling (WIS)

$$V_\pi(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$

**Properties**:
- **Biased**: But bias $\to 0$ as $n \to \infty$
- **Much lower variance**: Normalized weights are bounded
- **Generally preferred** in practice

## Comparison

| Property | Ordinary IS | Weighted IS |
|----------|-----------|------------|
| Bias | Unbiased | Biased (vanishing) |
| Variance | Can be infinite | Always finite |
| First sample | Can be extreme | Exactly $G_1$ |
| MSE | Higher | Lower (usually) |
| Preferred when | Theory, proofs | Practice |

## Per-Decision Importance Sampling

Instead of weighting the entire return, weight each reward individually:

$$G_t^{\text{PDIS}} = \sum_{k=t}^{T-1} \gamma^{k-t} \left(\prod_{j=t}^{k} \frac{\pi(A_j|S_j)}{b(A_j|S_j)}\right) R_{k+1}$$

This reduces variance because only the relevant portion of the trajectory is weighted for each reward.

## Variance Reduction Techniques

### Truncated Importance Sampling

Clip the importance ratio to prevent extreme values:

$$\bar{\rho} = \min(\rho, c) \quad \text{for some constant } c$$

This introduces bias but dramatically reduces variance. Common choices: $c \in [5, 20]$.

### Self-Normalized IS

$$\hat{V} = \frac{\sum_i w_i G_i}{\sum_i w_i} \quad \text{where } w_i = \rho_i$$

Always produces estimates within the range of observed returns.

### Control Variates

Use the known structure of the problem to reduce variance:

$$\hat{V} = \frac{1}{n} \sum_i \rho_i (G_i - f(X_i)) + \mathbb{E}_b[f(X)]$$

where $f$ is a control variate correlated with $G$.

## Practical Considerations

1. **Log-space computation**: Compute $\log \rho = \sum_k [\log \pi(A_k|S_k) - \log b(A_k|S_k)]$ to avoid numerical overflow
2. **Degeneracy**: In long episodes, $\rho$ often collapses to 0 for most trajectories — only a few trajectories dominate
3. **Effective sample size**: $\text{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}$ measures how many effective samples remain after reweighting

## Financial Applications

### Off-Policy Strategy Evaluation

Evaluate a new trading strategy using historical data collected under a different strategy:

$$\hat{V}_{\text{new}} = \frac{\sum_i \rho_i \cdot \text{Return}_i}{\sum_i \rho_i}$$

where $\rho_i = \prod_{t} \frac{\pi_{\text{new}}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$

### Counterfactual Analysis

IS enables answering: "What would have happened if we had followed a different strategy?" — crucial for strategy research and risk management.

## Summary

Importance sampling is the key technique enabling off-policy learning in MC methods. Weighted IS is generally preferred due to lower variance. Practical implementations use per-decision IS, truncation, and log-space computation to manage variance. In finance, IS enables counterfactual strategy evaluation from historical data.
