# 33.5.2 Conservative Q-Learning (CQL)

## Core Idea

**CQL** (Kumar et al., 2020) addresses offline RL's extrapolation error by learning a *conservative* Q-function that lower-bounds the true Q-values. The key insight: instead of preventing OOD actions, penalize Q-values for all actions and reward Q-values for actions in the dataset.

## Objective

CQL adds a regularizer to the standard Bellman loss:

$$\mathcal{L}_{\text{CQL}}(\theta) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D}} \left[\log \sum_a \exp(Q_\theta(s, a))\right] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q_\theta(s, a)]\right) + \frac{1}{2}\mathbb{E}_\mathcal{D}\left[(Q_\theta(s, a) - \hat{\mathcal{B}}^\pi Q_{\theta^-}(s, a))^2\right]$$

The regularizer has two terms:
1. **Push down**: $\log \sum_a \exp(Q(s, a))$ — softmax over all actions, pushes all Q-values down
2. **Push up**: $-\mathbb{E}_{a \sim \mathcal{D}}[Q(s, a)]$ — raises Q-values for actions in the dataset

The net effect: Q-values for in-distribution actions are approximately correct, while Q-values for out-of-distribution actions are conservative (pessimistic).

## Variants

### CQL(H) — Entropy-regularized
Uses the log-sum-exp as the push-down term (as above). This is equivalent to maximizing Q-values under a uniform distribution over actions.

### CQL(ρ) — Policy-based
Replaces the uniform distribution with a learned policy:
$$\mathbb{E}_{a \sim \rho(a|s)}[Q(s, a)] - \mathbb{E}_{a \sim \hat{\pi}_\beta(a|s)}[Q(s, a)]$$

where $\rho$ can be the current policy or a random policy.

## Theoretical Guarantee

CQL provides a lower bound on the true Q-function:

$$Q^{\text{CQL}}(s, a) \leq Q^{\pi}(s, a) \quad \text{for all } (s, a)$$

with high probability, under mild conditions. This ensures the policy derived from $Q^{\text{CQL}}$ has a guaranteed performance lower bound.

## Implementation for Discrete Actions

For discrete action spaces, CQL is straightforward:

```python
# Push down: log-sum-exp over all actions
logsumexp = torch.logsumexp(q_all_actions, dim=1).mean()

# Push up: Q-values for dataset actions
dataset_q = q_all_actions.gather(1, actions.unsqueeze(1)).squeeze(1).mean()

# CQL regularizer
cql_loss = alpha * (logsumexp - dataset_q)

# Total loss = CQL regularizer + standard Bellman loss
total_loss = cql_loss + bellman_loss
```

## Hyperparameters

| Parameter | Typical | Notes |
|-----------|---------|-------|
| $\alpha$ | 1.0–5.0 | Conservative penalty weight; higher = more conservative |
| $\alpha$ (auto-tuned) | Lagrangian | Tuned to achieve a target Q-value gap |
| Learning rate | $3 \times 10^{-4}$ | Standard for offline RL |

## Adaptive α (Lagrangian)

CQL can automatically tune $\alpha$ using a Lagrangian formulation:

$$\alpha^* = \arg\min_{\alpha \geq 0} \alpha \cdot (\text{CQL regularizer} - \tau)$$

where $\tau$ is a target threshold. This prevents over-conservatism.

## Finance Application

CQL is well-suited for financial trading because:
- Historical trading data is the "offline dataset"
- Conservative Q-values prevent the agent from taking extreme positions not supported by data
- The lower-bound guarantee provides risk-aware decision making
- $\alpha$ can be tuned to control the degree of conservatism (risk appetite)
