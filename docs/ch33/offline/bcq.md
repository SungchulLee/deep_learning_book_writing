# 33.5.3 Batch-Constrained Q-Learning (BCQ)

## Core Idea

**BCQ** (Fujimoto et al., 2019) addresses offline RL by constraining the policy to only select actions that are similar to those in the dataset. Instead of pessimistic Q-values (CQL), BCQ restricts the action space to prevent extrapolation error.

## Approach (Discrete BCQ)

For discrete action spaces, BCQ learns a generative model of the behavior policy and filters actions:

1. Train a behavior cloning network $G_\omega(a|s)$ to estimate $\mu(a|s)$
2. Only consider actions where $G_\omega(a|s) / \max_{a'} G_\omega(a'|s) > \tau$
3. Among the allowed actions, select greedily: $a = \arg\max_{a : G(a|s)/\max G > \tau} Q(s, a)$

The threshold $\tau$ controls how constrained the policy is:
- $\tau = 0$: No constraint (standard DQN)
- $\tau = 1$: Pure behavior cloning (only take the most likely action)
- $\tau \in (0.1, 0.3)$: Good balance (recommended)

## Algorithm

```
1. Train behavior model G_ω(a|s) via supervised learning on dataset
2. Train Q-network with modified Bellman backup:
   - For target: a' = argmax_{a: G(a|s')/max G > τ} Q_target(s', a')
   - Loss: (Q(s,a) - r - γ Q_target(s', a'))²
3. At deployment: a = argmax_{a: G(a|s)/max G > τ} Q(s, a)
```

## Continuous BCQ

For continuous actions, BCQ uses:
1. A **Conditional VAE** to generate action candidates similar to dataset actions
2. A **perturbation network** $\xi_\phi(s, a)$ that adjusts generated actions within a small range $[-\Phi, \Phi]$
3. Multiple candidate actions are generated and the one with highest Q-value is selected

$$a = \arg\max_{a_i} Q(s, a_i + \xi_\phi(s, a_i)), \quad a_i \sim G_\omega(s)$$

## Comparison with CQL

| Aspect | BCQ | CQL |
|--------|-----|-----|
| Mechanism | Action filtering | Q-value penalization |
| Requires behavior model | Yes | No |
| OOD handling | Prevents OOD actions | Penalizes OOD Q-values |
| Conservatism control | Threshold τ | Penalty α |
| Theoretical guarantee | Bounded extrapolation error | Q-value lower bound |

## Hyperparameters

| Parameter | Typical | Notes |
|-----------|---------|-------|
| $\tau$ (threshold) | 0.1–0.3 | Higher = more constrained |
| Behavior model LR | $10^{-3}$ | Trained via cross-entropy |
| Q-network LR | $3 \times 10^{-4}$ | Standard |
| $\Phi$ (perturbation range, continuous) | 0.05 | Small adjustments |
