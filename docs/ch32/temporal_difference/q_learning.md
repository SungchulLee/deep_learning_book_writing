# 32.6.4 Q-Learning

## Overview

**Q-Learning** (Watkins, 1989) is an **off-policy** TD control algorithm that directly learns the optimal action-value function $Q^*$, regardless of the exploration policy being followed. It is one of the most important algorithms in RL.

## Algorithm

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\right]$$

The key difference from SARSA: the update uses $\max_{a'} Q(S', a')$ instead of $Q(S', A')$.

### Pseudocode

```
Initialize Q(s,a) arbitrarily, Q(terminal, ·) = 0

For each episode:
    S = initial state
    For each step:
        A = ε-greedy action from Q(S, ·)
        Take A, observe R, S'
        Q(S,A) ← Q(S,A) + α [R + γ max_a' Q(S',a') - Q(S,A)]
        S ← S'
    Until S is terminal
```

## Off-Policy Nature

Q-learning is off-policy because:

- The **behavior policy** (e.g., ε-greedy) determines which actions are taken
- The **target policy** (greedy w.r.t. Q) determines the update target
- The update always uses the best possible Q-value at the next state, regardless of which action was actually taken

This separation means Q-learning directly approximates $Q^*$ even while exploring.

## Convergence

Q-learning converges to $Q^*$ with probability 1, provided:
1. All state-action pairs are visited infinitely often
2. Step sizes satisfy: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$

No GLIE condition on $\epsilon$ is needed for convergence of Q-values (but $\epsilon$ should decrease for the resulting policy to be optimal).

## The Maximization Bias

Q-learning's use of $\max$ introduces **maximization bias**: systematic overestimation of Q-values because:

$$\mathbb{E}[\max_a Q(s, a)] \geq \max_a \mathbb{E}[Q(s, a)]$$

This can slow learning and lead to suboptimal behavior.

### Double Q-Learning

Maintains two independent Q-tables, $Q_1$ and $Q_2$:

$$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha \left[R + \gamma Q_2(S', \arg\max_a Q_1(S', a)) - Q_1(S, A)\right]$$

Uses one table to select the best action and the other to evaluate it, eliminating the positive bias. (Extended to Double DQN in deep RL.)

## Q-Learning vs. SARSA

| Feature | Q-Learning | SARSA |
|---------|-----------|-------|
| Policy type | Off-policy | On-policy |
| Update target | $R + \gamma \max_{a'} Q(S',a')$ | $R + \gamma Q(S', A')$ |
| Learns about | Optimal policy $\pi^*$ | Current policy $\pi_\epsilon$ |
| Exploration impact | Ignored in updates | Included in updates |
| Cliff walking | Optimal (risky) path | Safe path |
| Convergence to $Q^*$ | Yes (with conditions) | Yes (with GLIE) |

## Significance

Q-learning was the first algorithm proven to converge to the optimal policy without requiring a model of the environment. It forms the basis for:

- **Deep Q-Networks (DQN)**: Q-learning with neural network function approximation
- **Rainbow**: Combining multiple Q-learning improvements
- **Many practical applications** in robotics, games, and finance

## Financial Application

Q-learning for optimal trade execution:
- **State**: Order book features, remaining quantity, time left
- **Action**: Order size, price level, order type
- **Reward**: Implementation shortfall (negative of slippage + market impact)
- **Off-policy**: Can learn optimal execution from historical data collected under different strategies
