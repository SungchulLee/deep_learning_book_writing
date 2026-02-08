# 32.6.3 SARSA

## Overview

**SARSA** (State-Action-Reward-State-Action) is an **on-policy** TD control algorithm. It learns $Q_\pi$ for the current (ε-greedy) policy and improves the policy simultaneously.

## Algorithm

The name comes from the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ used in each update:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]$$

### Pseudocode

```
Initialize Q(s,a) arbitrarily, Q(terminal, ·) = 0

For each episode:
    S = initial state
    A = ε-greedy action from Q(S, ·)
    For each step:
        Take A, observe R, S'
        A' = ε-greedy action from Q(S', ·)
        Q(S,A) ← Q(S,A) + α [R + γ Q(S',A') - Q(S,A)]
        S ← S', A ← A'
    Until S is terminal
```

## On-Policy Nature

SARSA is on-policy because the update target $R + \gamma Q(S', A')$ uses the **actual next action** $A'$ selected by the current policy (including exploration). This means:

- SARSA learns the value of the policy it's following (including exploration)
- The learned policy accounts for exploration costs
- It tends to find safer paths that avoid dangerous states even if the risky path has higher expected value

## SARSA vs. Q-Learning (Preview)

On the Cliff Walking problem:

- **SARSA (on-policy)**: Learns a safe path away from the cliff, accounting for the ε probability of random steps that could lead to falling
- **Q-Learning (off-policy)**: Learns the optimal path along the cliff edge, ignoring the fact that ε-greedy exploration might cause falls

## Convergence

SARSA converges to $Q^*$ under GLIE conditions:
1. All state-action pairs visited infinitely often
2. $\epsilon_t \to 0$ as $t \to \infty$ (e.g., $\epsilon_t = 1/t$)
3. Step sizes satisfy Robbins-Monro conditions

## Financial Application

SARSA is appropriate when the exploration policy itself must be evaluated. In trading, this means the learned Q-values reflect the actual trading costs of exploration (trying new strategies), giving a more realistic estimate of strategy performance during the learning phase.
