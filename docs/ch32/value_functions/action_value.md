# 32.3.2 Action Value Function

## Definition

The **action value function** (or **Q-function**) $Q_\pi(s, a)$ gives the expected return when starting in state $s$, taking action $a$, and then following policy $\pi$:

$$Q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$$

This answers: "How good is it to take action $a$ in state $s$ under policy $\pi$?"

## Relationship to State Value Function

The state value function is the expected Q-value under the policy:

$$V_\pi(s) = \sum_a \pi(a|s) Q_\pi(s, a) = \mathbb{E}_{a \sim \pi}[Q_\pi(s, a)]$$

Conversely, the Q-function can be expressed in terms of $V_\pi$:

$$Q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')$$

This relationship is key: the Q-function looks one step ahead (immediate reward + discounted value of next state).

## Advantage Function

The **advantage function** measures how much better action $a$ is compared to the average action under $\pi$:

$$A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)$$

Properties:
- $\sum_a \pi(a|s) A_\pi(s, a) = 0$ (advantages average to zero under $\pi$)
- $A_\pi(s, a) > 0$ means action $a$ is better than average in state $s$
- Used extensively in actor-critic methods (e.g., A2C, GAE)

## Optimal Action Value Function

$$Q^*(s, a) = \max_\pi Q_\pi(s, a)$$

The optimal Q-function directly determines the optimal policy without needing a model:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

This is why Q-learning and DQN are so popular — they learn $Q^*$ directly, and the optimal policy follows immediately.

## Computing $Q_\pi$

### From $V_\pi$ (one-step lookahead)

$$Q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')$$

### Direct Matrix Form

For finite MDPs, the Q-function can be represented as a matrix $\mathbf{Q} \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$:

$$Q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q_\pi(s', a')$$

### Monte Carlo Estimation

$$Q_\pi(s, a) \approx \frac{1}{N(s,a)} \sum_{i=1}^{N(s,a)} G_t^{(i)}$$

where $G_t^{(i)}$ is the return following the $i$-th visit to $(s, a)$.

## Why Q-Functions Are Central to RL

| Feature | V-function | Q-function |
|---------|-----------|-----------|
| **Inputs** | State only | State + action |
| **Policy extraction** | Needs model $P(s'|s,a)$ | Model-free: $\arg\max_a Q(s,a)$ |
| **Table size** | $|\mathcal{S}|$ | $|\mathcal{S}| \times |\mathcal{A}|$ |
| **Used in** | Policy evaluation, TD(0) | Q-learning, SARSA, DQN |

The Q-function is preferred for **model-free** control because it allows policy improvement without knowing the transition dynamics.

## Financial Interpretation

In a trading context:
- $Q_\pi(s, \text{buy})$: Expected future return from buying in market state $s$ under strategy $\pi$
- $Q_\pi(s, \text{sell})$: Expected future return from selling in state $s$
- $A_\pi(s, \text{buy}) > 0$: Buying is better than the strategy's average action in state $s$
- $Q^*(s, a)$: Best possible expected return for each action, enabling optimal decisions

## Summary

The action value function $Q_\pi(s,a)$ extends the state value function to incorporate action choices, making it the workhorse of model-free RL. The advantage function $A_\pi$ decomposes Q-values into state value and action-specific benefit. The optimal Q-function $Q^*$ directly yields the optimal policy without requiring environment dynamics.
