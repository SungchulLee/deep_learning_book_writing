# 32.1.3 The Reward Hypothesis

## Statement

The **reward hypothesis** is a foundational assumption of reinforcement learning:

> *That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).*
>
> — Sutton & Barto (2018)

This hypothesis asserts that any goal-directed behavior can be captured by a suitably designed reward function. It is both the great strength and a potential limitation of the RL framework.

## Formal Statement

The agent's objective is to maximize the **expected return**:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

where $\gamma \in [0, 1]$ is the **discount factor**.

The agent seeks a policy $\pi^*$ such that:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi[G_t]$$

for all states $S_t$.

## Why a Scalar Reward?

Using a single scalar reward signal might seem restrictive, but it provides:

1. **Clear optimization objective**: A scalar defines a total ordering over outcomes
2. **Simplicity**: Avoids the complexities of multi-objective optimization
3. **Generality**: Complex goals can often be decomposed into a scalar reward function
4. **Theoretical tractability**: Enables the Bellman equation framework

## The Return

The return $G_t$ can take several forms depending on the problem:

### Finite-Horizon Undiscounted Return

For episodic tasks with terminal time $T$:

$$G_t = R_{t+1} + R_{t+2} + \cdots + R_T = \sum_{k=0}^{T-t-1} R_{t+k+1}$$

### Infinite-Horizon Discounted Return

For continuing tasks (or when we want to weight near-term rewards more heavily):

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The discount factor $\gamma$ ensures convergence when rewards are bounded: if $|R_t| \leq R_{\max}$, then:

$$|G_t| \leq \frac{R_{\max}}{1 - \gamma}$$

### Average Reward

For continuing tasks, an alternative objective is the average reward per step:

$$r(\pi) = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}\left[\sum_{t=1}^{T} R_t \mid \pi\right]$$

## Recursive Property of Returns

A key property is that the return satisfies a recursive relationship:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

This recursive structure is the basis for **bootstrapping** in temporal difference methods and the **Bellman equations** for value functions.

## Designing Reward Functions

Effective reward design is crucial and often challenging. Poor reward functions lead to unintended behavior.

### Principles of Good Reward Design

1. **Reward what you want, not how to achieve it**: Reward the goal state, not intermediate steps (unless necessary for learning)
2. **Avoid reward hacking**: Anticipate ways the agent might exploit the reward function
3. **Keep it sparse vs. dense**: Dense rewards speed learning but may bias behavior; sparse rewards are more general but harder to learn from
4. **Align incentives**: Ensure that maximizing the reward truly corresponds to the desired behavior

### Common Reward Structures

| Structure | Description | Example |
|-----------|-------------|---------|
| **Sparse** | Reward only at goal | +1 at checkmate, 0 otherwise |
| **Dense** | Reward at every step | Distance reduction to goal |
| **Shaped** | Engineered guidance signals | Potential-based shaping |
| **Intrinsic** | Self-generated curiosity | Prediction error as reward |

### Reward Shaping

**Potential-based reward shaping** adds a supplementary reward without changing the optimal policy:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

where $\Phi: \mathcal{S} \to \mathbb{R}$ is a potential function. This is the only form of shaping guaranteed to preserve the optimal policy (Ng et al., 1999).

## Challenges and Limitations

### Multi-Objective Problems

Real-world problems often have multiple objectives. The scalar reward forces a tradeoff:

$$R_t = w_1 \cdot r_t^{(1)} + w_2 \cdot r_t^{(2)} + \cdots$$

The weights $w_i$ encode preferences, but choosing them correctly is itself a challenge.

### Reward Misspecification

When the reward function doesn't capture the true objective:

- **Reward hacking**: The agent finds unexpected ways to maximize reward without achieving the intended goal
- **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure"
- **Side effects**: The agent may cause unintended harm while pursuing the specified reward

### Sparse Rewards

When rewards are very sparse (e.g., only at the end of a long episode), learning becomes extremely difficult due to the **credit assignment problem**—the agent must determine which of its many actions contributed to the final outcome.

## Financial Reward Design

Designing reward functions for financial applications requires careful consideration:

### Common Financial Reward Functions

| Reward Function | Formula | Properties |
|----------------|---------|------------|
| **Simple return** | $r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | Easy to compute, ignores risk |
| **Log return** | $r_t = \ln\frac{P_t}{P_{t-1}}$ | Time-additive, better for compounding |
| **Risk-adjusted** | $r_t = \frac{\mu_t}{\sigma_t}$ (Sharpe-like) | Balances return and risk |
| **Differential Sharpe** | $r_t = \frac{\partial S_t}{\partial \eta}$ | Incremental Sharpe ratio update |
| **Drawdown-penalized** | $r_t = R_t - \lambda \cdot \text{DD}_t$ | Penalizes drawdowns |

### The Differential Sharpe Ratio

Moody & Saffell (2001) proposed using the **differential Sharpe ratio** as a reward signal:

$$D_t = \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$

where $A_t$ and $B_t$ are exponential moving averages of the first and second moments of returns:

$$A_t = A_{t-1} + \eta \Delta A_t, \quad \Delta A_t = R_t - A_{t-1}$$
$$B_t = B_{t-1} + \eta \Delta B_t, \quad \Delta B_t = R_t^2 - B_{t-1}$$

This provides a dense reward signal that directly optimizes the Sharpe ratio.

### Transaction Cost Integration

A practical reward function for trading should include transaction costs:

$$r_t = \sum_i w_{i,t} \cdot r_{i,t} - c \sum_i |w_{i,t} - w_{i,t-1}|$$

where $w_{i,t}$ are portfolio weights, $r_{i,t}$ are asset returns, and $c$ is the transaction cost rate.

## Summary

The reward hypothesis is the foundational assumption that enables the mathematical framework of RL. While powerful and general, it requires careful reward function design to avoid misspecification. In financial applications, reward design must balance multiple objectives—returns, risk, transaction costs, and regulatory constraints—within a single scalar signal. The recursive structure of returns enables efficient algorithms through bootstrapping and the Bellman equations.
