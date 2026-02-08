# 32.1.2 Agent-Environment Interface

## The Interaction Loop

The agent-environment interface is the central abstraction in reinforcement learning. At each discrete time step $t = 0, 1, 2, \ldots$, the interaction proceeds as follows:

1. The agent observes the current **state** $S_t \in \mathcal{S}$
2. The agent selects an **action** $A_t \in \mathcal{A}(S_t)$ according to its **policy** $\pi$
3. The environment transitions to a new state $S_{t+1}$ and emits a **reward** $R_{t+1} \in \mathbb{R}$
4. The process repeats

This produces a **trajectory** (or history):

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \ldots$$

Note the convention: the reward $R_{t+1}$ is received after taking action $A_t$ in state $S_t$, and it corresponds to the transition from $S_t$ to $S_{t+1}$.

## Formal Definition

The agent-environment interface is characterized by:

- **State space** $\mathcal{S}$: The set of all possible states
- **Action space** $\mathcal{A}$: The set of all possible actions (may depend on state)
- **Transition function** $p(s', r | s, a)$: The probability of transitioning to state $s'$ with reward $r$, given state $s$ and action $a$
- **Policy** $\pi(a|s)$: The probability that the agent takes action $a$ in state $s$

The transition function defines the **dynamics** of the environment:

$$p(s', r | s, a) = \Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a\}$$

for all $s, s' \in \mathcal{S}$, $a \in \mathcal{A}(s)$, and $r \in \mathbb{R}$.

## The Agent-Environment Boundary

The boundary between agent and environment is not the same as the physical boundary of a robot or organism. The key principle is:

> **Anything that cannot be changed arbitrarily by the agent is considered part of the environment.**

For example:
- In a chess-playing agent, the rules of chess are part of the environment
- A robot's motors might be considered part of the environment (the agent sends signals to them but cannot change their physics)
- In portfolio management, the financial market is the environment; the agent's internal decision logic is the agent

### What Constitutes the State?

A good state representation should contain enough information to make future predictions. The state might include:

- **Physical state**: Position, velocity, configuration
- **Information state**: Sufficient statistics of the history
- **Observation**: What the agent actually perceives (may be a partial view of the true state)

## Episodic vs. Continuing Tasks

### Episodic Tasks

In episodic tasks, the interaction naturally breaks into **episodes**, each ending in a **terminal state**. After reaching a terminal state, the environment resets.

Examples:
- A game of chess (ends in win/loss/draw)
- A single trading day
- A robot reaching its goal or falling

We use $\mathcal{S}^+$ to denote the full state space including terminal states.

### Continuing Tasks

In continuing tasks, the interaction goes on indefinitely without natural endpoints.

Examples:
- An ongoing process control task
- A continuously trading market-making agent
- Server resource management

For continuing tasks, we typically use **discounting** to ensure that the total return remains finite.

### Unified Notation

We can unify both cases by treating episodic tasks as special cases where the terminal state transitions only to itself with zero reward (an **absorbing state**).

## Reward Signal

The reward $R_{t+1}$ is a single scalar value. Despite its simplicity, this signal must encode the **goal** of the agent. The reward hypothesis (covered in the next section) states that all goals can be expressed as maximization of cumulative scalar reward.

### Properties of the Reward Signal

- **Scalar**: Reduces multi-objective problems to a single number
- **Immediate**: Received at each time step (though it may be zero for many steps)
- **External**: Determined by the environment, not the agent
- **Informative**: Should provide useful learning signal (reward shaping can help)

## Policy

A **policy** $\pi$ defines the agent's behavior. It can be:

- **Deterministic**: $a = \pi(s)$ — a direct mapping from state to action
- **Stochastic**: $\pi(a|s) = \Pr\{A_t = a \mid S_t = s\}$ — a probability distribution over actions given the state

Stochastic policies are important for:
1. **Exploration**: Trying different actions to discover their effects
2. **Mixed strategies**: In adversarial settings (e.g., game theory)
3. **Optimality**: Some environments require stochastic optimal policies

## Financial Application: Trading Agent Interface

Consider a simple stock trading agent:

```
State:  s_t = [price_t, volume_t, position_t, cash_t, indicator_t, ...]
Action: a_t ∈ {buy, hold, sell} (or continuous position sizing)

Agent-Environment Loop:
  1. Agent observes market state s_t
  2. Agent selects trading action a_t based on policy π
  3. Market evolves to s_{t+1} (next trading period)
  4. Agent receives reward r_{t+1} (e.g., portfolio return - costs)
```

| Component | Implementation |
|-----------|---------------|
| State | Feature vector of market data + portfolio state |
| Action | Discrete (buy/hold/sell) or continuous (target weight) |
| Reward | Period return, Sharpe contribution, or P&L |
| Episode | Trading period (day, week, quarter) |
| Terminal | End of evaluation period or drawdown limit |

## Summary

The agent-environment interface provides a clean, general framework for sequential decision problems. The agent observes states, takes actions, and receives rewards. The environment encapsulates everything the agent cannot directly control. This interface supports both episodic and continuing tasks and forms the foundation for all RL algorithms discussed in subsequent sections.
