# 32.2.1 MDP Fundamentals

## Definition

A **Markov Decision Process (MDP)** is the mathematical framework that formalizes sequential decision-making under uncertainty. An MDP is defined by the tuple:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

where:

- $\mathcal{S}$: **State space** — the set of all possible states
- $\mathcal{A}$: **Action space** — the set of all possible actions (may depend on state: $\mathcal{A}(s)$)
- $P$: **Transition function** — $P(s'|s,a) = \Pr\{S_{t+1}=s' \mid S_t=s, A_t=a\}$
- $R$: **Reward function** — $R(s,a,s') = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a, S_{t+1}=s']$
- $\gamma \in [0,1]$: **Discount factor**

## The Markov Property

The defining feature of an MDP is the **Markov property**:

$$\Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0\} = \Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t, A_t\}$$

The future is conditionally independent of the past given the present state and action. The state $S_t$ is a **sufficient statistic** for the history.

### Implications

1. **Memorylessness**: The optimal decision depends only on the current state
2. **Stationarity**: Transition probabilities don't change over time
3. **State sufficiency**: The state captures all information needed for future predictions

### When the Markov Property Holds

| Scenario | Markov? | Reason |
|----------|---------|--------|
| Chess (board position) | Yes | Full board state determines outcomes |
| Stock trading (price only) | No | Future depends on unobserved factors |
| Stock trading (price + features) | Approximately | Sufficient features make it partially Markov |
| Robot with full sensors | Yes | Full sensor state captures configuration |

### Handling Non-Markov Environments

- **State augmentation**: Include last $k$ observations in state
- **Recurrent architectures**: LSTMs/GRUs to learn history summaries
- **Belief states**: Probability distribution over hidden states (POMDP)

## The Four-Argument Dynamics Function

The most complete specification of environment dynamics is:

$$p(s', r | s, a) = \Pr\{S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a\}$$

This must satisfy:

$$\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r | s, a) = 1 \quad \text{for all } s \in \mathcal{S}, a \in \mathcal{A}(s)$$

### Derived Quantities

**State transition probabilities:**

$$P(s' | s, a) = \sum_{r \in \mathcal{R}} p(s', r | s, a)$$

**Expected reward:**

$$R(s, a) = \mathbb{E}[R_{t+1} | S_t = s, A_t = a] = \sum_{r} r \sum_{s'} p(s', r | s, a)$$

**Expected reward for transition:**

$$R(s, a, s') = \frac{\sum_{r} r \cdot p(s', r | s, a)}{P(s' | s, a)}$$

## Finite vs. Continuous MDPs

| Type | State Space | Action Space | Representation |
|------|------------|-------------|----------------|
| Finite MDP | Discrete, finite | Discrete, finite | Transition matrices |
| Continuous State | $\mathbb{R}^n$ | Discrete | Density functions |
| Continuous Action | Discrete | $\mathbb{R}^m$ | Density functions |
| Fully Continuous | $\mathbb{R}^n$ | $\mathbb{R}^m$ | Function approximation required |

For finite MDPs, dynamics can be represented as transition matrices $\mathbf{P}_a \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ for each action $a$.

## Financial Application: Portfolio Management as MDP

$$\mathcal{M}_{\text{portfolio}} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State**: $s_t = (\mathbf{p}_t, \mathbf{w}_t, \mathbf{x}_t)$ — prices, current weights, market features
- **Action**: $a_t = \mathbf{w}_{t+1}^{\text{target}}$ — target portfolio weights with $\sum_i w_i = 1$
- **Transition**: Determined by stochastic, non-stationary market dynamics
- **Reward**: $r_{t+1} = \sum_i w_{i,t} \cdot r_{i,t+1} - c \sum_i |w_{i,t+1} - w_{i,t}|$
- **Discount**: $\gamma$ encodes investment horizon preference

The Markov property is only approximately satisfied since market dynamics depend on unobservable factors. Enriching the state with technical indicators, sentiment features, and macroeconomic variables helps.

## Summary

MDPs provide the rigorous mathematical foundation for RL. The Markov property enables tractable algorithms by ensuring that optimal decisions can be made based solely on current state. Understanding MDP formulation is essential for correctly modeling real-world problems as RL tasks.
