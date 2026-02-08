# 32.2.3 Transition Dynamics

## Transition Function

The **transition function** (or dynamics function) specifies how the environment evolves:

$$P(s' | s, a) = \Pr\{S_{t+1} = s' \mid S_t = s, A_t = a\}$$

For each state-action pair $(s, a)$, $P(\cdot | s, a)$ is a probability distribution over next states:

$$\sum_{s' \in \mathcal{S}} P(s' | s, a) = 1, \quad P(s' | s, a) \geq 0$$

## Matrix Representation

For finite MDPs with $|\mathcal{S}| = n$ states and $|\mathcal{A}| = m$ actions, the transition dynamics are represented by $m$ **transition matrices**:

$$\mathbf{P}_a \in \mathbb{R}^{n \times n}, \quad [\mathbf{P}_a]_{ij} = P(s_j | s_i, a)$$

Each row of $\mathbf{P}_a$ is a probability distribution (rows sum to 1). These are **stochastic matrices** (or **row-stochastic matrices**).

### Properties of Transition Matrices

1. **Non-negative**: $[\mathbf{P}_a]_{ij} \geq 0$
2. **Row-stochastic**: $\sum_j [\mathbf{P}_a]_{ij} = 1$ for all $i$
3. **Eigenvalue**: Largest eigenvalue is 1 (Perron-Frobenius theorem)
4. **Stationary distribution**: Under ergodicity, $\exists \mathbf{d}$ such that $\mathbf{d}^T \mathbf{P}_a = \mathbf{d}^T$

## Types of Transitions

### Deterministic Transitions

$$P(s' | s, a) = \begin{cases} 1 & \text{if } s' = f(s, a) \\ 0 & \text{otherwise} \end{cases}$$

The next state is a deterministic function of current state and action. Examples: deterministic games, simple control systems.

### Stochastic Transitions

Multiple next states have positive probability. Examples: dice games, noisy sensors, financial markets.

### Absorbing States

A state $s^*$ is **absorbing** if $P(s^* | s^*, a) = 1$ for all actions $a$. Terminal states in episodic tasks are modeled as absorbing states.

## Multi-Step Transitions

The probability of reaching state $s'$ from state $s$ in exactly $k$ steps under policy $\pi$:

$$P^{(k)}_\pi(s' | s) = [\mathbf{P}_\pi^k]_{s, s'}$$

where $\mathbf{P}_\pi$ is the transition matrix under policy $\pi$:

$$[\mathbf{P}_\pi]_{ij} = \sum_a \pi(a | s_i) P(s_j | s_i, a)$$

## Stationary Distribution

For an ergodic Markov chain under a fixed policy $\pi$, there exists a unique **stationary distribution** $\mathbf{d}_\pi$:

$$\mathbf{d}_\pi^T \mathbf{P}_\pi = \mathbf{d}_\pi^T, \quad \sum_s d_\pi(s) = 1$$

This represents the long-run fraction of time spent in each state, which is important for:

- Average reward formulations
- Weighting in function approximation objectives
- Convergence analysis of on-policy methods

## Model-Based vs. Model-Free

| Aspect | Model-Based | Model-Free |
|--------|------------|-----------|
| Transition knowledge | $P(s'|s,a)$ known/learned | Not required |
| Planning | Can simulate ahead | Learn from direct experience |
| Sample efficiency | Higher (can reuse model) | Lower (needs more interaction) |
| Model errors | Compounding errors in planning | No model bias |
| Examples | Dynamic programming, MCTS | Q-learning, SARSA, policy gradient |

## Financial Application: Market Transition Models

### Regime-Switching Dynamics

Financial markets exhibit **regime switching** where transition dynamics change:

$$P(s_{t+1} | s_t, a_t, \text{regime}_t)$$

Common regimes: bull market, bear market, high volatility, low volatility.

A regime-switching MDP uses a **hidden Markov model** to capture:

$$P(\text{regime}_{t+1} | \text{regime}_t) = \begin{pmatrix} p_{BB} & p_{BL} \\ p_{LB} & p_{LL} \end{pmatrix}$$

where B = bull, L = bear, and each regime has different return distributions.

### Stochastic Price Dynamics

Under geometric Brownian motion, the discretized transition is:

$$\ln(P_{t+1}/P_t) \sim \mathcal{N}\left((\mu - \sigma^2/2)\Delta t, \sigma^2 \Delta t\right)$$

This can be discretized into a transition matrix for finite-state approximations.

## Summary

Transition dynamics define how the environment responds to agent actions. The matrix representation enables efficient computation for finite MDPs, while the Markov property ensures that transitions depend only on the current state-action pair. Understanding transition structure—deterministic vs. stochastic, ergodic vs. absorbing—guides algorithm selection and analysis.
