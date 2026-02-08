# 32.3.3 Bellman Equations

## Overview

The **Bellman equations** express value functions recursively: the value of a state equals the immediate reward plus the discounted value of the next state. These equations are the mathematical foundation of nearly all RL algorithms.

## Bellman Equation for $V_\pi$

$$V_\pi(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')\right]$$

Expanding step by step:

1. The agent is in state $s$
2. It selects action $a$ with probability $\pi(a|s)$
3. It receives expected reward $R(s,a)$
4. It transitions to state $s'$ with probability $P(s'|s,a)$
5. The value from $s'$ onward is $V_\pi(s')$, discounted by $\gamma$

### Matrix Form

$$\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_\pi$$

This is a system of $|\mathcal{S}|$ linear equations with $|\mathcal{S}|$ unknowns.

**Closed-form solution:**

$$\mathbf{v}_\pi = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi$$

The matrix $(\mathbf{I} - \gamma \mathbf{P}_\pi)$ is always invertible for $\gamma < 1$ because its eigenvalues have modulus at least $1 - \gamma > 0$.

## Bellman Equation for $Q_\pi$

$$Q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q_\pi(s', a')$$

Or equivalently, using $V_\pi$:

$$Q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')$$

## Bellman Backup Diagrams

The Bellman equations can be visualized as **backup diagrams** showing information flow:

### V-function backup:

```
     (s)          ← State node: sum over actions weighted by π(a|s)
    / | \
  a₁  a₂  a₃     ← Action nodes
  |   |   |
 s'₁ s'₂ s'₃     ← Next state nodes: sum over transitions P(s'|s,a)
```

The value of the root (state) is computed from the leaves (next states) — hence "backup."

### Q-function backup:

```
   (s,a)          ← State-action node: sum over transitions
   / | \
  s'₁ s'₂ s'₃    ← Next state nodes
  |   |   |
 a'₁ a'₂ a'₃     ← Next action nodes: sum over π(a'|s')
```

## Bellman Operator

The Bellman equation defines a **Bellman operator** $\mathcal{T}_\pi$:

$$(\mathcal{T}_\pi V)(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right]$$

$V_\pi$ is the **fixed point** of this operator: $\mathcal{T}_\pi V_\pi = V_\pi$.

### Contraction Mapping Property

$\mathcal{T}_\pi$ is a **$\gamma$-contraction** in the max norm:

$$\|\mathcal{T}_\pi V_1 - \mathcal{T}_\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$$

By the **Banach fixed-point theorem**, this guarantees:

1. **Existence and uniqueness**: $V_\pi$ is the unique fixed point
2. **Convergence**: Repeated application converges: $\mathcal{T}_\pi^k V \to V_\pi$ as $k \to \infty$
3. **Convergence rate**: $\|V_k - V_\pi\|_\infty \leq \gamma^k \|V_0 - V_\pi\|_\infty$

## Intuition: Why Bellman Equations Work

The Bellman equation decomposes the value of a state into:

$$\underbrace{V_\pi(s)}_{\text{Total value}} = \underbrace{R(s,a)}_{\text{Immediate reward}} + \gamma \underbrace{V_\pi(s')}_{\text{Future value}}$$

This **recursive decomposition** is the key insight that enables:

- **Dynamic programming**: Solve by iterating the Bellman equation
- **TD learning**: Update estimates toward the bootstrapped target $R + \gamma V(s')$
- **Q-learning**: Update toward $R + \gamma \max_{a'} Q(s', a')$

## Derivation from First Principles

Starting from the definition of $V_\pi$:

$$V_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$
$$= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s]$$
$$= \mathbb{E}_\pi[R_{t+1} | S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s]$$
$$= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \mathbb{E}_\pi[G_{t+1} | S_{t+1} = s']\right]$$
$$= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V_\pi(s')\right]$$

The key step uses the **Markov property**: $\mathbb{E}_\pi[G_{t+1} | S_{t+1} = s'] = V_\pi(s')$.

## Summary

The Bellman equations provide the recursive structure that makes RL computationally tractable. They decompose the value of a state into immediate reward plus discounted future value, enabling iterative algorithms. The contraction mapping property guarantees convergence, and the backup diagram provides visual intuition for information flow in value computation.
