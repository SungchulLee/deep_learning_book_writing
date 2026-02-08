# 33.4.1 Normalized Advantage Functions (NAF)

## The Continuous Action Challenge

Standard DQN requires $\arg\max_a Q(s, a)$ for action selection, which is trivial for discrete actions (enumerate all) but intractable for continuous action spaces. **Normalized Advantage Functions (NAF)** (Gu et al., 2016) solve this by parameterizing Q as a quadratic function of actions, making the argmax analytically computable.

## NAF Decomposition

NAF decomposes Q into value and a quadratic advantage:

$$Q(s, a) = V(s) + A(s, a)$$

where the advantage is a negative-definite quadratic:

$$A(s, a) = -\frac{1}{2}(a - \mu(s))^T P(s) (a - \mu(s))$$

Here:
- $V(s)$: State-value function (scalar)
- $\mu(s)$: Optimal action for state $s$ (vector, same dimension as action space)
- $P(s) = L(s) L(s)^T$: State-dependent positive-definite matrix
- $L(s)$: Lower-triangular matrix with positive diagonal (from network output)

## Key Properties

1. **Closed-form argmax**: Since $A(s, a) \leq 0$ with equality at $a = \mu(s)$:
   $$\arg\max_a Q(s, a) = \mu(s)$$
   No optimization required—the network directly outputs the optimal action.

2. **Closed-form max**: $\max_a Q(s, a) = V(s)$

3. **Exploration**: Add Gaussian noise to $\mu(s)$ during training:
   $$a = \mu(s) + \mathcal{N}(0, \sigma^2 I)$$

## Architecture

```
Input: state s
  → Shared hidden layers
  → Three output heads:
      V(s):  Linear → scalar
      μ(s):  Linear → action_dim (optimal action)
      L(s):  Linear → action_dim * (action_dim + 1) / 2
             → Reshape to lower-triangular
             → exp(diagonal) to ensure positive definiteness
  → P = L @ L^T
  → A(s,a) = -0.5 * (a-μ)^T P (a-μ)
  → Q(s,a) = V + A
```

## Training

NAF uses the same DQN training loop with experience replay and target networks:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma V_{\theta^-}(s') - Q_\theta(s, a)\right)^2\right]$$

Note: the target simplifies to $r + \gamma V_{\theta^-}(s')$ since $\max_a Q(s', a) = V(s')$.

## Limitations

- **Unimodal**: The quadratic form assumes a single optimal action per state. Cannot represent multi-modal action distributions
- **Limited expressiveness**: Quadratic advantage may not capture complex action-value landscapes
- **Scaling**: The $P$ matrix grows as $O(d^2)$ with action dimension $d$
- **Modern alternatives**: DDPG, TD3, and SAC have largely superseded NAF for continuous control

## Finance Relevance

NAF is useful for continuous portfolio allocation where:
- Action = portfolio weights (continuous)
- The quadratic form naturally captures the mean-variance structure of portfolio optimization
- The positive-definite $P$ matrix can be interpreted as a state-dependent risk penalty
