# 32.6.2 TD(0)

## Algorithm

**TD(0)** (or one-step TD) is the simplest TD method for estimating $V_\pi$:

```
Initialize V(s) arbitrarily, V(terminal) = 0

For each episode:
    S = initial state
    For each step:
        A = action from π(S)
        Take A, observe R, S'
        V(S) ← V(S) + α [R + γ V(S') - V(S)]
        S ← S'
    Until S is terminal
```

## The TD Error

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

This measures the surprise: how much better (or worse) the transition was than expected. The sign tells us:

- $\delta_t > 0$: The outcome was better than expected → increase $V(S_t)$
- $\delta_t < 0$: The outcome was worse than expected → decrease $V(S_t)$

## Convergence

TD(0) converges to $V_\pi$ under standard conditions:
1. Step sizes satisfy Robbins-Monro: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$
2. All states are visited infinitely often

With constant step size $\alpha$, TD(0) converges to a neighborhood of $V_\pi$.

## TD(0) as Stochastic Approximation

TD(0) is performing stochastic approximation of the Bellman equation:

$$V_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s]$$

Each update uses a single sample $(R_{t+1}, S_{t+1})$ instead of the full expectation.

## Batch TD(0)

Given a fixed batch of experience, repeatedly apply TD(0) updates until convergence. Batch TD(0) converges to the **maximum likelihood** value function — the $V$ that would be exactly correct for the maximum likelihood model of the MDP.

## Advantages Over MC

1. **Online learning**: Updates after every step, not episode end
2. **Continuing tasks**: Works without episode boundaries
3. **Lower variance**: Bootstrapping reduces variance
4. **Faster convergence**: Empirically faster for many problems

## Financial Application

TD(0) for real-time portfolio evaluation:
- After each trading period, observe the reward (return) and new market state
- Update the estimated strategy value immediately
- No need to wait for the end of the evaluation horizon
- Enables adaptive, streaming evaluation of trading strategies
