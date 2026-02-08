# 34.2.2 Advantage Actor-Critic (A2C)

## Introduction

Advantage Actor-Critic (A2C) is the synchronous, deterministic variant of A3C (Mnih et al., 2016). It runs multiple environment instances in parallel, collects fixed-length rollouts synchronously, and performs a single gradient update using all collected data. A2C is simpler to implement than A3C while achieving comparable or better performance due to more consistent gradient estimates.

## Algorithm

### A2C Pseudocode

```
Initialize actor-critic network θ
Initialize N parallel environments
For each iteration:
    For each environment n = 1, ..., N (in parallel):
        Collect T-step rollout using π_θ
        Store (s_t^n, a_t^n, r_t^n, done_t^n) for t = 0, ..., T-1
    
    For each environment n:
        If terminal: V_target = 0
        Else: V_target = V_θ(s_T^n)  (bootstrap)
        For t = T-1, ..., 0:
            V_target = r_t^n + γ · V_target
            A_t^n = V_target - V_θ(s_t^n)
    
    Compute losses over all N × T transitions:
        L_policy = -(1/NT) Σ log π_θ(a|s) · A
        L_value  = (1/NT) Σ (V_target - V_θ(s))²
        L_entropy = -(1/NT) Σ H(π_θ(·|s))
        L_total = L_policy + c_v · L_value + c_e · L_entropy
    
    Update θ using ∇L_total
```

### Key Design Elements

1. **Synchronous parallelism**: All $N$ environments step together, producing a batch of $N \times T$ transitions
2. **Fixed-length rollouts**: Each environment collects exactly $T$ steps before the update
3. **Bootstrapped returns**: Uses $V_\theta(s_T)$ to estimate future returns beyond the rollout
4. **Combined loss**: Single optimizer handles policy, value, and entropy objectives

## Vectorized Environments

A2C leverages vectorized environments for efficient parallel data collection:

$$\text{Batch size} = N_{\text{envs}} \times T_{\text{steps}}$$

Typical values: $N_{\text{envs}} = 8\text{--}32$, $T_{\text{steps}} = 5\text{--}128$.

Benefits:
- Better GPU utilization through batched inference
- More diverse experiences per update
- Smoother gradient estimates

## Loss Components

### Policy Loss (Actor)

$$L_\text{policy} = -\frac{1}{NT}\sum_{n,t} \log \pi_\theta(a_t^n | s_t^n) \cdot \hat{A}_t^n$$

where $\hat{A}_t^n$ is the normalized advantage estimate.

### Value Loss (Critic)

$$L_\text{value} = \frac{1}{NT}\sum_{n,t} \left(V_\theta(s_t^n) - G_t^n\right)^2$$

The coefficient $c_v$ (typically $0.5$) balances the value gradient magnitude against the policy gradient.

### Entropy Bonus

$$L_\text{entropy} = -\frac{1}{NT}\sum_{n,t} \mathcal{H}(\pi_\theta(\cdot | s_t^n))$$

The coefficient $c_e$ (typically $0.01$) prevents premature convergence to deterministic policies.

## Advantage Estimation in A2C

A2C typically uses either:

1. **N-step returns**: Direct Monte Carlo return to the end of the rollout with bootstrapping
2. **GAE**: Generalized Advantage Estimation for smoother bias-variance trade-off

The standard A2C computes advantages as:

$$\hat{A}_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k} + \gamma^{T-t} V_\theta(s_T) - V_\theta(s_t)$$

## A2C vs. A3C

| Feature | A2C | A3C |
|---------|-----|-----|
| Parallelism | Synchronous | Asynchronous |
| Updates | Single batched | Per-worker |
| Gradient staleness | None | Possible |
| GPU efficiency | High (batched) | Low (sequential) |
| Implementation | Simple | Complex |
| Performance | Comparable/better | Comparable |

A2C is generally preferred on GPU hardware because batched inference is more efficient than the asynchronous worker model of A3C.

## Hyperparameters

| Parameter | Typical Value | Role |
|-----------|--------------|------|
| Learning rate | $2.5 \times 10^{-4}$ to $7 \times 10^{-4}$ | Step size |
| $\gamma$ | 0.99 | Discount factor |
| $N_\text{envs}$ | 8-32 | Parallel environments |
| $T_\text{steps}$ | 5-128 | Rollout length |
| $c_v$ | 0.5 | Value loss coefficient |
| $c_e$ | 0.01 | Entropy coefficient |
| Max grad norm | 0.5 | Gradient clipping |

## Summary

A2C provides a clean, efficient baseline for on-policy policy gradient methods. Its synchronous design simplifies implementation while maintaining the benefits of parallel data collection. The combined policy-value-entropy loss enables end-to-end training of the actor-critic network.
