# 34.3.4 PPO Implementation

## Introduction

This section provides a complete, production-quality PPO implementation following the "37 Implementation Details" best practices (Huang et al., 2022). The implementation covers vectorized environments, proper advantage computation, the full training loop, and critical implementation details that significantly affect performance.

## Critical Implementation Details

### 1. Vectorized Environments
Run $N$ environment copies in parallel to collect diverse data efficiently. Each environment auto-resets on episode termination.

### 2. Advantage Normalization
Normalize advantages at the minibatch level (not the full batch) for stable gradient magnitudes:
$$\hat{A}_t = \frac{A_t - \mu_\text{mb}}{\sigma_\text{mb} + \epsilon}$$

### 3. Value Function Clipping (Optional)
Clip value predictions to prevent large value updates:
$$V_\text{clip} = V_\text{old} + \text{clip}(V_\theta - V_\text{old}, -\epsilon, \epsilon)$$
$$L_V = \max\left[(V_\theta - V_\text{target})^2, (V_\text{clip} - V_\text{target})^2\right]$$

### 4. Learning Rate Annealing
Linear decay of learning rate to zero over training improves final performance.

### 5. Global Gradient Clipping
Clip the global gradient norm (not per-parameter) to 0.5.

### 6. Orthogonal Initialization
Initialize weights with orthogonal matrices:
- Hidden layers: gain $\sqrt{2}$
- Policy head: gain $0.01$
- Value head: gain $1.0$

### 7. Observation Normalization
Running mean/variance normalization of observations with clipping to $[-10, 10]$.

### 8. Reward Normalization
Normalize rewards by the running standard deviation (but not the mean) to handle different reward scales.

## Training Loop Structure

```
Total updates = total_timesteps / (n_envs × n_steps)

For each update:
    # Rollout phase
    For t = 0, ..., n_steps-1:
        action, logprob, value = agent(obs)
        obs, reward, done = envs.step(action)
        store(obs, action, reward, done, logprob, value)
    
    # Bootstrap last value
    last_value = agent.get_value(last_obs)
    
    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones, last_value)
    
    # Flatten batch
    batch = flatten(rollout)  # shape: (n_envs × n_steps, ...)
    
    # Optimization phase
    For epoch = 1, ..., n_epochs:
        indices = random_permutation(batch_size)
        For each minibatch of indices:
            ratio = exp(new_logprob - old_logprob)
            clipped_loss = min(ratio × adv, clip(ratio) × adv)
            value_loss = MSE(value, returns)
            entropy = policy.entropy()
            loss = -clip_loss + c_v × value_loss - c_e × entropy
            optimizer.step(loss)
    
    # Anneal learning rate
```

## Performance Benchmarks

With proper implementation, PPO achieves:
- **CartPole-v1**: Solved (~500 reward) in < 100K steps
- **LunarLander-v2**: Solved (~200 reward) in ~500K steps
- **Atari (Breakout)**: ~400+ score in ~10M frames

## Common Implementation Pitfalls

1. **Not normalizing advantages per minibatch**: Leads to unstable training
2. **Wrong log probability computation**: Must sum log probs across action dimensions for continuous
3. **Forgetting to detach old log probs**: Old log probs must not carry gradients
4. **Missing auto-reset**: Environments must auto-reset on termination
5. **Incorrect GAE at episode boundaries**: Must mask with done flags
6. **Not annealing learning rate**: Final performance suffers without annealing

## Summary

A correct PPO implementation requires careful attention to many details beyond the core clipped objective. The combination of vectorized environments, GAE, proper normalization, and learning rate scheduling creates a robust and efficient training system.
