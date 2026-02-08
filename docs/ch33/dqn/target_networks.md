# 33.1.3 Target Networks

## The Instability Problem

Without a target network, the DQN loss is:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a)\right)^2\right]$$

Both the prediction $Q_\theta(s, a)$ and the target $r + \gamma \max_{a'} Q_\theta(s', a')$ depend on the same parameters $\theta$. Each gradient step shifts both, creating a "moving target" problem that can cause oscillation or divergence.

## Solution: Separate Target Network

DQN maintains two networks:

1. **Online network** $Q_\theta$: Updated every gradient step
2. **Target network** $Q_{\theta^-}$: Updated infrequently, provides stable TD targets

The loss becomes:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right]$$

Since $\theta^-$ is held fixed during gradient updates, the targets are stable.

## Update Strategies

### Hard Update (Periodic Copy)

Every $C$ steps, copy the online network parameters to the target network:

$$\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}$$

- **Original DQN**: $C = 10{,}000$
- **Pros**: Simple, clear separation between target and online
- **Cons**: Abrupt changes in target values every $C$ steps

### Soft Update (Polyak Averaging)

Continuously blend the online parameters into the target:

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

where $\tau \ll 1$ (typically 0.001 to 0.01).

- **Pros**: Smooth target evolution, no abrupt jumps
- **Cons**: Slightly more complex; target always slightly stale
- **Common in**: DDPG, TD3, SAC, and modern actor-critic methods

### Comparison

| Aspect | Hard Update | Soft Update |
|--------|------------|-------------|
| Update frequency | Every $C$ steps | Every step |
| Target stability | Very stable between updates | Smoothly evolving |
| Implementation | Copy state dict | Weighted average |
| Typical setting | $C = 10{,}000$ | $\tau = 0.005$ |

## Why Target Networks Work

### Stability Analysis (Intuition)

Consider the fixed-point iteration view. Without a target network:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

where $\mathcal{L}$ itself depends on $\theta_t$. This is a nonlinear, non-stationary optimization—the loss surface changes every step.

With a target network (hard update every $C$ steps):
- For $C$ steps, the optimization is *supervised learning* with fixed targets
- Convergence within each period is more reliable
- The overall procedure resembles **fitted Q-iteration**

### Connection to Fitted Q-Iteration

Fitted Q-Iteration (FQI) alternates:
1. Fix target values: $y_i = r_i + \gamma \max_{a'} Q_{\theta^-}(s'_i, a')$
2. Regress: $\theta \leftarrow \arg\min_\theta \sum_i (Q_\theta(s_i, a_i) - y_i)^2$

DQN with hard target updates approximates FQI where step 2 uses $C$ gradient steps instead of full regression.

## Practical Considerations

### Initialization
Both networks should be initialized identically:
```python
target_net.load_state_dict(online_net.state_dict())
```

### Gradient Isolation
The target network should **never** receive gradients:
```python
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1)[0]
    targets = rewards + (1 - dones) * gamma * next_q
```

### Memory
Target networks double the GPU memory for Q-network parameters. For large architectures, this can be significant.

### Multiple Target Networks
Some algorithms (e.g., TD3) maintain multiple target networks for different purposes (critics, actors). The same update logic applies to each.

## Impact on Training Dynamics

Empirical observations:
- **Too frequent updates** ($C$ too small or $\tau$ too large): Targets become unstable, training may diverge
- **Too infrequent updates** ($C$ too large or $\tau$ too small): Targets become stale, slow learning
- **Sweet spot**: Depends on environment complexity and learning rate
