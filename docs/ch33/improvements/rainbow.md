# 33.2.6 Rainbow

## Overview

**Rainbow** (Hessel et al., 2018) combines six orthogonal improvements to DQN into a single integrated agent:

1. **Double DQN** — Reduces overestimation bias
2. **Dueling Architecture** — Separate value and advantage streams
3. **Prioritized Experience Replay** — Focus on informative transitions
4. **Multi-step Returns** (N-step) — Better credit assignment
5. **Distributional RL** (C51) — Model full return distribution
6. **Noisy Networks** — Learned exploration (replaces ε-greedy)

## Architecture

Rainbow uses a **Dueling** network with **NoisyLinear** layers that outputs a **categorical distribution** (C51) over returns:

```
Input: state
  → Shared feature layers
  → Split into Value and Advantage streams
  → Each stream uses NoisyLinear layers
  → Value stream: (n_atoms,) — distribution of V(s)
  → Advantage stream: (action_dim × n_atoms) — distribution of A(s, a)
  → Combine: Z(s, a) = V(s) + A(s, a) - mean(A)  [distributional]
  → Softmax over atoms
```

## Training Algorithm

```
For each training step:
  1. Sample prioritized batch from replay buffer (PER)
  2. Compute n-step distributional targets (C51 + n-step)
  3. Use Double DQN for action selection in target
  4. Compute distributional cross-entropy loss with IS weights (PER)
  5. Update priorities based on loss
  6. No ε-greedy needed (NoisyNets handle exploration)
```

## N-step Target Integration

Rainbow uses n-step returns with C51:

$$R^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$$
$$y_t = R^{(n)}_t + \gamma^n Z_{\theta^-}(s_{t+n}, a^*)$$

This requires storing $n$ consecutive transitions and computing truncated returns.

## Ablation Results

Hessel et al. (2018) performed a comprehensive ablation study. Removing each component individually:

| Removed Component | Median Score Drop |
|-------------------|------------------|
| Prioritized Replay | Largest drop |
| Multi-step Returns | Large drop |
| Distributional (C51) | Moderate drop |
| Noisy Networks | Moderate drop |
| Dueling | Small drop |
| Double DQN | Small drop |

**Key finding**: Prioritized replay and multi-step returns contributed most to Rainbow's performance.

## Practical Considerations

### Hyperparameters

| Parameter | Rainbow Value |
|-----------|--------------|
| N-step | 3 |
| Atoms (C51) | 51 |
| $V_\text{min}, V_\text{max}$ | -10, 10 |
| PER $\alpha$ | 0.5 |
| PER $\beta$ | 0.4 → 1.0 |
| NoisyNet $\sigma_0$ | 0.5 |
| Learning rate | 6.25 × 10⁻⁵ |
| Batch size | 32 |
| Target update freq | 8000 |

### Computational Cost

Rainbow is approximately 2× the compute of DQN per training step due to:
- Distributional output (51 atoms × actions vs. 1 × actions)
- Sum tree operations for PER
- NoisyLinear forward passes

However, the improved sample efficiency often more than compensates.

## Simplified Rainbow

For practical applications, a simplified version combining the 3 most impactful components often suffices:
- **Prioritized Replay** + **Multi-step Returns** + **Double DQN**

This captures ~80% of Rainbow's gains with significantly less implementation complexity.
