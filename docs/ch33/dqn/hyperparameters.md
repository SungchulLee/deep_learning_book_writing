# 33.1.5 DQN Hyperparameters

## Overview

DQN performance is highly sensitive to hyperparameter choices. This section provides a comprehensive guide to DQN hyperparameters, their typical ranges, interactions, and tuning strategies.

## Core Hyperparameters

### Learning Rate ($\alpha$)

| Setting | Value | Notes |
|---------|-------|-------|
| Atari (original DQN) | $2.5 \times 10^{-4}$ | With RMSProp |
| Simple environments | $10^{-3}$ | With Adam |
| Complex / large networks | $10^{-4}$ to $3 \times 10^{-4}$ | With Adam |

The learning rate interacts strongly with batch size, target update frequency, and network size. Larger batches may tolerate higher LR, while more frequent target updates require smaller LR to maintain stability.

### Discount Factor ($\gamma$)

| $\gamma$ | Effective horizon | Use case |
|-----------|------------------|----------|
| 0.95 | ~20 steps | Short-horizon tasks |
| 0.99 | ~100 steps | Standard RL tasks |
| 0.999 | ~1000 steps | Long-horizon planning |

The effective horizon is approximately $\frac{1}{1 - \gamma}$. For daily trading with annual objectives, $\gamma = 0.999$ corresponds to roughly 1000 trading days (~4 years), while $\gamma = 0.99$ captures ~100 days (~5 months).

### Batch Size ($B$)

| Value | Trade-off |
|-------|-----------|
| 32 | Low compute, higher variance |
| 64 | Good default |
| 128–256 | Lower variance, more GPU utilization |
| 512+ | Diminishing returns, may hurt exploration |

### Replay Buffer Capacity ($N$)

| Environment | Typical $N$ |
|-------------|-------------|
| CartPole / simple | $10^4$ – $5 \times 10^4$ |
| Atari | $10^5$ – $10^6$ |
| Robotics / complex | $10^6$ |
| Financial trading | $10^5$ – $5 \times 10^5$ |

Buffer too small leads to correlated samples and forgetting; too large wastes memory and includes stale data.

### Target Network Update Frequency ($C$)

| Environment | Hard update $C$ | Soft update $\tau$ |
|-------------|----------------|-------------------|
| Simple | 100–500 | 0.005–0.01 |
| Atari | 1,000–10,000 | 0.001–0.005 |
| Complex | 5,000–10,000 | 0.001 |

## Exploration Hyperparameters

### Epsilon Schedule

The standard linear annealing schedule:

$$\epsilon(t) = \max\left(\epsilon_\text{end},\; \epsilon_\text{start} - \frac{\epsilon_\text{start} - \epsilon_\text{end}}{\text{decay\_steps}} \cdot t\right)$$

| Parameter | Typical value |
|-----------|--------------|
| $\epsilon_\text{start}$ | 1.0 |
| $\epsilon_\text{end}$ | 0.01 – 0.1 |
| decay_steps | 10K – 1M |

### Minimum Buffer Size Before Training

Collect random transitions before starting gradient updates:
- **Too small**: Initial updates are on non-diverse data
- **Too large**: Wastes time on random exploration
- **Typical**: 1,000 – 50,000 (10% of buffer capacity is a reasonable rule of thumb)

## Network Architecture Hyperparameters

### Hidden Layer Dimensions

| Problem complexity | Architecture |
|-------------------|-------------|
| Low-dimensional state (< 10) | 2 layers, 64–128 units |
| Medium (10–100) | 2–3 layers, 128–256 units |
| High-dimensional / image | CNN backbone + 512 FC |

### Activation Functions

- **ReLU**: Standard default, works well in most cases
- **LeakyReLU**: Can help with dying neuron problems
- **ELU/GELU**: Marginal improvements in some settings

## Hyperparameter Interactions

Several hyperparameters interact in non-obvious ways:

1. **LR × Batch Size**: Larger batches produce lower-variance gradients; LR can be scaled proportionally (linear scaling rule)
2. **LR × Target Update Freq**: Fast target updates + high LR → instability; slow updates + low LR → slow convergence
3. **Buffer Size × $\gamma$**: High $\gamma$ (long horizon) benefits from larger buffers containing diverse long-term outcomes
4. **$\epsilon$ decay × Buffer size**: Slow $\epsilon$ decay fills the buffer with more diverse experiences

## Tuning Strategy

### Phase 1: Sanity Check
1. Verify environment setup with random agent
2. Start with known-good defaults (see table below)
3. Train for enough episodes to see learning signal

### Phase 2: Coarse Search
1. Sweep learning rate: $\{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$
2. Sweep target update freq: $\{100, 500, 1000\}$
3. Fix other hyperparameters at defaults

### Phase 3: Fine-Tuning
1. Fix best LR and target update
2. Tune $\epsilon$ schedule and buffer size
3. Try Huber loss vs MSE
4. Adjust network architecture

### Recommended Defaults

| Hyperparameter | CartPole | Atari | Finance |
|---------------|----------|-------|---------|
| Learning rate | $10^{-3}$ | $2.5 \times 10^{-4}$ | $10^{-4}$ |
| $\gamma$ | 0.99 | 0.99 | 0.999 |
| Batch size | 64 | 32 | 128 |
| Buffer capacity | 50,000 | 1,000,000 | 200,000 |
| Target update $C$ | 200 | 10,000 | 1,000 |
| $\epsilon_\text{end}$ | 0.01 | 0.01 | 0.05 |
| $\epsilon$ decay steps | 5,000 | 1,000,000 | 50,000 |
| Loss function | Huber | Huber | MSE |
| Optimizer | Adam | RMSProp | Adam |
| Gradient clip | 10.0 | 10.0 | 1.0 |

## Sensitivity Analysis

The most sensitive hyperparameters (ranked by impact):
1. **Learning rate** — Most impactful; wrong by 10× often means no learning
2. **Target update frequency** — Too fast → divergence; too slow → stale targets
3. **Epsilon schedule** — Insufficient exploration → suboptimal policy
4. **Discount factor** — Mismatched horizon → myopic or unstable behavior
5. **Network size** — Underfitting or overfitting
6. **Batch size** — Moderate impact on stability and speed
7. **Buffer size** — Least sensitive if within reasonable range
