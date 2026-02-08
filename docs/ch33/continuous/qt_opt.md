# 33.4.2 QT-Opt

## Overview

**QT-Opt** (Kalashnikov et al., 2018) extends Q-learning to continuous action spaces using the **Cross-Entropy Method (CEM)** to approximately solve the $\arg\max$ over actions. Originally developed at Google for robotic grasping, QT-Opt achieved a 96% grasp success rate on previously unseen objects.

## Core Idea

Instead of restricting Q to a special form (like NAF's quadratic), QT-Opt uses an arbitrary neural network $Q_\theta(s, a)$ and optimizes actions via sampling:

$$a^* = \arg\max_a Q_\theta(s, a) \approx \text{CEM}(Q_\theta(s, \cdot))$$

## Cross-Entropy Method for Action Optimization

CEM iteratively refines a sampling distribution to find high-Q actions:

```
Initialize: μ = 0, σ = 1 (or from prior)
For iteration i = 1 to K:
    1. Sample N actions: {a_j} ~ N(μ, diag(σ²))
    2. Evaluate: Q_j = Q(s, a_j) for each sample
    3. Select top M samples (elite set) by Q-value
    4. Refit: μ = mean(elite), σ = std(elite)
Return: μ (or best sample)
```

### CEM Hyperparameters

| Parameter | Typical value | Description |
|-----------|--------------|-------------|
| N (samples) | 64 | Number of action samples per iteration |
| K (iterations) | 2–3 | CEM iterations |
| M (elite) | 6–10 | Top samples for refitting |

## Architecture

QT-Opt uses a Q-network that takes both state and action as inputs:

```
Input: [state, action] concatenated
  → Linear(state_dim + action_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 1) → Q-value
```

## Training

Standard DQN-style training with:
- **Experience replay**: Large distributed replay buffer
- **Target network**: Polyak averaging
- **Bellman target**: $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ where the max is computed via CEM

## Advantages over Actor-Critic

| Aspect | QT-Opt | Actor-Critic (DDPG/TD3) |
|--------|--------|------------------------|
| Action optimization | CEM (sampling) | Learned actor network |
| Multimodal actions | Yes (CEM can find multiple modes) | No (single deterministic actor) |
| Architecture | Single Q-network | Q-network + actor network |
| Training stability | More stable (no actor gradients) | Can be unstable |
| Computation | CEM overhead per action | Single forward pass |

## Limitations

- **Computational cost**: CEM requires multiple Q-network evaluations per action (N × K per step)
- **Scalability**: CEM becomes less efficient in very high-dimensional action spaces (>20D)
- **Online latency**: Multiple CEM iterations per action can be slow for real-time applications

## Finance Application

QT-Opt is well-suited for portfolio optimization where:
- Actions are continuous portfolio weights
- The value landscape may be multimodal (multiple good allocation strategies)
- CEM can incorporate constraints (e.g., weight bounds, sector limits) by projecting samples
