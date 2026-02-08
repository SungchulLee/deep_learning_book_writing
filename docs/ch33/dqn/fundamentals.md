# 33.1.1 DQN Fundamentals

## From Q-Learning to Deep Q-Networks

Classical Q-learning maintains a table $Q(s, a)$ for every state-action pair and updates it via the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

This approach fails for large or continuous state spaces. **Deep Q-Networks (DQN)** replace the Q-table with a neural network $Q_\theta(s, a)$ parameterized by $\theta$, enabling generalization across states.

## The DQN Loss Function

DQN minimizes the temporal difference (TD) error using a mean-squared error loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

where:
- $\mathcal{D}$ is the replay buffer (see Experience Replay)
- $\theta^-$ are the parameters of a separate **target network** (see Target Networks)
- The term $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ is the **TD target**

## Key Innovations

DQN introduced two critical stabilization techniques that made deep RL practical:

### 1. Experience Replay
Instead of learning from sequential transitions, DQN stores transitions $(s, a, r, s', \text{done})$ in a replay buffer and samples random mini-batches. This:
- Breaks temporal correlations between consecutive samples
- Improves data efficiency through reuse
- Reduces variance of updates

### 2. Target Networks
A separate copy of the Q-network, updated periodically (or via Polyak averaging), provides stable TD targets. Without this, the targets shift with every gradient step, causing instability and divergence.

## Architecture

The original DQN for Atari used a convolutional architecture:

```
Input: 84×84×4 (stacked frames)
  → Conv2d(4, 32, 8, stride=4) → ReLU
  → Conv2d(32, 64, 4, stride=2) → ReLU
  → Conv2d(64, 64, 3, stride=1) → ReLU
  → Flatten
  → Linear(3136, 512) → ReLU
  → Linear(512, num_actions)
Output: Q-values for each action
```

For environments with vector observations (e.g., CartPole), a simpler MLP suffices:

```
Input: state_dim
  → Linear(state_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  → Linear(128, action_dim)
Output: Q-values for each action
```

## Action Selection: ε-Greedy Policy

DQN uses an ε-greedy policy for exploration:

$$a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q_\theta(s, a) & \text{with probability } 1 - \epsilon \end{cases}$$

Epsilon is typically annealed linearly from 1.0 to a small value (e.g., 0.01) over the first portion of training.

## The Deadly Triad

DQN must address the "deadly triad" of instability in RL:
1. **Function approximation** (neural networks)
2. **Bootstrapping** (TD targets depend on current estimates)
3. **Off-policy learning** (replay buffer contains old transitions)

Experience replay and target networks mitigate but don't fully eliminate these issues, motivating the improvements covered in Section 33.2.

## Theoretical Properties

- **Universal approximation**: Neural networks can represent any Q-function given sufficient capacity
- **No convergence guarantee**: Unlike tabular Q-learning, DQN with function approximation may diverge
- **Overestimation bias**: The max operator in the TD target systematically overestimates Q-values (addressed by Double DQN)
- **Sample complexity**: DQN is significantly more sample-efficient than pure policy gradient methods due to off-policy learning

## Finance Relevance

DQN is particularly suited for financial applications with discrete action spaces:
- **Trading signals**: Buy, hold, sell decisions
- **Order execution**: Discretized volume or timing choices
- **Portfolio rebalancing**: Discrete allocation adjustments

The ability to learn from historical data (off-policy) aligns well with backtesting paradigms in quantitative finance.
