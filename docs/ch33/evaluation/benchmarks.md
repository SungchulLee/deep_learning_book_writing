# 33.6.2 Benchmarks

## Standard RL Benchmarks

### Classic Control (Gymnasium)
Simple environments for algorithm development and debugging:

| Environment | State | Actions | Max Return | Solved Threshold |
|------------|-------|---------|------------|-----------------|
| CartPole-v1 | 4 (continuous) | 2 (discrete) | 500 | 475 (avg 100 ep) |
| MountainCar-v0 | 2 (continuous) | 3 (discrete) | -110 | -110 |
| LunarLander-v2 | 8 (continuous) | 4 (discrete) | ~300 | 200 |
| Acrobot-v1 | 6 (continuous) | 3 (discrete) | -100 | -100 |

### Atari (ALE)
The standard DQN benchmark with 57 games:
- **Scoring**: Human-normalized score = $\frac{\text{agent} - \text{random}}{\text{human} - \text{random}} \times 100\%$
- **Key games**: Breakout, Pong, Space Invaders, Seaquest, Montezuma's Revenge
- **Frame stacking**: 4 frames, 84×84 grayscale
- **Evaluation protocol**: 30 no-op starts, average over 100 episodes

### D4RL (Offline RL)
Standardized offline RL benchmark:
- **MuJoCo**: HalfCheetah, Hopper, Walker2d
- **Dataset types**: random, medium, medium-replay, medium-expert, expert
- **Normalized score**: 0 = random, 100 = expert

## Algorithm Performance Comparison

### Atari (Median Human-Normalized Score)

| Algorithm | Median Score | Year |
|-----------|-------------|------|
| DQN | 79% | 2015 |
| Double DQN | 117% | 2016 |
| Dueling DQN | 117% | 2016 |
| Prioritized DQN | 128% | 2016 |
| C51 | 178% | 2017 |
| Rainbow | 230% | 2018 |
| Agent57 | 5089% | 2020 |

### D4RL Offline RL (Average Normalized Score)

| Algorithm | HalfCheetah-medium | Hopper-medium |
|-----------|-------------------|---------------|
| BC | 42.6 | 52.5 |
| DQN (offline) | 28.4 | 10.8 |
| CQL | 44.0 | 58.5 |
| IQL | 47.4 | 66.3 |

## Benchmarking Best Practices

1. **Multiple seeds**: Report mean ± std over at least 3 seeds (preferably 5–10)
2. **Consistent evaluation**: Same number of episodes, same ε
3. **Wall-clock time**: Report both sample efficiency (steps) and wall-clock time
4. **Hyperparameter transparency**: Report exact hyperparameters and tuning procedure
5. **Statistical tests**: Use proper tests for significance (see Section 33.6.3)
6. **Environment version**: Specify exact Gymnasium/ALE version
