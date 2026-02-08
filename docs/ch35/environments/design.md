# 35.1.1 Environment Design

## Learning Objectives

- Design Gymnasium-compatible financial trading environments
- Understand the architecture of financial RL environments
- Implement modular, extensible environment components
- Handle data feeding, episode management, and environment resets

## Introduction

The foundation of any RL application in finance is a well-designed environment that faithfully simulates market dynamics. Unlike Atari games or robotic control, financial environments must capture the nuances of real markets: discrete time steps aligned with trading frequency, realistic order execution, and proper handling of portfolio accounting.

A financial RL environment follows the standard Gymnasium interface but requires careful attention to several finance-specific concerns.

## Environment Architecture

A financial trading environment consists of several interconnected components:

```
┌─────────────────────────────────────────────┐
│              Trading Environment             │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Data    │  │ Portfolio │  │  Market   │  │
│  │  Feeder   │  │ Manager  │  │ Simulator │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │        │
│       ▼              ▼              ▼        │
│  ┌──────────────────────────────────────┐   │
│  │        State Constructor             │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│       ┌─────────┴─────────┐                 │
│       ▼                   ▼                 │
│  ┌─────────┐        ┌──────────┐           │
│  │ Reward  │        │  Action   │           │
│  │ Engine  │        │ Executor  │           │
│  └─────────┘        └──────────┘           │
└─────────────────────────────────────────────┘
```

### Key Design Principles

1. **Modularity**: Each component (data feeding, execution, reward) should be independently configurable
2. **Reproducibility**: Given the same seed and data, the environment must produce identical trajectories
3. **Efficiency**: Vectorized operations for state computation; avoid per-step Python loops
4. **Realism**: Include transaction costs, slippage models, and market impact
5. **Flexibility**: Support different trading frequencies (daily, hourly, minute-level)

## Base Environment Class

The base environment inherits from `gymnasium.Env` and provides the standard `reset()`, `step()`, and `render()` interface:

```python
class TradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, config):
        self.data_feeder = DataFeeder(config)
        self.portfolio = PortfolioManager(config)
        self.market_sim = MarketSimulator(config)
        self.reward_engine = RewardEngine(config)
        
        self.observation_space = self._build_obs_space(config)
        self.action_space = self._build_action_space(config)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data_feeder.reset()
        self.portfolio.reset()
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Execute action → get fills
        fills = self.market_sim.execute(action, self.portfolio)
        # Update portfolio
        self.portfolio.update(fills)
        # Advance time
        self.data_feeder.step()
        # Compute reward
        reward = self.reward_engine.compute(self.portfolio)
        # Check termination
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        return self._get_obs(), reward, terminated, truncated, self._get_info()
```

## Data Feeding

The data feeder manages historical market data and provides it to the environment at each time step:

```python
class DataFeeder:
    def __init__(self, config):
        self.prices = config['prices']       # (T, N) array
        self.features = config['features']   # (T, N, F) array
        self.lookback = config['lookback']   # Window size
        self.current_step = 0
        self.max_steps = len(self.prices) - self.lookback
    
    def reset(self, start_idx=None):
        if start_idx is not None:
            self.current_step = start_idx
        else:
            self.current_step = 0
    
    def get_window(self):
        start = self.current_step
        end = start + self.lookback
        return {
            'prices': self.prices[start:end],
            'features': self.features[start:end],
            'current_price': self.prices[end - 1]
        }
    
    def step(self):
        self.current_step += 1
```

## Episode Management

Financial environments require careful episode management:

- **Fixed-length episodes**: Train on fixed windows (e.g., 252 trading days = 1 year)
- **Rolling windows**: Episodes start at different points in the data
- **Randomized starts**: Randomly sample start dates for diverse experience
- **Train/validation/test splits**: Temporal splits to prevent look-ahead bias

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    
    if self.mode == 'train':
        # Random start within training period
        max_start = self.train_end - self.episode_length - self.lookback
        start = self.np_random.integers(self.train_start, max_start)
    elif self.mode == 'eval':
        # Sequential evaluation
        start = self.eval_start
    
    self.data_feeder.reset(start_idx=start)
    self.portfolio.reset(initial_capital=self.initial_capital)
    self.step_count = 0
    
    return self._get_obs(), self._get_info()
```

## Configuration Pattern

A configuration-driven design allows easy experimentation:

```python
config = {
    'data': {
        'assets': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        'frequency': 'daily',
        'lookback': 60,
        'episode_length': 252,
    },
    'portfolio': {
        'initial_capital': 1_000_000,
        'max_leverage': 1.0,
        'transaction_cost': 0.001,
    },
    'reward': {
        'type': 'sharpe',
        'risk_free_rate': 0.02,
        'window': 20,
    },
    'action': {
        'type': 'continuous',  # or 'discrete'
        'allow_short': False,
    }
}
```

## Summary

Financial environment design requires balancing simulation fidelity with computational efficiency. The modular architecture separates concerns (data, execution, reward) and supports rapid experimentation. Key decisions include trading frequency, episode structure, and the level of market microstructure detail to include.

## References

- OpenAI Gymnasium documentation: https://gymnasium.farama.org/
- FinRL: A Deep Reinforcement Learning Library for Quantitative Finance (Liu et al., 2020)
- Practical Deep Reinforcement Learning Approach for Stock Trading (Xiong et al., 2018)
