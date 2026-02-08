"""
Chapter 35.1.1: Financial Trading Environment Design
=====================================================
Complete Gymnasium-compatible trading environment with modular architecture.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# ============================================================
# Configuration
# ============================================================

@dataclass
class EnvironmentConfig:
    """Configuration for the trading environment."""
    # Data
    num_assets: int = 4
    lookback: int = 60
    episode_length: int = 252
    
    # Portfolio
    initial_capital: float = 1_000_000.0
    max_leverage: float = 1.0
    max_position: float = 0.25
    allow_short: bool = False
    
    # Transaction costs
    transaction_cost: float = 0.001  # 10 bps
    slippage_std: float = 0.0005
    min_trade_size: float = 0.001
    
    # Reward
    reward_type: str = "log_return"  # log_return, sharpe, risk_adjusted
    risk_penalty: float = 0.5
    
    # Mode
    mode: str = "train"  # train, eval, test


# ============================================================
# Data Feeder
# ============================================================

class DataFeeder:
    """Manages market data and provides windowed observations."""
    
    def __init__(self, prices: np.ndarray, features: Optional[np.ndarray] = None,
                 lookback: int = 60):
        """
        Args:
            prices: (T, N) array of asset prices
            features: (T, N, F) array of additional features (optional)
            lookback: Number of past steps to include in observation
        """
        self.prices = prices
        self.features = features
        self.lookback = lookback
        self.num_steps = len(prices)
        self.num_assets = prices.shape[1]
        self.current_step = 0
        
        # Precompute log returns
        self.log_returns = np.diff(np.log(prices), axis=0)
        # Pad first row with zeros
        self.log_returns = np.vstack([np.zeros((1, self.num_assets)), self.log_returns])
    
    def reset(self, start_idx: int = 0):
        """Reset to a specific starting index."""
        self.current_step = start_idx
    
    def get_window(self) -> Dict[str, np.ndarray]:
        """Get the current observation window."""
        start = self.current_step
        end = start + self.lookback
        
        result = {
            'prices': self.prices[start:end],
            'log_returns': self.log_returns[start:end],
            'current_price': self.prices[end - 1],
        }
        
        if self.features is not None:
            result['features'] = self.features[start:end]
        
        return result
    
    def get_current_price(self) -> np.ndarray:
        """Get the current (latest) price."""
        idx = self.current_step + self.lookback - 1
        return self.prices[idx]
    
    def get_next_price(self) -> np.ndarray:
        """Get the next period's price (for computing returns)."""
        idx = self.current_step + self.lookback
        if idx < self.num_steps:
            return self.prices[idx]
        return self.prices[-1]
    
    def step(self):
        """Advance one time step."""
        self.current_step += 1
    
    @property
    def done(self) -> bool:
        return self.current_step + self.lookback >= self.num_steps
    
    @property
    def max_start(self) -> int:
        return self.num_steps - self.lookback - 1


# ============================================================
# Portfolio Manager
# ============================================================

class PortfolioManager:
    """Tracks portfolio state, positions, and performance."""
    
    def __init__(self, num_assets: int, initial_capital: float = 1_000_000.0):
        self.num_assets = num_assets
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = np.zeros(self.num_assets)  # Number of shares
        self.weights = np.zeros(self.num_assets)
        self.total_value = self.initial_capital
        self.prev_total_value = self.initial_capital
        
        # Performance tracking
        self.value_history = [self.initial_capital]
        self.return_history = []
        self.trade_history = []
        self.cost_history = []
    
    def get_weights(self) -> np.ndarray:
        """Get current portfolio weights."""
        return self.weights.copy()
    
    def update(self, new_weights: np.ndarray, current_prices: np.ndarray,
               costs: float = 0.0):
        """Update portfolio with new weights after trades."""
        self.prev_total_value = self.total_value
        
        # Update weights and positions
        self.weights = new_weights.copy()
        
        # Track costs
        self.cost_history.append(costs)
        self.total_value -= costs
    
    def mark_to_market(self, current_prices: np.ndarray):
        """Revalue portfolio at current prices."""
        if np.any(self.weights != 0):
            # Compute return from price changes
            asset_returns = current_prices / self._prev_prices - 1
            portfolio_return = np.dot(self.weights, asset_returns)
            self.total_value = self.prev_total_value * (1 + portfolio_return)
        
        self.value_history.append(self.total_value)
        period_return = self.total_value / self.prev_total_value - 1
        self.return_history.append(period_return)
    
    def set_prices(self, prices: np.ndarray):
        """Set reference prices for return computation."""
        self._prev_prices = prices.copy()
    
    @property
    def cumulative_return(self) -> float:
        return self.total_value / self.initial_capital - 1
    
    @property
    def max_drawdown(self) -> float:
        values = np.array(self.value_history)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return drawdown.max() if len(drawdown) > 0 else 0.0
    
    @property
    def sharpe_ratio(self) -> float:
        if len(self.return_history) < 2:
            return 0.0
        returns = np.array(self.return_history)
        if returns.std() == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()


# ============================================================
# Market Simulator
# ============================================================

class MarketSimulator:
    """Simulates order execution with realistic friction."""
    
    def __init__(self, transaction_cost: float = 0.001,
                 slippage_std: float = 0.0005,
                 impact_coeff: float = 0.1):
        self.transaction_cost = transaction_cost
        self.slippage_std = slippage_std
        self.impact_coeff = impact_coeff
    
    def execute(self, target_weights: np.ndarray,
                current_weights: np.ndarray,
                total_value: float,
                current_prices: np.ndarray,
                rng: np.random.Generator) -> Dict[str, Any]:
        """
        Execute trades to move from current to target weights.
        
        Returns:
            Dictionary with fill prices, costs, and trade details.
        """
        trades = target_weights - current_weights
        trade_values = np.abs(trades) * total_value
        
        # Compute slippage
        slippage = rng.normal(0, self.slippage_std, len(current_prices))
        fill_prices = current_prices * (1 + np.sign(trades) * np.abs(slippage))
        
        # Compute transaction costs
        total_cost = self.transaction_cost * trade_values.sum()
        
        return {
            'fill_prices': fill_prices,
            'total_cost': total_cost,
            'trades': trades,
            'trade_values': trade_values,
            'turnover': trade_values.sum() / total_value,
        }


# ============================================================
# Reward Engine
# ============================================================

class RewardEngine:
    """Computes reward signals for the RL agent."""
    
    def __init__(self, reward_type: str = "log_return",
                 risk_penalty: float = 0.5,
                 risk_free_rate: float = 0.02):
        self.reward_type = reward_type
        self.risk_penalty = risk_penalty
        self.risk_free_rate = risk_free_rate / 252  # Daily
        
        # For differential Sharpe ratio
        self._ema_return = 0.0
        self._ema_return_sq = 0.0
        self._eta = 0.01
    
    def reset(self):
        """Reset internal state."""
        self._ema_return = 0.0
        self._ema_return_sq = 0.0
    
    def compute(self, portfolio: PortfolioManager, costs: float = 0.0) -> float:
        """Compute reward based on portfolio state."""
        if len(portfolio.return_history) == 0:
            return 0.0
        
        period_return = portfolio.return_history[-1]
        net_return = period_return - costs / portfolio.prev_total_value
        
        if self.reward_type == "log_return":
            return self._log_return_reward(net_return)
        elif self.reward_type == "sharpe":
            return self._differential_sharpe(net_return)
        elif self.reward_type == "risk_adjusted":
            return self._risk_adjusted_reward(net_return, portfolio)
        elif self.reward_type == "sortino":
            return self._sortino_reward(net_return)
        else:
            return net_return
    
    def _log_return_reward(self, net_return: float) -> float:
        return np.log(1 + net_return)
    
    def _differential_sharpe(self, net_return: float) -> float:
        """Differential Sharpe Ratio (Moody & Saffell, 2001)."""
        delta_A = net_return - self._ema_return
        delta_B = net_return ** 2 - self._ema_return_sq
        
        denominator = (self._ema_return_sq - self._ema_return ** 2) ** 1.5
        
        if abs(denominator) < 1e-12:
            dsr = net_return  # Fallback
        else:
            dsr = (self._ema_return_sq * delta_A - 0.5 * self._ema_return * delta_B) / denominator
        
        # Update EMAs
        self._ema_return += self._eta * delta_A
        self._ema_return_sq += self._eta * delta_B
        
        return dsr
    
    def _risk_adjusted_reward(self, net_return: float,
                               portfolio: PortfolioManager) -> float:
        """Return minus risk penalty."""
        drawdown = portfolio.max_drawdown
        return net_return - self.risk_penalty * drawdown
    
    def _sortino_reward(self, net_return: float) -> float:
        """Sortino-based: extra penalty for negative returns."""
        downside_penalty = self.risk_penalty * max(0, -net_return) ** 2
        return net_return - downside_penalty


# ============================================================
# Main Trading Environment
# ============================================================

class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment.
    
    Supports multi-asset portfolio management with realistic
    transaction costs, slippage, and configurable reward functions.
    
    Observation Space:
        Dict with 'market' (lookback x features), 'portfolio' (weights),
        and 'account' (value, cash_ratio, drawdown).
    
    Action Space:
        Box(0, 1, num_assets) - target portfolio weights (long-only).
        Automatically normalized via softmax.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, prices: np.ndarray,
                 features: Optional[np.ndarray] = None,
                 config: Optional[EnvironmentConfig] = None):
        """
        Args:
            prices: (T, N) array of asset prices
            features: (T, N, F) optional additional features
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or EnvironmentConfig()
        self.config.num_assets = prices.shape[1]
        
        # Initialize components
        self.data_feeder = DataFeeder(prices, features, self.config.lookback)
        self.portfolio = PortfolioManager(
            self.config.num_assets, self.config.initial_capital
        )
        self.market_sim = MarketSimulator(
            transaction_cost=self.config.transaction_cost,
            slippage_std=self.config.slippage_std,
        )
        self.reward_engine = RewardEngine(
            reward_type=self.config.reward_type,
            risk_penalty=self.config.risk_penalty,
        )
        
        # Define spaces
        num_features = self.config.num_assets * 4  # returns, vol, momentum, rsi
        
        self.observation_space = spaces.Dict({
            'market': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.config.lookback, self.config.num_assets * 4),
                dtype=np.float32
            ),
            'portfolio': spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.config.num_assets,),
                dtype=np.float32
            ),
            'account': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,),  # normalized_value, cash_ratio, drawdown
                dtype=np.float32
            ),
        })
        
        if self.config.allow_short:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.config.num_assets,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(self.config.num_assets,),
                dtype=np.float32
            )
        
        self.step_count = 0
        self._rng = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Reset components
        self.portfolio.reset()
        self.reward_engine.reset()
        
        # Choose start index
        if self.config.mode == 'train':
            max_start = self.data_feeder.max_start - self.config.episode_length
            if max_start > 0:
                start = self._rng.integers(0, max_start)
            else:
                start = 0
        else:
            start = options.get('start_idx', 0) if options else 0
        
        self.data_feeder.reset(start_idx=start)
        self.portfolio.set_prices(self.data_feeder.get_current_price())
        self.step_count = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one trading step."""
        # Transform action to valid weights
        target_weights = self._transform_action(action)
        
        # Execute trades
        current_prices = self.data_feeder.get_current_price()
        execution = self.market_sim.execute(
            target_weights=target_weights,
            current_weights=self.portfolio.get_weights(),
            total_value=self.portfolio.total_value,
            current_prices=current_prices,
            rng=self._rng,
        )
        
        # Update portfolio weights
        self.portfolio.update(target_weights, current_prices, execution['total_cost'])
        self.portfolio.set_prices(current_prices)
        
        # Advance time
        self.data_feeder.step()
        self.step_count += 1
        
        # Mark-to-market with new prices
        new_prices = self.data_feeder.get_current_price()
        self.portfolio.mark_to_market(new_prices)
        
        # Compute reward
        reward = self.reward_engine.compute(
            self.portfolio, execution['total_cost']
        )
        
        # Check termination
        terminated = self.portfolio.total_value <= 0  # Bankruptcy
        truncated = (
            self.step_count >= self.config.episode_length
            or self.data_feeder.done
        )
        
        info = self._get_info()
        info.update({
            'turnover': execution['turnover'],
            'costs': execution['total_cost'],
        })
        
        return self._get_obs(), float(reward), terminated, truncated, info
    
    def _transform_action(self, action: np.ndarray) -> np.ndarray:
        """Transform raw action to valid portfolio weights."""
        action = np.asarray(action, dtype=np.float64)
        
        if self.config.allow_short:
            # Tanh activation, normalize by gross leverage
            weights = np.tanh(action)
            gross = np.abs(weights).sum()
            if gross > self.config.max_leverage:
                weights *= self.config.max_leverage / gross
        else:
            # Softmax for long-only
            exp_a = np.exp(action - np.max(action))  # Numerical stability
            weights = exp_a / exp_a.sum()
        
        # Enforce max position constraint
        weights = np.clip(weights, -self.config.max_position, self.config.max_position)
        
        # Re-normalize if needed
        total = weights.sum()
        if abs(total) > 1e-8:
            weights = weights / abs(total) * min(abs(total), self.config.max_leverage)
        
        # Filter small trades
        current_weights = self.portfolio.get_weights()
        trades = weights - current_weights
        small = np.abs(trades) < self.config.min_trade_size
        weights[small] = current_weights[small]
        
        return weights
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary."""
        window = self.data_feeder.get_window()
        returns = window['log_returns']
        
        # Compute features per asset
        features_list = []
        for i in range(self.config.num_assets):
            asset_returns = returns[:, i]
            
            # Rolling volatility (20-day)
            vol = np.array([
                asset_returns[max(0, j-19):j+1].std()
                for j in range(len(asset_returns))
            ])
            
            # Cumulative momentum (various windows)
            momentum = np.cumsum(asset_returns)
            
            # Simple RSI approximation
            gains = np.maximum(asset_returns, 0)
            losses = np.maximum(-asset_returns, 0)
            avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
            avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 1 - 1 / (1 + rs)
            
            features_list.append(np.column_stack([
                asset_returns, vol, momentum, rsi
            ]))
        
        market_features = np.concatenate(features_list, axis=-1)
        
        # Normalize (z-score)
        mean = market_features.mean(axis=0, keepdims=True)
        std = market_features.std(axis=0, keepdims=True) + 1e-8
        market_features = (market_features - mean) / std
        
        return {
            'market': market_features.astype(np.float32),
            'portfolio': self.portfolio.get_weights().astype(np.float32),
            'account': np.array([
                self.portfolio.total_value / self.config.initial_capital,
                1.0 - np.abs(self.portfolio.weights).sum(),  # Cash ratio
                self.portfolio.max_drawdown,
            ], dtype=np.float32),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Return current info dictionary."""
        return {
            'total_value': self.portfolio.total_value,
            'cumulative_return': self.portfolio.cumulative_return,
            'max_drawdown': self.portfolio.max_drawdown,
            'sharpe_ratio': self.portfolio.sharpe_ratio,
            'step': self.step_count,
            'weights': self.portfolio.get_weights().tolist(),
        }


# ============================================================
# Demo: Create and run environment
# ============================================================

def generate_synthetic_data(num_steps: int = 1000, num_assets: int = 4,
                            seed: int = 42) -> np.ndarray:
    """Generate synthetic price data for testing."""
    rng = np.random.default_rng(seed)
    
    # Parameters
    mu = rng.uniform(0.0001, 0.0005, num_assets)  # Daily drift
    sigma = rng.uniform(0.01, 0.03, num_assets)    # Daily volatility
    
    # Generate correlated returns
    correlation = np.eye(num_assets)
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            correlation[i, j] = correlation[j, i] = rng.uniform(0.2, 0.6)
    
    L = np.linalg.cholesky(correlation)
    
    log_returns = np.zeros((num_steps, num_assets))
    for t in range(num_steps):
        z = rng.standard_normal(num_assets)
        log_returns[t] = mu + sigma * (L @ z)
    
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    
    return prices


def demo_environment():
    """Demonstrate the trading environment."""
    print("=" * 60)
    print("Trading Environment Demo")
    print("=" * 60)
    
    # Generate synthetic data
    prices = generate_synthetic_data(num_steps=500, num_assets=4)
    print(f"\nPrice data shape: {prices.shape}")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Create environment
    config = EnvironmentConfig(
        lookback=30,
        episode_length=200,
        initial_capital=1_000_000,
        transaction_cost=0.001,
        reward_type='log_return',
        mode='train',
    )
    
    env = TradingEnv(prices, config=config)
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a random episode
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shapes:")
    for key, val in obs.items():
        print(f"  {key}: {val.shape}")
    
    total_reward = 0
    step = 0
    
    while True:
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode completed:")
    print(f"  Steps: {step}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final value: ${info['total_value']:,.2f}")
    print(f"  Cumulative return: {info['cumulative_return']:.2%}")
    print(f"  Max drawdown: {info['max_drawdown']:.2%}")
    print(f"  Sharpe ratio: {info['sharpe_ratio']:.4f}")
    
    # Test different reward types
    print(f"\n{'=' * 60}")
    print("Testing different reward types")
    print("=" * 60)
    
    for reward_type in ['log_return', 'sharpe', 'risk_adjusted', 'sortino']:
        config.reward_type = reward_type
        env = TradingEnv(prices, config=config)
        obs, info = env.reset(seed=42)
        
        total_reward = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"  {reward_type:15s}: total_reward={total_reward:8.4f}, "
              f"return={info['cumulative_return']:7.2%}")
    
    print("\nEnvironment design demo complete!")


if __name__ == "__main__":
    demo_environment()
