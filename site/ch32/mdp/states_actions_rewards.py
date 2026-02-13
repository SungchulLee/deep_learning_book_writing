"""
Chapter 32.2: States, Actions, Reward Functions, and Discount Factor
=====================================================================
Demonstrates state/action space design, reward function variants,
reward shaping, and financial MDP modeling.
"""

import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


# =============================================================================
# 1. State Space Design Examples
# =============================================================================

class TradingStateBuilder:
    """
    Builds state representations for a trading RL agent.
    Demonstrates feature engineering for financial states.
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def build_basic_state(self, prices: np.ndarray, position: float,
                          t: int) -> np.ndarray:
        """Basic state: current return + position."""
        if t < 1:
            return np.array([0.0, position])
        ret = (prices[t] - prices[t-1]) / prices[t-1]
        return np.array([ret, position])

    def build_technical_state(self, prices: np.ndarray, volumes: np.ndarray,
                               position: float, t: int) -> np.ndarray:
        """
        Rich state with technical indicators:
        [return, volatility, rsi, volume_ratio, sma_cross, position]
        """
        if t < self.lookback:
            return np.zeros(6)

        window = prices[t - self.lookback:t + 1]
        vol_window = volumes[t - self.lookback:t + 1]

        # Returns
        rets = np.diff(np.log(window))
        current_ret = rets[-1]

        # Realized volatility
        vol = np.std(rets) * np.sqrt(252)

        # RSI
        gains = np.maximum(rets, 0)
        losses = np.maximum(-rets, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        rsi = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
        rsi_normalized = (rsi - 50) / 50  # Center and normalize

        # Volume ratio
        vol_ratio = vol_window[-1] / (np.mean(vol_window[:-1]) + 1e-10)

        # SMA crossover signal
        sma_fast = np.mean(window[-5:])
        sma_slow = np.mean(window[-20:])
        sma_cross = (sma_fast - sma_slow) / sma_slow

        return np.array([current_ret, vol, rsi_normalized, vol_ratio, sma_cross, position])

    @staticmethod
    def discretize_state(continuous_state: np.ndarray,
                          bins_per_feature: int = 10) -> int:
        """Discretize continuous state into a single integer index."""
        # Clip to [-3, 3] standard deviations then bin
        clipped = np.clip(continuous_state, -3, 3)
        normalized = (clipped + 3) / 6  # Map to [0, 1]
        binned = np.minimum((normalized * bins_per_feature).astype(int),
                           bins_per_feature - 1)
        # Convert multi-dim bin to single index
        index = 0
        for i, b in enumerate(binned):
            index = index * bins_per_feature + b
        return index


# =============================================================================
# 2. Action Space Design
# =============================================================================

class ActionSpaceDesign:
    """Demonstrates different action space designs for trading."""

    @staticmethod
    def discrete_actions() -> Dict[int, str]:
        """Simple discrete: {sell, hold, buy}"""
        return {0: "sell", 1: "hold", 2: "buy"}

    @staticmethod
    def fine_discrete_actions() -> Dict[int, str]:
        """Finer discrete: 5 levels"""
        return {0: "strong_sell", 1: "sell", 2: "hold",
                3: "buy", 4: "strong_buy"}

    @staticmethod
    def position_target_actions(n_levels: int = 11) -> np.ndarray:
        """Discrete set of target positions from -1 to +1."""
        return np.linspace(-1, 1, n_levels)

    @staticmethod
    def continuous_to_discrete(continuous_action: float,
                                levels: np.ndarray) -> int:
        """Map a continuous action to the nearest discrete level."""
        return int(np.argmin(np.abs(levels - continuous_action)))


# =============================================================================
# 3. Reward Function Implementations
# =============================================================================

class RewardFunctions:
    """Collection of reward functions for financial RL."""

    @staticmethod
    def simple_pnl(portfolio_return: float) -> float:
        """Raw P&L reward."""
        return portfolio_return

    @staticmethod
    def log_return(price_now: float, price_prev: float,
                   position: float) -> float:
        """Log return: r = position * log(P_t / P_{t-1})"""
        return position * np.log(price_now / price_prev)

    @staticmethod
    def risk_adjusted(returns_window: np.ndarray) -> float:
        """Sharpe-like reward using recent window."""
        if len(returns_window) < 2:
            return 0.0
        mu = np.mean(returns_window)
        sigma = np.std(returns_window) + 1e-8
        return mu / sigma

    @staticmethod
    def differential_sharpe(return_t: float, A: float, B: float,
                             eta: float = 0.01) -> Tuple[float, float, float]:
        """
        Differential Sharpe Ratio (Moody & Saffell, 2001).
        Returns (reward, updated_A, updated_B).
        """
        delta_A = return_t - A
        delta_B = return_t**2 - B
        denom = (B - A**2) ** 1.5

        if abs(denom) > 1e-10:
            dsr = (B * delta_A - 0.5 * A * delta_B) / denom
        else:
            dsr = 0.0

        new_A = A + eta * delta_A
        new_B = B + eta * delta_B
        return dsr, new_A, new_B

    @staticmethod
    def drawdown_penalty(return_t: float, peak: float, current_value: float,
                          lambda_dd: float = 0.5) -> Tuple[float, float]:
        """
        Return with drawdown penalty.
        Returns (reward, updated_peak).
        """
        new_peak = max(peak, current_value)
        dd = max(0, (new_peak - current_value) / (new_peak + 1e-10))
        reward = return_t - lambda_dd * dd
        return reward, new_peak

    @staticmethod
    def transaction_cost_adjusted(portfolio_return: float,
                                    position_change: float,
                                    cost_rate: float = 0.001) -> float:
        """Return minus proportional transaction costs."""
        return portfolio_return - cost_rate * abs(position_change)

    @staticmethod
    def potential_based_shaping(base_reward: float, state: np.ndarray,
                                 next_state: np.ndarray, gamma: float,
                                 potential_fn=None) -> float:
        """
        F(s,a,s') = γΦ(s') - Φ(s)
        Preserves optimal policy.
        """
        if potential_fn is None:
            potential_fn = lambda s: -np.sum(np.abs(s))  # Default: negative L1 norm
        shaping = gamma * potential_fn(next_state) - potential_fn(state)
        return base_reward + shaping


# =============================================================================
# 4. Reward Function Comparison Simulation
# =============================================================================

def simulate_trading_with_rewards(n_days: int = 252, seed: int = 42):
    """Run a simple trading sim comparing different reward functions."""
    np.random.seed(seed)

    # Generate price series (GBM)
    dt = 1/252
    mu, sigma = 0.08, 0.20
    daily_rets = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    prices = 100 * np.cumprod(np.concatenate([[1], 1 + daily_rets]))

    # Generate volumes
    volumes = np.random.lognormal(mean=15, sigma=0.5, size=len(prices))

    # Simple momentum strategy
    positions = np.zeros(n_days + 1)
    for t in range(20, n_days + 1):
        sma5 = np.mean(prices[t-5:t])
        sma20 = np.mean(prices[t-20:t])
        positions[t] = 1.0 if sma5 > sma20 else -1.0

    # Compute portfolio returns
    port_returns = positions[:-1] * daily_rets

    # Compare reward functions
    rf = RewardFunctions()
    rewards = {
        "Simple PnL": [],
        "Log Return": [],
        "Risk-Adjusted": [],
        "Diff Sharpe": [],
        "DD-Penalized": [],
        "TC-Adjusted": [],
    }

    A, B = 0.0, 0.0  # For differential Sharpe
    peak = 100.0
    cumval = 100.0
    returns_buffer = []

    for t in range(n_days):
        r = port_returns[t]
        returns_buffer.append(r)
        cumval *= (1 + r)

        # Simple PnL
        rewards["Simple PnL"].append(rf.simple_pnl(r))

        # Log return
        if t > 0:
            rewards["Log Return"].append(
                rf.log_return(prices[t+1], prices[t], positions[t]))
        else:
            rewards["Log Return"].append(0.0)

        # Risk-adjusted (rolling window)
        if len(returns_buffer) >= 20:
            rewards["Risk-Adjusted"].append(
                rf.risk_adjusted(np.array(returns_buffer[-20:])))
        else:
            rewards["Risk-Adjusted"].append(0.0)

        # Differential Sharpe
        dsr, A, B = rf.differential_sharpe(r, A, B, eta=0.01)
        rewards["Diff Sharpe"].append(dsr)

        # Drawdown-penalized
        dd_r, peak = rf.drawdown_penalty(r, peak, cumval, lambda_dd=0.5)
        rewards["DD-Penalized"].append(dd_r)

        # TC-adjusted
        pos_change = abs(positions[t+1] - positions[t]) if t < n_days - 1 else 0
        rewards["TC-Adjusted"].append(
            rf.transaction_cost_adjusted(r, pos_change, cost_rate=0.001))

    # Print summary
    print("=" * 70)
    print("Reward Function Comparison (Momentum Strategy, 1-year sim)")
    print("=" * 70)
    print(f"{'Reward Type':<20} {'Mean':>10} {'Std':>10} {'Total':>10} {'Sharpe':>10}")
    print("-" * 60)
    for name, rew in rewards.items():
        r = np.array(rew)
        sharpe = np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(252)
        print(f"{name:<20} {np.mean(r):>10.6f} {np.std(r):>10.6f} "
              f"{np.sum(r):>10.4f} {sharpe:>10.4f}")

    return prices, positions, rewards


# =============================================================================
# 5. Visualization
# =============================================================================

def plot_state_features(prices: np.ndarray, volumes: np.ndarray):
    """Visualize different state representations."""
    builder = TradingStateBuilder(lookback=20)

    states = []
    for t in range(20, len(prices)):
        s = builder.build_technical_state(prices, volumes, 0.0, t)
        states.append(s)
    states = np.array(states)

    feature_names = ['Return', 'Volatility', 'RSI (norm)', 'Volume Ratio',
                     'SMA Cross', 'Position']

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        if i < states.shape[1]:
            ax.plot(states[:, i], alpha=0.7, linewidth=0.8)
            ax.set_title(f'State Feature: {name}')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.2)

    plt.suptitle('Trading State Representation Features', fontsize=14)
    plt.tight_layout()
    plt.savefig("mdp_state_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mdp_state_features.png")


def plot_reward_comparison(rewards: Dict[str, List[float]]):
    """Plot cumulative rewards under different reward functions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (name, rew) in enumerate(rewards.items()):
        ax = axes[idx]
        r = np.array(rew)
        cumulative = np.cumsum(r)
        ax.plot(cumulative, linewidth=1)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel('Day')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Cumulative Rewards: Different Reward Functions', fontsize=14)
    plt.tight_layout()
    plt.savefig("mdp_reward_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mdp_reward_comparison.png")


def plot_action_space_designs():
    """Visualize different action space designs."""
    asd = ActionSpaceDesign()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Discrete 3
    ax = axes[0]
    actions_3 = asd.discrete_actions()
    ax.bar(actions_3.keys(), [1]*3, tick_label=list(actions_3.values()),
           color=['#e74c3c', '#95a5a6', '#2ecc71'])
    ax.set_title('3-Action (Sell/Hold/Buy)')
    ax.set_ylabel('Available')

    # Discrete 5
    ax = axes[1]
    actions_5 = asd.fine_discrete_actions()
    colors = ['#c0392b', '#e74c3c', '#95a5a6', '#2ecc71', '#27ae60']
    ax.bar(actions_5.keys(), [1]*5, tick_label=list(actions_5.values()),
           color=colors)
    ax.set_title('5-Action (Fine-grained)')
    ax.tick_params(axis='x', rotation=30)

    # Position targets
    ax = axes[2]
    levels = asd.position_target_actions(n_levels=11)
    colors = ['#e74c3c' if l < 0 else '#2ecc71' if l > 0 else '#95a5a6' for l in levels]
    ax.bar(range(len(levels)), levels, color=colors)
    ax.set_xlabel('Action Index')
    ax.set_ylabel('Target Position')
    ax.set_title('11-Level Position Targets')
    ax.axhline(y=0, color='k', linewidth=0.5)

    plt.suptitle('Trading Action Space Designs', fontsize=14)
    plt.tight_layout()
    plt.savefig("mdp_action_spaces.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: mdp_action_spaces.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Generate data
    n_days = 252
    dt = 1/252
    daily_rets = np.random.normal(0.08*dt, 0.20*np.sqrt(dt), n_days)
    prices = 100 * np.cumprod(np.concatenate([[1], 1 + daily_rets]))
    volumes = np.random.lognormal(15, 0.5, len(prices))

    # 2. State space demo
    print("=" * 60)
    print("State Space Design")
    print("=" * 60)
    builder = TradingStateBuilder(lookback=20)
    basic = builder.build_basic_state(prices, 1.0, 50)
    print(f"Basic state (2 features): {basic}")
    tech = builder.build_technical_state(prices, volumes, 1.0, 50)
    print(f"Technical state (6 features): {[f'{x:.4f}' for x in tech]}")
    disc = builder.discretize_state(tech, bins_per_feature=10)
    print(f"Discretized state index: {disc}")

    # 3. Reward comparison
    prices2, positions2, rewards = simulate_trading_with_rewards()

    # 4. Visualizations
    plot_state_features(prices, volumes)
    plot_reward_comparison(rewards)
    plot_action_space_designs()

    print("\n✓ States, Actions, and Rewards demonstrations complete.")
