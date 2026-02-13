"""
Chapter 32.1.3: The Reward Hypothesis
=======================================
Demonstrates reward design, return computation, discount factors,
reward shaping, and financial reward functions.
"""

import numpy as np
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt


# =============================================================================
# 1. Return Computation with Different Discount Factors
# =============================================================================

def compute_return(rewards: List[float], gamma: float, start_t: int = 0) -> float:
    """Compute discounted return G_t = sum_{k=0}^{T-t-1} gamma^k * R_{t+k+1}."""
    G = 0.0
    for k, r in enumerate(rewards[start_t:]):
        G += (gamma ** k) * r
    return G


def compute_all_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute G_t for all time steps (backward recursion: G_t = R_{t+1} + γG_{t+1})."""
    n = len(rewards)
    returns = [0.0] * n
    G = 0.0
    for t in range(n - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def demonstrate_discount_effects():
    """Show how discount factor affects the return."""
    print("=" * 60)
    print("Effect of Discount Factor on Returns")
    print("=" * 60)

    # Scenario: reward of 1 at each step for 100 steps
    n_steps = 100
    rewards = [1.0] * n_steps

    gammas = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    print(f"\nConstant reward = 1.0 for {n_steps} steps:")
    for gamma in gammas:
        G = compute_return(rewards, gamma)
        theoretical = (1 - gamma**n_steps) / (1 - gamma) if gamma < 1 else n_steps
        print(f"  γ = {gamma:.2f} → G_0 = {G:.4f}  (theoretical: {theoretical:.4f})")

    # Scenario: delayed reward
    print(f"\nDelayed reward (reward=10 only at step 50):")
    delayed_rewards = [0.0] * 49 + [10.0] + [0.0] * 50
    for gamma in gammas:
        G = compute_return(delayed_rewards, gamma)
        print(f"  γ = {gamma:.2f} → G_0 = {G:.6f}")

    # Scenario: immediate vs delayed
    print(f"\nImmediate (10 at t=0) vs Delayed (10 at t=50):")
    immediate = [10.0] + [0.0] * 99
    delayed = [0.0] * 50 + [10.0] + [0.0] * 49
    for gamma in [0.9, 0.95, 0.99]:
        G_imm = compute_return(immediate, gamma)
        G_del = compute_return(delayed, gamma)
        print(f"  γ = {gamma:.2f} → Immediate: {G_imm:.4f}, Delayed: {G_del:.4f}, "
              f"Ratio: {G_del/G_imm:.4f}")


# =============================================================================
# 2. Reward Design Examples
# =============================================================================

class RewardDesignDemo:
    """Demonstrates different reward function designs for a simple navigation task."""

    def __init__(self, grid_size: int = 10, goal: Tuple[int, int] = None):
        self.grid_size = grid_size
        self.goal = goal or (grid_size - 1, grid_size - 1)

    def sparse_reward(self, state: Tuple[int, int], action: int,
                      next_state: Tuple[int, int]) -> float:
        """Sparse reward: +1 at goal, 0 otherwise."""
        return 1.0 if next_state == self.goal else 0.0

    def step_penalty(self, state: Tuple[int, int], action: int,
                     next_state: Tuple[int, int]) -> float:
        """Step penalty: -1 per step, encourages efficiency."""
        return -1.0

    def distance_based(self, state: Tuple[int, int], action: int,
                       next_state: Tuple[int, int]) -> float:
        """Dense reward based on distance change to goal."""
        old_dist = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
        new_dist = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
        return (old_dist - new_dist) / self.grid_size

    def potential_based_shaping(self, state: Tuple[int, int], action: int,
                                 next_state: Tuple[int, int],
                                 gamma: float = 0.99) -> float:
        """
        Potential-based reward shaping: F(s,a,s') = γΦ(s') - Φ(s)
        This preserves the optimal policy (Ng et al., 1999).
        """
        def potential(s):
            return -abs(s[0] - self.goal[0]) - abs(s[1] - self.goal[1])

        base_reward = self.step_penalty(state, action, next_state)
        shaping = gamma * potential(next_state) - potential(state)
        return base_reward + shaping

    def compare_rewards(self):
        """Compare reward signals along a trajectory toward the goal."""
        # Simulated trajectory: diagonal path from (0,0) to goal
        trajectory = [(i, i) for i in range(min(self.grid_size, self.goal[0] + 1))]
        # Extend if needed
        if len(trajectory) > 1:
            last = trajectory[-1]
            while last != self.goal:
                if last[0] < self.goal[0]:
                    last = (last[0] + 1, last[1])
                else:
                    last = (last[0], last[1] + 1)
                trajectory.append(last)

        reward_funcs = {
            "Sparse": self.sparse_reward,
            "Step Penalty": self.step_penalty,
            "Distance-Based": self.distance_based,
            "Potential-Shaped": self.potential_based_shaping,
        }

        print("\n" + "=" * 60)
        print("Reward Function Comparison Along Trajectory")
        print("=" * 60)
        print(f"Trajectory: {trajectory[:5]}...{trajectory[-2:]}")

        for name, func in reward_funcs.items():
            rewards = []
            for i in range(len(trajectory) - 1):
                r = func(trajectory[i], 0, trajectory[i + 1])
                rewards.append(r)
            total = sum(rewards)
            print(f"\n{name}:")
            print(f"  Rewards: {[f'{r:.3f}' for r in rewards[:5]]}...{[f'{r:.3f}' for r in rewards[-2:]]}")
            print(f"  Total: {total:.4f}")

        return trajectory, reward_funcs


# =============================================================================
# 3. Financial Reward Functions
# =============================================================================

class FinancialRewards:
    """Collection of reward functions used in financial RL."""

    @staticmethod
    def simple_return(prices: np.ndarray, position: np.ndarray) -> np.ndarray:
        """
        Simple return: r_t = position_t * (P_{t+1} - P_t) / P_t
        """
        price_returns = np.diff(prices) / prices[:-1]
        return position[:-1] * price_returns

    @staticmethod
    def log_return(prices: np.ndarray, position: np.ndarray) -> np.ndarray:
        """
        Log return: r_t = position_t * log(P_{t+1} / P_t)
        Time-additive property makes this preferred for compounding.
        """
        log_returns = np.diff(np.log(prices))
        return position[:-1] * log_returns

    @staticmethod
    def risk_adjusted_return(prices: np.ndarray, position: np.ndarray,
                              window: int = 20) -> np.ndarray:
        """
        Sharpe-like reward: r_t = running_mean / running_std
        """
        simple_returns = np.diff(prices) / prices[:-1]
        portfolio_returns = position[:-1] * simple_returns

        rewards = np.zeros(len(portfolio_returns))
        for t in range(window, len(portfolio_returns)):
            window_returns = portfolio_returns[t - window:t]
            mu = np.mean(window_returns)
            sigma = np.std(window_returns) + 1e-8
            rewards[t] = mu / sigma

        return rewards

    @staticmethod
    def differential_sharpe(returns: np.ndarray, eta: float = 0.01) -> np.ndarray:
        """
        Differential Sharpe Ratio (Moody & Saffell, 2001).
        Provides dense reward signal that optimizes the Sharpe ratio.

        D_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} - A_{t-1}^2)^(3/2)
        """
        n = len(returns)
        A = 0.0  # EMA of first moment
        B = 0.0  # EMA of second moment
        dsr = np.zeros(n)

        for t in range(n):
            delta_A = returns[t] - A
            delta_B = returns[t] ** 2 - B

            denominator = (B - A ** 2) ** 1.5
            if abs(denominator) > 1e-10 and t > 0:
                dsr[t] = (B * delta_A - 0.5 * A * delta_B) / denominator
            else:
                dsr[t] = 0.0

            A = A + eta * delta_A
            B = B + eta * delta_B

        return dsr

    @staticmethod
    def drawdown_penalized(returns: np.ndarray, lambda_dd: float = 0.5) -> np.ndarray:
        """
        Return penalized by drawdown:
        r_t = R_t - λ * max(0, max_cumulative - cumulative_t) / max_cumulative
        """
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = np.where(running_max > 0,
                           (running_max - cumulative) / (running_max + 1e-8),
                           0.0)
        return returns - lambda_dd * drawdown

    @staticmethod
    def transaction_cost_adjusted(returns: np.ndarray, positions: np.ndarray,
                                   cost_rate: float = 0.001) -> np.ndarray:
        """
        Returns adjusted for transaction costs.
        r_t = portfolio_return_t - cost * |Δposition_t|
        """
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = cost_rate * position_changes[:len(returns)]
        return returns - costs


def demonstrate_financial_rewards():
    """Demonstrate different financial reward functions."""
    np.random.seed(42)

    # Generate synthetic price series
    n_days = 252
    dt = 1.0 / 252
    mu, sigma = 0.10, 0.20
    daily_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    prices = 100 * np.cumprod(np.concatenate([[1], 1 + daily_returns]))

    # Simulate a simple trend-following position
    sma_fast = np.convolve(prices, np.ones(5)/5, mode='same')
    sma_slow = np.convolve(prices, np.ones(20)/20, mode='same')
    positions = np.where(sma_fast > sma_slow, 1.0, -1.0)

    fr = FinancialRewards()

    print("\n" + "=" * 60)
    print("Financial Reward Functions Comparison")
    print("=" * 60)

    # Compute various rewards
    simple_ret = fr.simple_return(prices, positions)
    log_ret = fr.log_return(prices, positions)
    risk_adj = fr.risk_adjusted_return(prices, positions, window=20)
    dsr = fr.differential_sharpe(simple_ret, eta=0.01)
    dd_pen = fr.drawdown_penalized(simple_ret, lambda_dd=0.5)
    tc_adj = fr.transaction_cost_adjusted(simple_ret, positions, cost_rate=0.001)

    rewards_dict = {
        "Simple Return": simple_ret,
        "Log Return": log_ret,
        "Risk-Adjusted": risk_adj,
        "Diff. Sharpe": dsr,
        "Drawdown-Penalized": dd_pen,
        "TC-Adjusted": tc_adj,
    }

    for name, rewards in rewards_dict.items():
        print(f"\n{name}:")
        print(f"  Mean:   {np.mean(rewards):.6f}")
        print(f"  Std:    {np.std(rewards):.6f}")
        print(f"  Total:  {np.sum(rewards):.4f}")
        print(f"  Sharpe: {np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252):.4f}")

    return prices, positions, rewards_dict


# =============================================================================
# 4. Visualization
# =============================================================================

def plot_discount_effects():
    """Visualize how discount factor affects value of future rewards."""
    gammas = [0.5, 0.9, 0.95, 0.99, 1.0]
    steps = np.arange(100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Discount curves
    ax = axes[0]
    for gamma in gammas:
        weights = gamma ** steps
        ax.plot(steps, weights, label=f'γ={gamma}')
    ax.set_xlabel('Steps into Future (k)')
    ax.set_ylabel('Discount Weight γ^k')
    ax.set_title('Discount Factor: Weight of Future Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effective horizon
    ax = axes[1]
    gammas_fine = np.linspace(0.01, 0.999, 100)
    effective_horizon = 1.0 / (1.0 - gammas_fine)
    ax.plot(gammas_fine, effective_horizon)
    ax.set_xlabel('Discount Factor γ')
    ax.set_ylabel('Effective Horizon 1/(1-γ)')
    ax.set_title('Effective Planning Horizon')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100 steps')
    ax.axhline(y=1000, color='g', linestyle='--', alpha=0.5, label='1000 steps')
    ax.legend()

    plt.tight_layout()
    plt.savefig("reward_discount_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reward_discount_effects.png")


def plot_financial_rewards(prices, positions, rewards_dict):
    """Visualize financial reward functions."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, rewards) in enumerate(rewards_dict.items()):
        ax = axes[idx]
        ax.plot(rewards, alpha=0.7, linewidth=0.8)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.2)

        # Add cumulative return on secondary axis
        ax2 = ax.twinx()
        ax2.plot(np.cumsum(rewards), color='red', alpha=0.4, linewidth=1)
        ax2.set_ylabel('Cumulative', color='red', alpha=0.6)

    plt.suptitle('Financial Reward Functions Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("financial_reward_functions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: financial_reward_functions.png")


def plot_reward_shaping():
    """Visualize effect of reward shaping."""
    demo = RewardDesignDemo(grid_size=10)

    # Create trajectory
    trajectory = [(0, 0)]
    for i in range(1, 10):
        trajectory.append((i, i))

    reward_funcs = {
        "Sparse": demo.sparse_reward,
        "Step Penalty": demo.step_penalty,
        "Distance": demo.distance_based,
        "Potential-Shaped": demo.potential_based_shaping,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(reward_funcs.items()):
        rewards = [func(trajectory[i], 0, trajectory[i + 1])
                   for i in range(len(trajectory) - 1)]

        ax = axes[idx]
        ax.bar(range(len(rewards)), rewards, alpha=0.7)
        ax.set_title(f'{name} Reward')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Reward Function Design: Sparse vs Dense vs Shaped', fontsize=14)
    plt.tight_layout()
    plt.savefig("reward_shaping_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reward_shaping_comparison.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Discount factor effects
    demonstrate_discount_effects()

    # 2. Reward design comparison
    demo = RewardDesignDemo(grid_size=10)
    demo.compare_rewards()

    # 3. Financial reward functions
    prices, positions, rewards_dict = demonstrate_financial_rewards()

    # 4. Visualizations
    plot_discount_effects()
    plot_financial_rewards(prices, positions, rewards_dict)
    plot_reward_shaping()

    print("\n✓ Reward Hypothesis demonstrations complete.")
