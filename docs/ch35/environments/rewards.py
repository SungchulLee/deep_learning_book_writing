"""
Chapter 35.1.4: Reward Engineering for Financial RL
====================================================
Reward functions, risk-adjusted metrics, and reward shaping.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Reward Functions
# ============================================================

class SimpleReturnReward:
    """Raw or log portfolio return as reward."""
    
    def __init__(self, use_log: bool = True):
        self.use_log = use_log
    
    def compute(self, portfolio_return: float, **kwargs) -> float:
        if self.use_log:
            return np.log(1 + portfolio_return + 1e-10)
        return portfolio_return


class DifferentialSharpeReward:
    """
    Differential Sharpe Ratio (Moody & Saffell, 2001).
    
    An incremental approximation of the Sharpe ratio suitable for
    online/sequential optimization.
    
    DSR_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}
    
    where A_t and B_t are EMAs of returns and squared returns.
    """
    
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.reset()
    
    def reset(self):
        self._A = 0.0  # EMA of returns
        self._B = 1e-6  # EMA of squared returns (small init to avoid div by 0)
    
    def compute(self, portfolio_return: float, **kwargs) -> float:
        R = portfolio_return
        
        delta_A = R - self._A
        delta_B = R ** 2 - self._B
        
        denom = (self._B - self._A ** 2) ** 1.5
        
        if abs(denom) < 1e-12:
            dsr = R  # Fallback for insufficient history
        else:
            dsr = (self._B * delta_A - 0.5 * self._A * delta_B) / denom
        
        # Update EMAs
        self._A += self.eta * delta_A
        self._B += self.eta * delta_B
        
        return float(dsr)


class SortinoReward:
    """
    Sortino-based reward: penalizes downside risk more heavily.
    
    r_t = R_t - λ * max(0, -R_t)^2
    """
    
    def __init__(self, penalty: float = 1.0, target_return: float = 0.0):
        self.penalty = penalty
        self.target_return = target_return
    
    def compute(self, portfolio_return: float, **kwargs) -> float:
        R = portfolio_return
        downside = max(0, self.target_return - R)
        return R - self.penalty * downside ** 2


class RiskAdjustedReward:
    """
    Return minus risk penalty.
    
    r_t = R_t - λ_dd * DD_t - λ_var * Var_t
    """
    
    def __init__(self, drawdown_penalty: float = 0.5,
                 variance_penalty: float = 0.0,
                 return_window: int = 20):
        self.dd_penalty = drawdown_penalty
        self.var_penalty = variance_penalty
        self.return_window = return_window
        self._returns = []
        self._peak_value = 1.0
        self._current_value = 1.0
    
    def reset(self):
        self._returns = []
        self._peak_value = 1.0
        self._current_value = 1.0
    
    def compute(self, portfolio_return: float, **kwargs) -> float:
        self._returns.append(portfolio_return)
        self._current_value *= (1 + portfolio_return)
        self._peak_value = max(self._peak_value, self._current_value)
        
        # Drawdown
        drawdown = (self._peak_value - self._current_value) / self._peak_value
        
        # Rolling variance
        if len(self._returns) >= self.return_window:
            recent = self._returns[-self.return_window:]
            variance = np.var(recent)
        else:
            variance = 0.0
        
        reward = portfolio_return - self.dd_penalty * drawdown - self.var_penalty * variance
        return float(reward)


class BenchmarkRelativeReward:
    """
    Reward relative to a benchmark.
    
    r_t = α * (R_p - R_b) - β * (R_p - R_b - ᾱ)^2
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        self.alpha_coeff = alpha
        self.beta = beta
        self._excess_returns = []
    
    def reset(self):
        self._excess_returns = []
    
    def compute(self, portfolio_return: float,
                benchmark_return: float = 0.0, **kwargs) -> float:
        excess = portfolio_return - benchmark_return
        self._excess_returns.append(excess)
        
        mean_excess = np.mean(self._excess_returns) if self._excess_returns else 0
        tracking_penalty = self.beta * (excess - mean_excess) ** 2
        
        return self.alpha_coeff * excess - tracking_penalty


class MultiObjectiveReward:
    """
    Weighted combination of multiple reward components.
    
    r_t = Σ w_i * component_i(t)
    """
    
    def __init__(self, return_weight: float = 1.0,
                 risk_weight: float = 0.5,
                 cost_weight: float = 1.0,
                 turnover_weight: float = 0.1):
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.cost_weight = cost_weight
        self.turnover_weight = turnover_weight
        self._peak = 1.0
        self._value = 1.0
    
    def reset(self):
        self._peak = 1.0
        self._value = 1.0
    
    def compute(self, portfolio_return: float,
                transaction_costs: float = 0.0,
                turnover: float = 0.0, **kwargs) -> float:
        self._value *= (1 + portfolio_return)
        self._peak = max(self._peak, self._value)
        drawdown = (self._peak - self._value) / self._peak
        
        reward = (
            self.return_weight * portfolio_return
            - self.risk_weight * drawdown
            - self.cost_weight * transaction_costs
            - self.turnover_weight * turnover
        )
        return float(reward)


# ============================================================
# Reward Shaping
# ============================================================

class PotentialBasedShaping:
    """
    Potential-based reward shaping (Ng et al., 1999).
    
    r'_t = r_t + γ * Φ(s_{t+1}) - Φ(s_t)
    
    Preserves optimal policy under the shaped reward.
    """
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self._prev_potential = 0.0
    
    def reset(self):
        self._prev_potential = 0.0
    
    def shape(self, reward: float, potential: float) -> float:
        """
        Apply potential-based shaping.
        
        Args:
            reward: Original reward
            potential: Potential function Φ(s_{t+1})
        """
        shaped = reward + self.gamma * potential - self._prev_potential
        self._prev_potential = potential
        return shaped


def sharpe_potential(returns: List[float], window: int = 20,
                    annualization: float = np.sqrt(252)) -> float:
    """Use rolling Sharpe ratio as potential function."""
    if len(returns) < window:
        return 0.0
    recent = np.array(returns[-window:])
    if recent.std() < 1e-10:
        return 0.0
    return annualization * recent.mean() / recent.std()


class RewardNormalizer:
    """
    Running normalization for reward scaling.
    
    Maintains exponential moving statistics and normalizes
    rewards to approximately unit variance.
    """
    
    def __init__(self, alpha: float = 0.001, epsilon: float = 1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self._mean = 0.0
        self._var = 1.0
        self._count = 0
    
    def normalize(self, reward: float) -> float:
        self._count += 1
        
        if self._count == 1:
            self._mean = reward
            return 0.0
        
        self._mean = self.alpha * reward + (1 - self.alpha) * self._mean
        self._var = self.alpha * (reward - self._mean) ** 2 + (1 - self.alpha) * self._var
        
        return (reward - self._mean) / (np.sqrt(self._var) + self.epsilon)
    
    def reset(self):
        self._mean = 0.0
        self._var = 1.0
        self._count = 0


# ============================================================
# Reward Factory
# ============================================================

def create_reward(reward_type: str, **kwargs):
    """Factory function for creating reward objects."""
    reward_map = {
        'simple': SimpleReturnReward,
        'log_return': lambda **kw: SimpleReturnReward(use_log=True),
        'sharpe': DifferentialSharpeReward,
        'sortino': SortinoReward,
        'risk_adjusted': RiskAdjustedReward,
        'benchmark': BenchmarkRelativeReward,
        'multi_objective': MultiObjectiveReward,
    }
    
    if reward_type not in reward_map:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                         f"Available: {list(reward_map.keys())}")
    
    creator = reward_map[reward_type]
    if callable(creator) and not isinstance(creator, type):
        return creator(**kwargs)
    return creator(**kwargs)


# ============================================================
# Demo
# ============================================================

def demo_reward_engineering():
    """Demonstrate reward function designs."""
    print("=" * 60)
    print("Reward Engineering Demo")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate a trajectory of returns
    T = 252
    daily_returns = np.random.normal(0.0003, 0.015, T)
    # Add a drawdown period
    daily_returns[100:130] = np.random.normal(-0.005, 0.02, 30)
    
    print(f"\nSimulated {T} days of returns")
    print(f"Mean daily return: {daily_returns.mean():.6f}")
    print(f"Std daily return:  {daily_returns.std():.6f}")
    print(f"Annualized Sharpe: {np.sqrt(252) * daily_returns.mean() / daily_returns.std():.4f}")
    
    # 1. Compare reward functions
    print("\n--- Reward Function Comparison ---")
    
    rewards = {
        'Simple': SimpleReturnReward(use_log=False),
        'Log Return': SimpleReturnReward(use_log=True),
        'Diff Sharpe': DifferentialSharpeReward(eta=0.01),
        'Sortino': SortinoReward(penalty=1.0),
        'Risk-Adjusted': RiskAdjustedReward(drawdown_penalty=0.5),
        'Multi-Objective': MultiObjectiveReward(return_weight=1.0, risk_weight=0.5),
    }
    
    reward_traces = {name: [] for name in rewards}
    
    for t in range(T):
        for name, reward_fn in rewards.items():
            r = reward_fn.compute(
                portfolio_return=daily_returns[t],
                transaction_costs=0.0001,
                turnover=0.05,
            )
            reward_traces[name].append(r)
    
    print(f"\n{'Reward Type':<18} {'Mean':>10} {'Std':>10} {'Sum':>10} {'Min':>10} {'Max':>10}")
    print("-" * 68)
    for name, trace in reward_traces.items():
        arr = np.array(trace)
        print(f"{name:<18} {arr.mean():10.6f} {arr.std():10.6f} "
              f"{arr.sum():10.4f} {arr.min():10.6f} {arr.max():10.6f}")
    
    # 2. Reward shaping
    print("\n--- Potential-Based Reward Shaping ---")
    
    shaper = PotentialBasedShaping(gamma=0.99)
    base_reward = SimpleReturnReward(use_log=True)
    
    returns_list = []
    shaped_rewards = []
    base_rewards = []
    
    for t in range(T):
        r_base = base_reward.compute(daily_returns[t])
        base_rewards.append(r_base)
        returns_list.append(daily_returns[t])
        
        potential = sharpe_potential(returns_list, window=20)
        r_shaped = shaper.shape(r_base, potential)
        shaped_rewards.append(r_shaped)
    
    base_arr = np.array(base_rewards)
    shaped_arr = np.array(shaped_rewards)
    
    print(f"Base rewards  - mean: {base_arr.mean():.6f}, std: {base_arr.std():.6f}")
    print(f"Shaped rewards - mean: {shaped_arr.mean():.6f}, std: {shaped_arr.std():.6f}")
    print(f"Correlation:    {np.corrcoef(base_arr, shaped_arr)[0, 1]:.4f}")
    
    # 3. Reward normalization
    print("\n--- Reward Normalization ---")
    
    normalizer = RewardNormalizer(alpha=0.01)
    normalized = []
    for t in range(T):
        n = normalizer.normalize(daily_returns[t])
        normalized.append(n)
    
    norm_arr = np.array(normalized[10:])  # Skip warmup
    print(f"Raw rewards    - mean: {daily_returns[10:].mean():.6f}, "
          f"std: {daily_returns[10:].std():.6f}")
    print(f"Normalized     - mean: {norm_arr.mean():.6f}, std: {norm_arr.std():.4f}")
    
    # 4. Benchmark-relative reward
    print("\n--- Benchmark-Relative Reward ---")
    
    benchmark_returns = np.random.normal(0.0002, 0.012, T)
    bench_reward = BenchmarkRelativeReward(alpha=1.0, beta=0.5)
    
    bench_rewards = []
    for t in range(T):
        r = bench_reward.compute(daily_returns[t], benchmark_returns[t])
        bench_rewards.append(r)
    
    bench_arr = np.array(bench_rewards)
    excess = daily_returns - benchmark_returns
    print(f"Mean excess return: {excess.mean():.6f}")
    print(f"Mean bench reward:  {bench_arr.mean():.6f}")
    print(f"Info ratio:         {np.sqrt(252) * excess.mean() / excess.std():.4f}")
    
    # 5. Impact of drawdown penalty
    print("\n--- Drawdown Penalty Impact ---")
    
    for dd_penalty in [0.0, 0.5, 1.0, 2.0]:
        ra = RiskAdjustedReward(drawdown_penalty=dd_penalty)
        trace = [ra.compute(daily_returns[t]) for t in range(T)]
        arr = np.array(trace)
        print(f"DD penalty={dd_penalty:.1f}: mean={arr.mean():.6f}, "
              f"std={arr.std():.6f}, sum={arr.sum():.4f}")
    
    # 6. Reward factory
    print("\n--- Reward Factory ---")
    for rtype in ['log_return', 'sharpe', 'sortino', 'risk_adjusted', 'multi_objective']:
        r = create_reward(rtype)
        vals = [r.compute(daily_returns[t], transaction_costs=0.0001, turnover=0.05)
                for t in range(min(50, T))]
        print(f"  {rtype:<18}: mean={np.mean(vals):.6f}")
    
    print("\nReward engineering demo complete!")


if __name__ == "__main__":
    demo_reward_engineering()
