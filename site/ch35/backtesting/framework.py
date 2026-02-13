"""
Chapter 35.6.1: Backtesting Framework
=======================================
Comprehensive backtesting engine for RL trading strategies.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001
    slippage_std: float = 0.0005
    max_leverage: float = 1.0
    risk_free_rate: float = 0.02 / 252


class PortfolioTracker:
    """Track portfolio state and history during backtest."""

    def __init__(self, initial_capital: float, num_assets: int):
        self.initial_capital = initial_capital
        self.num_assets = num_assets
        self.cash = initial_capital
        self.positions = np.zeros(num_assets)
        self.weights = np.zeros(num_assets)
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital

        self.value_history: List[float] = [initial_capital]
        self.return_history: List[float] = []
        self.weight_history: List[np.ndarray] = []
        self.turnover_history: List[float] = []
        self.cost_history: List[float] = []

    def update(self, new_weights: np.ndarray, prices: np.ndarray,
               next_prices: np.ndarray, tc_rate: float,
               slippage_std: float = 0.0) -> Dict:
        old_weights = self.weights.copy()
        turnover = np.sum(np.abs(new_weights - old_weights))
        tc = tc_rate * turnover * self.portfolio_value
        slippage = slippage_std * np.sqrt(turnover) * self.portfolio_value * np.abs(np.random.randn())

        returns = (next_prices - prices) / (prices + 1e-8)
        port_return = float(np.dot(new_weights, returns))
        net_return = port_return - (tc + slippage) / (self.portfolio_value + 1e-8)

        self.portfolio_value *= (1 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Drift weights
        drifted = new_weights * (1 + returns)
        self.weights = drifted / (np.sum(drifted) + 1e-8) if np.sum(drifted) > 0 else np.zeros(self.num_assets)

        self.value_history.append(self.portfolio_value)
        self.return_history.append(net_return)
        self.weight_history.append(new_weights.copy())
        self.turnover_history.append(turnover)
        self.cost_history.append(tc + slippage)

        return {
            "portfolio_value": self.portfolio_value,
            "return": net_return,
            "turnover": turnover,
            "cost": tc + slippage,
            "drawdown": (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8),
        }


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(self, prices: np.ndarray, config: BacktestConfig):
        self.prices = prices
        self.config = config
        self.num_steps = len(prices) - 1
        self.num_assets = prices.shape[1]

    def run(self, strategy: Callable) -> Dict:
        """
        Run backtest with a strategy function.

        Args:
            strategy: function(prices_history, current_weights, step) -> new_weights
        """
        tracker = PortfolioTracker(self.config.initial_capital, self.num_assets)

        for t in range(self.num_steps):
            prices_so_far = self.prices[:t + 1]
            new_weights = strategy(prices_so_far, tracker.weights, t)

            # Ensure valid weights
            if np.any(np.isnan(new_weights)):
                new_weights = tracker.weights.copy()

            tracker.update(
                new_weights, self.prices[t], self.prices[t + 1],
                self.config.transaction_cost, self.config.slippage_std,
            )

        return {
            "value_history": np.array(tracker.value_history),
            "return_history": np.array(tracker.return_history),
            "weight_history": np.array(tracker.weight_history),
            "turnover_history": np.array(tracker.turnover_history),
            "cost_history": np.array(tracker.cost_history),
            "final_value": tracker.portfolio_value,
            "total_return": (tracker.portfolio_value / self.config.initial_capital - 1),
        }


# Benchmark strategies
def equal_weight_strategy(prices, weights, step):
    N = prices.shape[1]
    return np.ones(N) / N

def buy_and_hold_strategy(prices, weights, step):
    if step == 0:
        return np.ones(prices.shape[1]) / prices.shape[1]
    return weights  # Don't rebalance


def demo_backtesting():
    """Demonstrate backtesting framework."""
    print("=" * 70)
    print("Backtesting Framework Demonstration")
    print("=" * 70)

    np.random.seed(42)
    N, T = 5, 500
    returns = np.random.randn(T, N) * 0.015 + 0.0003
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    config = BacktestConfig(transaction_cost=0.001, slippage_std=0.0003)
    engine = BacktestEngine(prices, config)

    # Equal weight
    result_ew = engine.run(equal_weight_strategy)
    print(f"\nEqual Weight: Return={result_ew['total_return']*100:.2f}%, "
          f"Avg Turnover={np.mean(result_ew['turnover_history']):.4f}")

    # Buy and hold
    result_bh = engine.run(buy_and_hold_strategy)
    print(f"Buy & Hold:   Return={result_bh['total_return']*100:.2f}%, "
          f"Avg Turnover={np.mean(result_bh['turnover_history']):.4f}")

    # Momentum strategy
    def momentum_strat(prices_hist, weights, step):
        if len(prices_hist) < 21:
            return np.ones(N) / N
        ret = (prices_hist[-1] / prices_hist[-21]) - 1
        w = np.maximum(ret, 0)
        return w / (np.sum(w) + 1e-8) if np.sum(w) > 0 else np.ones(N) / N

    result_mom = engine.run(momentum_strat)
    print(f"Momentum:     Return={result_mom['total_return']*100:.2f}%, "
          f"Avg Turnover={np.mean(result_mom['turnover_history']):.4f}")

    total_costs = np.sum(result_mom['cost_history'])
    print(f"\nMomentum total trading costs: ${total_costs:,.0f}")


if __name__ == "__main__":
    demo_backtesting()
