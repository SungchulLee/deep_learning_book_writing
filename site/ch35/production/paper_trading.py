"""
Chapter 35.7.2: Paper Trading
================================
Paper trading validation system.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class PaperTradingConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001
    slippage_model: str = "gaussian"
    slippage_std: float = 0.0005
    max_position: float = 0.25


class PaperTradingEngine:
    """Paper trading engine with realistic execution simulation."""

    def __init__(self, config: PaperTradingConfig, num_assets: int):
        self.config = config
        self.num_assets = num_assets
        self.portfolio_value = config.initial_capital
        self.peak_value = config.initial_capital
        self.weights = np.zeros(num_assets)
        self.step = 0

        self.value_history: List[float] = [config.initial_capital]
        self.return_history: List[float] = []
        self.trade_log: List[Dict] = []
        self.daily_pnl: List[float] = []

    def execute(self, target_weights: np.ndarray, prices: np.ndarray,
                next_prices: np.ndarray) -> Dict:
        target_weights = np.clip(target_weights, -self.config.max_position, self.config.max_position)

        turnover = np.sum(np.abs(target_weights - self.weights))
        tc = self.config.transaction_cost * turnover

        # Slippage
        slippage = self.config.slippage_std * np.sqrt(turnover) * abs(np.random.randn())

        returns = (next_prices - prices) / (prices + 1e-8)
        port_return = float(np.dot(target_weights, returns)) - tc - slippage

        self.portfolio_value *= (1 + port_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        self.weights = target_weights.copy()
        self.value_history.append(self.portfolio_value)
        self.return_history.append(port_return)
        self.step += 1

        self.trade_log.append({
            "step": self.step,
            "turnover": turnover,
            "tc": tc,
            "slippage": slippage,
            "return": port_return,
        })

        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        return {
            "portfolio_value": self.portfolio_value,
            "return": port_return,
            "drawdown": drawdown,
            "turnover": turnover,
        }

    def get_performance_summary(self) -> Dict:
        returns = np.array(self.return_history)
        if len(returns) == 0:
            return {"no_data": True}

        total_return = (self.portfolio_value / self.config.initial_capital - 1)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_dd = (self.peak_value - min(self.value_history)) / (self.peak_value + 1e-8)
        avg_turnover = np.mean([t["turnover"] for t in self.trade_log])
        total_tc = sum(t["tc"] + t["slippage"] for t in self.trade_log)

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "avg_turnover": float(avg_turnover),
            "total_costs": float(total_tc * self.config.initial_capital),
            "num_steps": self.step,
        }


class BacktestPaperComparator:
    """Compare backtest results with paper trading results."""

    @staticmethod
    def compare(backtest_returns: np.ndarray, paper_returns: np.ndarray) -> Dict:
        bt_sharpe = np.mean(backtest_returns) / (np.std(backtest_returns) + 1e-8) * np.sqrt(252)
        pt_sharpe = np.mean(paper_returns) / (np.std(paper_returns) + 1e-8) * np.sqrt(252)

        bt_vol = np.std(backtest_returns) * np.sqrt(252)
        pt_vol = np.std(paper_returns) * np.sqrt(252)

        return {
            "backtest_sharpe": float(bt_sharpe),
            "paper_sharpe": float(pt_sharpe),
            "sharpe_diff": float(bt_sharpe - pt_sharpe),
            "backtest_vol": float(bt_vol),
            "paper_vol": float(pt_vol),
            "correlation": float(np.corrcoef(
                backtest_returns[:min(len(backtest_returns), len(paper_returns))],
                paper_returns[:min(len(backtest_returns), len(paper_returns))]
            )[0, 1]) if len(backtest_returns) > 1 and len(paper_returns) > 1 else 0,
        }


def demo_paper_trading():
    """Demonstrate paper trading."""
    print("=" * 70)
    print("Paper Trading Demonstration")
    print("=" * 70)

    np.random.seed(42)
    N, T = 5, 200
    returns_data = np.random.randn(T, N) * 0.015 + 0.0003
    prices = 100 * np.exp(np.cumsum(returns_data, axis=0))

    config = PaperTradingConfig(transaction_cost=0.001)
    engine = PaperTradingEngine(config, N)

    for t in range(T - 1):
        weights = np.random.dirichlet(np.ones(N))
        result = engine.execute(weights, prices[t], prices[t + 1])

    summary = engine.get_performance_summary()
    print(f"\n--- Paper Trading Summary ---")
    for k, v in summary.items():
        if "return" in k or "drawdown" in k:
            print(f"  {k:<20}: {v*100:.2f}%")
        elif "ratio" in k:
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")

    # Compare with backtest
    print("\n--- Backtest vs Paper Comparison ---")
    bt_returns = np.random.randn(200) * 0.012 + 0.0004
    pt_returns = np.array(engine.return_history)
    comp = BacktestPaperComparator.compare(bt_returns, pt_returns)
    for k, v in comp.items():
        print(f"  {k:<20}: {v:.4f}")


if __name__ == "__main__":
    demo_paper_trading()
