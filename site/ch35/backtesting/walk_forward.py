"""
Chapter 35.6.2: Walk-Forward Analysis
========================================
Walk-forward validation for RL strategy evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class WalkForwardConfig:
    train_window: int = 252
    test_window: int = 63
    gap: int = 5
    expanding: bool = False
    min_train: int = 126


class WalkForwardAnalyzer:
    """Walk-forward analysis engine."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_splits(self, T: int) -> List[Dict]:
        splits = []
        cfg = self.config

        if cfg.expanding:
            test_start = cfg.min_train + cfg.gap
            while test_start + cfg.test_window <= T:
                splits.append({
                    "train_start": 0,
                    "train_end": test_start - cfg.gap,
                    "test_start": test_start,
                    "test_end": min(test_start + cfg.test_window, T),
                })
                test_start += cfg.test_window
        else:
            start = 0
            while start + cfg.train_window + cfg.gap + cfg.test_window <= T:
                splits.append({
                    "train_start": start,
                    "train_end": start + cfg.train_window,
                    "test_start": start + cfg.train_window + cfg.gap,
                    "test_end": start + cfg.train_window + cfg.gap + cfg.test_window,
                })
                start += cfg.test_window

        return splits

    def run(self, returns: np.ndarray,
            train_fn: Callable, eval_fn: Callable) -> Dict:
        """
        Run walk-forward analysis.

        Args:
            returns: (T, N) or (T,) return series
            train_fn: function(train_returns) -> model/weights
            eval_fn: function(model, test_returns) -> strategy_returns
        """
        T = len(returns)
        splits = self.generate_splits(T)
        results = []

        for i, split in enumerate(splits):
            train_r = returns[split["train_start"]:split["train_end"]]
            test_r = returns[split["test_start"]:split["test_end"]]

            model = train_fn(train_r)
            strat_returns = eval_fn(model, test_r)

            is_sharpe = np.mean(train_r if train_r.ndim == 1 else train_r.mean(1)) / (
                np.std(train_r if train_r.ndim == 1 else train_r.mean(1)) + 1e-8) * np.sqrt(252)
            oos_sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)

            results.append({
                "split": i,
                "train_size": split["train_end"] - split["train_start"],
                "test_size": split["test_end"] - split["test_start"],
                "is_sharpe": float(is_sharpe),
                "oos_sharpe": float(oos_sharpe),
                "oos_return": float(np.sum(strat_returns)),
                "oos_returns": strat_returns,
            })

        # Aggregate
        oos_sharpes = [r["oos_sharpe"] for r in results]
        is_sharpes = [r["is_sharpe"] for r in results]
        all_oos = np.concatenate([r["oos_returns"] for r in results])

        return {
            "splits": results,
            "mean_oos_sharpe": float(np.mean(oos_sharpes)),
            "std_oos_sharpe": float(np.std(oos_sharpes)),
            "mean_is_sharpe": float(np.mean(is_sharpes)),
            "degradation": float(np.mean(is_sharpes) - np.mean(oos_sharpes)),
            "aggregate_sharpe": float(np.mean(all_oos) / (np.std(all_oos) + 1e-8) * np.sqrt(252)),
            "num_splits": len(results),
        }


def demo_walk_forward():
    """Demonstrate walk-forward analysis."""
    print("=" * 70)
    print("Walk-Forward Analysis Demonstration")
    print("=" * 70)

    np.random.seed(42)
    T = 1000
    returns = np.random.randn(T) * 0.015 + 0.0002

    # Simple momentum strategy
    def train_fn(train_r):
        return {"signal": np.sign(np.mean(train_r))}

    def eval_fn(model, test_r):
        return test_r * model["signal"]

    for expanding in [False, True]:
        name = "Expanding" if expanding else "Rolling"
        config = WalkForwardConfig(
            train_window=252, test_window=63, gap=5, expanding=expanding
        )
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(returns, train_fn, eval_fn)

        print(f"\n--- {name} Walk-Forward ---")
        print(f"Splits: {result['num_splits']}")
        print(f"Mean IS Sharpe:  {result['mean_is_sharpe']:.4f}")
        print(f"Mean OOS Sharpe: {result['mean_oos_sharpe']:.4f}")
        print(f"Degradation:     {result['degradation']:.4f}")
        print(f"Aggregate OOS:   {result['aggregate_sharpe']:.4f}")


if __name__ == "__main__":
    demo_walk_forward()
