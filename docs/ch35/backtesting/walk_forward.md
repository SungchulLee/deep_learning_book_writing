# Walk-Forward Analysis

Walk-forward analysis is the gold standard for evaluating time series strategies. Unlike standard cross-validation, it respects the temporal ordering of data by training on past data and testing on future data in a rolling or expanding window fashion. This approach provides realistic estimates of out-of-sample performance and reveals how strategy effectiveness varies across market regimes.

## Code

```python
"""
Chapter 35.6.2: Walk-Forward Analysis
========================================
Walk-forward validation for RL strategy evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

# ========================================================================
# Main
# ========================================================================


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
```

## Discussion

The walk-forward analyzer generates non-overlapping train-test splits that respect temporal ordering. Each split consists of a training window (where the strategy is optimized), an optional gap (to prevent look-ahead bias from data leakage), and a test window (where performance is evaluated). The process slides forward through time, producing multiple out-of-sample performance estimates that can be aggregated into an overall assessment.

Two windowing schemes are commonly used: rolling and expanding. Rolling windows use a fixed-size training period that moves forward in time, giving equal weight to recent and distant history within the window. Expanding windows grow the training set over time, incorporating all available history. Rolling windows are preferred when market dynamics change (non-stationarity), while expanding windows are better when more data consistently improves model quality.

The degradation metric -- the difference between average in-sample and out-of-sample Sharpe ratios -- is a direct measure of overfitting. Large degradation indicates that the strategy captures noise rather than signal. A well-designed strategy should show consistent out-of-sample performance across splits, with degradation close to zero. Monitoring per-split performance also reveals regime dependence: a strategy that works in some periods but fails in others may be exploiting transient market conditions rather than a persistent edge.

## Exercises

**Exercise 1.**
For a dataset of 1000 daily returns, compute the number of walk-forward splits using a rolling window with train=252, test=63, and gap=5. How many out-of-sample data points are generated?

??? success "Solution to Exercise 1"
    Each split consumes $252 + 5 + 63 = 320$ days. The step size equals the test window (63 days). Starting from day 0:

    Split 1: train [0, 252), test [257, 320)
    Split 2: train [63, 315), test [320, 383)
    ...
    
    Number of splits $= \lfloor (1000 - 320) / 63 \rfloor + 1 = \lfloor 680/63 \rfloor + 1 = 10 + 1 = 11$ (approximately 10-11 splits).
    
    Total out-of-sample days $= 11 \times 63 = 693$ days (covering most of the dataset beyond the first training window).

---


**Exercise 2.**
Explain why a gap between training and test windows is important in walk-forward analysis for financial data. Give an example of a specific look-ahead bias it prevents.

??? success "Solution to Exercise 2"
    The gap prevents information leakage from features that span the train-test boundary. For example, if a strategy uses a 20-day moving average as a feature, the moving average computed on the last training day includes information from the 20 days prior. Without a gap, the first few test days' features partially overlap with training data, inflating apparent performance.

    A 5-day gap ensures that even features with short lookbacks do not leak training information into the test period. For features with longer lookbacks (e.g., 60-day volatility), a correspondingly longer gap may be needed. The gap also accounts for the practical delay between model development (training) and deployment (testing) in real trading systems.

---


**Exercise 3.**
Design a walk-forward analysis that uses an expanding window and computes a stability metric defined as the coefficient of variation (standard deviation divided by mean) of out-of-sample Sharpe ratios across splits.

??? success "Solution to Exercise 3"
    ```python
    def expanding_walk_forward(returns, train_fn, eval_fn, 
                               min_train=252, test_window=63, gap=5):
        T = len(returns)
        splits = []
        test_start = min_train + gap
        
        while test_start + test_window <= T:
            train_r = returns[:test_start - gap]
            test_r = returns[test_start:test_start + test_window]
            
            model = train_fn(train_r)
            strat_returns = eval_fn(model, test_r)
            sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)
            splits.append(sharpe)
            test_start += test_window
        
        splits = np.array(splits)
        stability = np.std(splits) / (np.abs(np.mean(splits)) + 1e-8)
        
        return {'mean_oos_sharpe': np.mean(splits),
                'stability_cv': stability,
                'consistent': stability < 1.0}
    ```
    A stability coefficient of variation below 1.0 indicates the strategy produces consistently positive Sharpe ratios across periods. Values above 2.0 suggest highly regime-dependent performance that may not persist.

