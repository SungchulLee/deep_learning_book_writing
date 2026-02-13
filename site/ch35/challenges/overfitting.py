"""
Chapter 35.5.4: Overfitting Prevention
========================================
Walk-forward validation, regularization, and statistical tests
for overfitting prevention in financial RL.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OverfittingConfig:
    train_window: int = 252
    test_window: int = 63
    gap: int = 5
    n_splits: int = 5
    significance_level: float = 0.05


class WalkForwardValidator:
    """Walk-forward cross-validation for time series."""

    def __init__(self, config: OverfittingConfig):
        self.config = config

    def generate_splits(self, T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        splits = []
        train_w = self.config.train_window
        test_w = self.config.test_window
        gap = self.config.gap

        start = 0
        while start + train_w + gap + test_w <= T:
            train_idx = np.arange(start, start + train_w)
            test_idx = np.arange(start + train_w + gap, start + train_w + gap + test_w)
            splits.append((train_idx, test_idx))
            start += test_w
        return splits

    def evaluate(self, returns: np.ndarray, strategy_returns_func) -> Dict:
        T = len(returns)
        splits = self.generate_splits(T)

        is_sharpes = []
        oos_sharpes = []

        for train_idx, test_idx in splits:
            train_r = strategy_returns_func(returns, train_idx)
            test_r = strategy_returns_func(returns, test_idx)

            is_sharpe = np.mean(train_r) / (np.std(train_r) + 1e-8) * np.sqrt(252)
            oos_sharpe = np.mean(test_r) / (np.std(test_r) + 1e-8) * np.sqrt(252)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

        return {
            "is_sharpes": is_sharpes,
            "oos_sharpes": oos_sharpes,
            "mean_is": float(np.mean(is_sharpes)),
            "mean_oos": float(np.mean(oos_sharpes)),
            "overfit_ratio": float(np.mean(is_sharpes)) / (float(np.mean(oos_sharpes)) + 1e-8),
            "num_splits": len(splits),
        }


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Adjusts for multiple testing.
    """

    @staticmethod
    def compute(observed_sharpe: float, num_trials: int,
                sharpe_std: float = 1.0, skewness: float = 0.0,
                kurtosis: float = 3.0, T: int = 252) -> Dict:
        from scipy.stats import norm

        # Expected max Sharpe under null
        e_max = sharpe_std * ((1 - 0.5772) / (np.log(num_trials) + 1e-8) +
                              0.5772 / (np.sqrt(2 * np.log(num_trials)) + 1e-8))

        # Standard error of Sharpe ratio
        se = np.sqrt((1 + 0.5 * observed_sharpe**2 -
                       skewness * observed_sharpe +
                       (kurtosis - 3) / 4 * observed_sharpe**2) / T)

        # Deflated test statistic
        dsr_stat = (observed_sharpe - e_max) / (se + 1e-8)
        p_value = 1 - norm.cdf(dsr_stat)

        return {
            "dsr_statistic": float(dsr_stat),
            "p_value": float(p_value),
            "expected_max_sharpe": float(e_max),
            "significant": p_value < 0.05,
        }


class ProbabilityOfOverfitting:
    """Estimate probability of backtest overfitting (PBO)."""

    @staticmethod
    def compute(is_returns: List[np.ndarray], oos_returns: List[np.ndarray]) -> Dict:
        n = len(is_returns)
        is_sharpes = [np.mean(r) / (np.std(r) + 1e-8) for r in is_returns]
        oos_sharpes = [np.mean(r) / (np.std(r) + 1e-8) for r in oos_returns]

        best_is_idx = np.argmax(is_sharpes)
        best_is_oos = oos_sharpes[best_is_idx]

        # PBO: fraction of cases where best IS underperforms median OOS
        median_oos = np.median(oos_sharpes)
        pbo = float(best_is_oos < median_oos)

        return {
            "pbo": pbo,
            "best_is_sharpe": float(is_sharpes[best_is_idx]),
            "best_is_oos_sharpe": float(best_is_oos),
            "median_oos_sharpe": float(median_oos),
        }


def demo_overfitting():
    """Demonstrate overfitting detection and prevention."""
    print("=" * 70)
    print("Overfitting Prevention Demonstration")
    print("=" * 70)

    np.random.seed(42)
    T = 1000
    returns = np.random.randn(T) * 0.015 + 0.0002

    # Walk-forward validation
    print("\n--- Walk-Forward Validation ---")
    config = OverfittingConfig(train_window=252, test_window=63, gap=5)
    validator = WalkForwardValidator(config)

    def momentum_strategy(returns, idx):
        r = returns[idx]
        signal = np.sign(np.cumsum(r)[-1] if len(r) > 0 else 0)
        return r * signal

    result = validator.evaluate(returns, momentum_strategy)
    print(f"Splits: {result['num_splits']}")
    print(f"Mean IS Sharpe:  {result['mean_is']:.4f}")
    print(f"Mean OOS Sharpe: {result['mean_oos']:.4f}")
    print(f"Overfit ratio:   {result['overfit_ratio']:.4f}")

    # Deflated Sharpe Ratio
    print("\n--- Deflated Sharpe Ratio ---")
    for n_trials in [1, 10, 50, 100, 500]:
        try:
            dsr = DeflatedSharpeRatio.compute(
                observed_sharpe=1.5, num_trials=n_trials, T=252)
            print(f"  Trials={n_trials:>4}: DSR stat={dsr['dsr_statistic']:.3f}, "
                  f"p={dsr['p_value']:.4f}, significant={dsr['significant']}")
        except ImportError:
            print(f"  (scipy required for DSR computation)")
            break

    # PBO
    print("\n--- Probability of Backtest Overfitting ---")
    is_rets = [np.random.randn(63) * 0.015 + 0.001 * (i + 1) for i in range(10)]
    oos_rets = [np.random.randn(63) * 0.015 + 0.0001 for _ in range(10)]
    pbo = ProbabilityOfOverfitting.compute(is_rets, oos_rets)
    print(f"PBO: {pbo['pbo']}")
    print(f"Best IS Sharpe: {pbo['best_is_sharpe']:.4f}")
    print(f"Its OOS Sharpe: {pbo['best_is_oos_sharpe']:.4f}")


if __name__ == "__main__":
    demo_overfitting()
