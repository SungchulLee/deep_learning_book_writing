"""
Chapter 35.6.4: Statistical Significance
==========================================
Bootstrap, permutation tests, and multiple testing corrections.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


class BootstrapTest:
    """Bootstrap hypothesis test for strategy performance."""

    def __init__(self, num_bootstrap: int = 10000, block_size: int = 20):
        self.num_bootstrap = num_bootstrap
        self.block_size = block_size

    def test_sharpe(self, returns: np.ndarray, null_mean: float = 0.0) -> Dict:
        T = len(returns)
        observed_sharpe = (np.mean(returns) - null_mean) / (np.std(returns) + 1e-8) * np.sqrt(252)

        # Block bootstrap
        centered = returns - np.mean(returns) + null_mean
        n_blocks = T // self.block_size + 1

        boot_sharpes = []
        for _ in range(self.num_bootstrap):
            indices = np.random.randint(0, T - self.block_size, n_blocks)
            boot_sample = np.concatenate([centered[i:i+self.block_size] for i in indices])[:T]
            sr = np.mean(boot_sample) / (np.std(boot_sample) + 1e-8) * np.sqrt(252)
            boot_sharpes.append(sr)

        boot_sharpes = np.array(boot_sharpes)
        p_value = np.mean(boot_sharpes >= observed_sharpe)

        # Confidence interval
        ci_lower = np.percentile(boot_sharpes, 2.5)
        ci_upper = np.percentile(boot_sharpes, 97.5)

        return {
            "observed_sharpe": float(observed_sharpe),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "significant_5pct": p_value < 0.05,
        }


class PermutationTest:
    """Permutation test for timing skill."""

    def __init__(self, num_permutations: int = 10000):
        self.num_permutations = num_permutations

    def test_timing(self, positions: np.ndarray, returns: np.ndarray) -> Dict:
        observed = np.sum(positions * returns)

        perm_results = []
        for _ in range(self.num_permutations):
            shuffled_pos = np.random.permutation(positions)
            perm_results.append(np.sum(shuffled_pos * returns))
        perm_results = np.array(perm_results)

        p_value = np.mean(perm_results >= observed)

        return {
            "observed_pnl": float(observed),
            "mean_perm_pnl": float(np.mean(perm_results)),
            "p_value": float(p_value),
            "significant_5pct": p_value < 0.05,
        }


class MultipleTestingCorrection:
    """Corrections for multiple hypothesis testing."""

    @staticmethod
    def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
        adjusted = np.minimum(p_values * len(p_values), 1.0)
        return {
            "adjusted_p_values": adjusted.tolist(),
            "significant": (adjusted < alpha).tolist(),
            "n_significant": int(np.sum(adjusted < alpha)),
        }

    @staticmethod
    def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = min(sorted_p[i] * (n - i), 1.0)

        # Enforce monotonicity
        for i in range(1, n):
            adjusted[sorted_idx[i]] = max(adjusted[sorted_idx[i]], adjusted[sorted_idx[i-1]])

        return {
            "adjusted_p_values": adjusted.tolist(),
            "significant": (adjusted < alpha).tolist(),
            "n_significant": int(np.sum(adjusted < alpha)),
        }


class MinimumBacktestLength:
    """Compute minimum backtest length for significance."""

    @staticmethod
    def compute(target_sharpe: float, confidence: float = 0.95) -> Dict:
        from scipy.stats import norm
        z = norm.ppf(confidence)
        T_min_days = (z / target_sharpe) ** 2
        return {
            "min_days": int(np.ceil(T_min_days)),
            "min_years": float(T_min_days / 252),
            "target_sharpe": target_sharpe,
            "confidence": confidence,
        }


def demo_significance():
    """Demonstrate statistical significance tests."""
    print("=" * 70)
    print("Statistical Significance Tests")
    print("=" * 70)

    np.random.seed(42)
    T = 504
    returns = np.random.randn(T) * 0.015 + 0.0004

    # Bootstrap test
    print("\n--- Bootstrap Sharpe Test ---")
    boot = BootstrapTest(num_bootstrap=5000)
    result = boot.test_sharpe(returns)
    print(f"Observed Sharpe: {result['observed_sharpe']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"Significant: {result['significant_5pct']}")

    # Permutation test
    print("\n--- Permutation Test (Timing Skill) ---")
    positions = np.sign(np.random.randn(T))
    perm = PermutationTest(num_permutations=5000)
    result = perm.test_timing(positions, returns)
    print(f"Observed PnL: {result['observed_pnl']:.4f}")
    print(f"Mean perm PnL: {result['mean_perm_pnl']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")

    # Multiple testing
    print("\n--- Multiple Testing Corrections ---")
    p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50])
    bonf = MultipleTestingCorrection.bonferroni(p_values)
    holm = MultipleTestingCorrection.holm_bonferroni(p_values)
    print(f"{'Raw p':>8} {'Bonferroni':>12} {'Holm':>12}")
    print("-" * 34)
    for i in range(len(p_values)):
        print(f"{p_values[i]:>8.3f} {bonf['adjusted_p_values'][i]:>11.3f} "
              f"{holm['adjusted_p_values'][i]:>11.3f}")

    # Minimum backtest length
    print("\n--- Minimum Backtest Length ---")
    try:
        for sr in [0.25, 0.5, 1.0, 1.5, 2.0]:
            mbl = MinimumBacktestLength.compute(sr)
            print(f"  SR={sr:.2f}: {mbl['min_years']:.1f} years ({mbl['min_days']} days)")
    except ImportError:
        print("  (scipy needed)")


if __name__ == "__main__":
    demo_significance()
