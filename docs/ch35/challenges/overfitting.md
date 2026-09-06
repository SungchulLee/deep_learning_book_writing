# Overfitting Prevention

Overfitting is the most dangerous pitfall in quantitative trading: a strategy that performs brilliantly on historical data but fails in live markets. Walk-forward validation, the Deflated Sharpe Ratio, and probability of backtest overfitting provide rigorous tools to detect and prevent this failure mode before real capital is at risk.

## Code

```python
"""
Chapter 35.5.4: Overfitting Prevention
========================================
Walk-forward validation, regularization, and statistical tests
for overfitting prevention in financial RL.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ========================================================================
# Main
# ========================================================================


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
```

## Discussion

Walk-forward validation for time series provides an honest assessment of out-of-sample performance by ensuring all training data precedes test data. The overfit ratio -- the ratio of in-sample to out-of-sample Sharpe -- directly measures the degree to which a strategy exploits noise rather than signal. A ratio close to 1.0 indicates robust generalization, while ratios above 2-3 indicate severe overfitting.

The Deflated Sharpe Ratio (DSR) adjusts the observed Sharpe ratio for the number of strategies tried during development. If a researcher tests 100 strategy variants, the best one will appear profitable by chance even with no real edge. The DSR computes the expected maximum Sharpe under the null hypothesis and tests whether the observed Sharpe significantly exceeds it. With 500 trials, a Sharpe of 1.5 over one year is often not statistically significant.

The Probability of Backtest Overfitting (PBO) estimates the likelihood that the best in-sample strategy underperforms in the out-of-sample period. It uses combinatorial cross-validation to generate many train-test splits and checks whether the strategy that optimizes in-sample performance consistently performs well out-of-sample. A PBO close to 1.0 indicates that in-sample optimization is essentially random.

## Exercises

**Exercise 1.**
A researcher tests 50 strategy variants and selects the one with a Sharpe ratio of 2.0 over 252 trading days. Using the Deflated Sharpe Ratio framework, assess whether this result is statistically significant.

??? success "Solution to Exercise 1"
    The expected maximum Sharpe under the null (no skill) with 50 independent trials is approximately:

    $E[\max SR] \approx \sigma_{SR} \left(\frac{1 - 0.5772}{\log(50)} + \frac{0.5772}{\sqrt{2\log(50)}}\right)$
    
    With $\sigma_{SR} \approx 1$ and $\log(50) \approx 3.91$: $E[\max SR] \approx 0.108 + 0.206 \approx 0.99$.
    
    The standard error of the Sharpe ratio over 252 days is $SE \approx \sqrt{(1 + 0.5 \cdot 2^2)/252} \approx 0.109$.
    
    DSR statistic: $(2.0 - 0.99) / 0.109 \approx 9.27$, which is highly significant (p < 0.001). However, if the 50 variants are correlated (similar strategies), the effective number of trials is lower, making the result even more significant.

---


**Exercise 2.**
Explain the difference between in-sample overfitting and out-of-sample degradation. Can a strategy show low degradation but still be overfit?

??? success "Solution to Exercise 2"
    In-sample overfitting occurs when a strategy captures noise patterns specific to the training data. Out-of-sample degradation is the observed drop in performance when moving from training to test data. While high degradation implies overfitting, low degradation does not guarantee robustness.

    A strategy can show low degradation and still be overfit if: (1) The test period happens to be similar to the training period (lucky split). (2) The strategy was selected from many variants based on its walk-forward results, introducing a second layer of selection bias. (3) The test period is too short to distinguish skill from luck. This is why multiple walk-forward splits and statistical significance tests are necessary -- a single train-test split is insufficient.

---


**Exercise 3.**
Implement a simple PBO estimator that uses 10 random train-test splits of historical returns and reports the probability that the best in-sample strategy underperforms the median out-of-sample.

??? success "Solution to Exercise 3"
    ```python
    def estimate_pbo(returns, n_strategies=10, n_splits=16):
        T = len(returns)
        half = T // 2
        underperform_count = 0
        
        for _ in range(n_splits):
            # Random partition into two halves
            idx = np.random.permutation(T)
            is_idx = idx[:half]
            oos_idx = idx[half:]
            
            # Generate strategy returns (e.g., different lookbacks)
            is_sharpes = []
            oos_sharpes = []
            for lookback in np.linspace(5, 60, n_strategies).astype(int):
                signal_is = np.sign(np.convolve(returns[is_idx], np.ones(lookback)/lookback, 'same'))
                signal_oos = np.sign(np.convolve(returns[oos_idx], np.ones(lookback)/lookback, 'same'))
                is_sharpes.append(np.mean(signal_is * returns[is_idx]))
                oos_sharpes.append(np.mean(signal_oos * returns[oos_idx]))
            
            best_is = np.argmax(is_sharpes)
            if oos_sharpes[best_is] < np.median(oos_sharpes):
                underperform_count += 1
        
        return underperform_count / n_splits
    ```
    A PBO above 0.5 indicates that selecting the best in-sample strategy is no better than random selection out-of-sample, strongly suggesting overfitting.

