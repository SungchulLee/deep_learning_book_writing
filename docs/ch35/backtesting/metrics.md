# Performance Metrics

Comprehensive performance metrics are the language through which trading strategy quality is communicated. Beyond simple returns, risk-adjusted measures like the Sharpe ratio, Sortino ratio, and Calmar ratio capture the trade-off between reward and risk. Tail risk metrics such as Value-at-Risk and Conditional Value-at-Risk quantify exposure to extreme losses, while trading metrics like win rate and profit factor reveal strategy behavior patterns.

## Code

```python
"""
Chapter 35.6.3: Performance Metrics
=====================================
Comprehensive performance metrics for trading strategies.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# ========================================================================
# Main
# ========================================================================


class PerformanceMetrics:
    """Compute comprehensive performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02 / 252):
        self.rf = risk_free_rate

    def compute_all(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        metrics = {}

        # Return metrics
        metrics["total_return"] = float(np.prod(1 + returns) - 1)
        metrics["cagr"] = float((1 + metrics["total_return"]) ** (252 / max(len(returns), 1)) - 1)
        metrics["daily_mean"] = float(np.mean(returns))

        # Risk metrics
        metrics["volatility"] = float(np.std(returns) * np.sqrt(252))
        metrics["downside_vol"] = float(np.std(returns[returns < 0]) * np.sqrt(252)) if np.any(returns < 0) else 0.0

        # Drawdown
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / (peak + 1e-8)
        metrics["max_drawdown"] = float(np.max(dd))

        # Find max drawdown duration
        underwater = dd > 0
        durations = []
        current = 0
        for u in underwater:
            if u:
                current += 1
            else:
                if current > 0:
                    durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)
        metrics["max_dd_duration"] = max(durations) if durations else 0

        # VaR and CVaR
        sorted_r = np.sort(returns)
        n5 = max(1, int(len(sorted_r) * 0.05))
        metrics["var_95"] = float(-sorted_r[n5])
        metrics["cvar_95"] = float(-np.mean(sorted_r[:n5]))

        # Risk-adjusted
        excess = returns - self.rf
        metrics["sharpe_ratio"] = float(np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252))

        downside = returns[returns < self.rf]
        ds_std = np.std(downside) * np.sqrt(252) if len(downside) > 1 else 1e-8
        metrics["sortino_ratio"] = float((metrics["cagr"] - 0.02) / (ds_std + 1e-8))

        metrics["calmar_ratio"] = float(metrics["cagr"] / (metrics["max_drawdown"] + 1e-8))

        # Trading metrics
        metrics["win_rate"] = float(np.mean(returns > 0))
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        metrics["profit_factor"] = float(np.sum(gains) / (np.abs(np.sum(losses)) + 1e-8))
        metrics["avg_win"] = float(np.mean(gains)) if len(gains) > 0 else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if len(losses) > 0 else 0.0

        # Tail ratios
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        metrics["tail_ratio"] = float(abs(p95) / (abs(p5) + 1e-8))

        # Benchmark comparison
        if benchmark_returns is not None:
            te = returns - benchmark_returns
            metrics["tracking_error"] = float(np.std(te) * np.sqrt(252))
            metrics["information_ratio"] = float(np.mean(te) / (np.std(te) + 1e-8) * np.sqrt(252))
            metrics["beta"] = float(np.cov(returns, benchmark_returns)[0, 1] / (np.var(benchmark_returns) + 1e-8))
            metrics["alpha"] = float((metrics["cagr"] - 0.02) - metrics["beta"] * (np.mean(benchmark_returns) * 252 - 0.02))

        return metrics

    def format_report(self, metrics: Dict) -> str:
        lines = ["=" * 50, "Performance Report", "=" * 50]
        sections = {
            "Returns": ["total_return", "cagr", "daily_mean"],
            "Risk": ["volatility", "max_drawdown", "max_dd_duration", "var_95", "cvar_95"],
            "Risk-Adjusted": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
            "Trading": ["win_rate", "profit_factor", "avg_win", "avg_loss", "tail_ratio"],
        }
        for section, keys in sections.items():
            lines.append(f"\n--- {section} ---")
            for k in keys:
                if k in metrics:
                    v = metrics[k]
                    if "return" in k or "cagr" in k or "rate" in k or "alpha" in k:
                        lines.append(f"  {k:<22}: {v*100:>10.2f}%")
                    elif "ratio" in k or "factor" in k or "beta" in k:
                        lines.append(f"  {k:<22}: {v:>10.4f}")
                    elif "duration" in k:
                        lines.append(f"  {k:<22}: {v:>10.0f} days")
                    else:
                        lines.append(f"  {k:<22}: {v:>10.6f}")
        return "\n".join(lines)


def demo_metrics():
    """Demonstrate performance metrics."""
    print("=" * 70)
    print("Performance Metrics Demonstration")
    print("=" * 70)

    np.random.seed(42)
    T = 504  # 2 years
    strategy_returns = np.random.randn(T) * 0.012 + 0.0004
    strategy_returns[100:120] -= 0.02  # Drawdown

    benchmark_returns = np.random.randn(T) * 0.01 + 0.0003

    pm = PerformanceMetrics()
    metrics = pm.compute_all(strategy_returns, benchmark_returns)
    print(pm.format_report(metrics))


if __name__ == "__main__":
    demo_metrics()
```

## Discussion

The Sharpe ratio, defined as the annualized excess return divided by annualized volatility, remains the most widely cited risk-adjusted performance measure. However, it treats upside and downside volatility symmetrically, which penalizes strategies with positively skewed returns. The Sortino ratio addresses this by using only downside deviation in the denominator, making it more appropriate for strategies that occasionally produce large gains.

Drawdown analysis reveals a strategy's worst-case behavior. Maximum drawdown measures the largest peak-to-trough decline, while maximum drawdown duration captures how long a strategy stayed underwater. The Calmar ratio (annualized return divided by maximum drawdown) provides a single number summarizing return relative to worst-case loss. These metrics are particularly important for live trading, where an investor must psychologically withstand drawdowns.

When a benchmark is available, relative performance metrics become essential. Tracking error measures the volatility of return differences between strategy and benchmark. The information ratio (excess return over tracking error) quantifies whether active bets are rewarded. Alpha and beta from the CAPM framework decompose returns into market-driven and skill-driven components, helping determine whether a strategy truly generates alpha or merely takes on systematic risk.

## Exercises

**Exercise 1.**
Compute the Sharpe ratio, Sortino ratio, and maximum drawdown for a return series with daily mean return 0.05%, daily standard deviation 1.5%, and a 20-day period where daily returns averaged -0.8%.

??? success "Solution to Exercise 1"
    Annualized mean excess return $= (0.0005 - 0.02/252) \times 252 \approx 0.106$ (10.6%).

    Annualized volatility $= 0.015 \times \sqrt{252} \approx 0.238$ (23.8%).
    
    Sharpe ratio $= 0.106 / 0.238 \approx 0.445$.
    
    For the Sortino ratio, downside deviation uses only returns below the risk-free rate. Assuming roughly half of days are negative, annualized downside deviation $\approx 0.015 \times \sqrt{252/2} \approx 0.168$. Sortino $\approx 0.106 / 0.168 \approx 0.631$.
    
    During the drawdown period: cumulative loss $\approx 20 \times (-0.008) = -0.16$ or 16%. If this was from a peak, maximum drawdown $\approx 16\%$.

---


**Exercise 2.**
Explain why the profit factor (sum of gains divided by sum of losses) can be misleading for strategies with very different trade frequencies. Propose a normalized alternative.

??? success "Solution to Exercise 2"
    A strategy that trades once per year with a single large gain and a single small loss can have a very high profit factor without demonstrating consistent skill. Conversely, a high-frequency strategy with thousands of small gains and slightly fewer small losses may have a moderate profit factor despite being highly reliable.

    A better alternative is the per-trade profit factor or the gain-to-pain ratio: $\text{GtP} = \sum r_i / \sum |r_i^-|$ where $r_i^-$ are negative returns. Another approach is to compute profit factor over rolling windows and report its stability (standard deviation of rolling profit factors), capturing both magnitude and consistency.

---


**Exercise 3.**
Implement a function that computes the tail ratio (ratio of the 95th percentile gain to the absolute value of the 5th percentile loss) and explain its interpretation for strategy evaluation.

??? success "Solution to Exercise 3"
    ```python
    def tail_ratio(returns):
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95) / (abs(p5) + 1e-8)
    ```
    
    A tail ratio greater than 1 indicates that the right tail (large gains) is fatter than the left tail (large losses), which is desirable. A tail ratio less than 1 means the strategy has fatter left tails -- large losses are more extreme than large gains. For trend-following strategies, the tail ratio is typically above 1 (many small losses, few large gains). For mean-reversion strategies, it is often below 1 (many small gains, occasional large losses).

