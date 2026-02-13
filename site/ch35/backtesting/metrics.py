"""
Chapter 35.6.3: Performance Metrics
=====================================
Comprehensive performance metrics for trading strategies.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


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
