"""
Chapter 35.4.1: Risk-Adjusted Rewards
=======================================
Comprehensive risk-adjusted reward functions for financial RL including
differential Sharpe, Sortino, Calmar, and CVaR-based rewards.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================
# Configuration
# ============================================================

@dataclass
class RiskRewardConfig:
    """Configuration for risk-adjusted rewards."""
    risk_free_rate: float = 0.02 / 252  # Daily
    risk_aversion: float = 0.5
    sharpe_window: int = 20
    ema_decay: float = 0.05    # For differential Sharpe
    drawdown_penalty: float = 1.0
    downside_penalty: float = 1.0


# ============================================================
# Base Reward
# ============================================================

class BaseReward:
    """Base class for reward functions."""

    def __init__(self, config: RiskRewardConfig):
        self.config = config
        self.return_history: List[float] = []

    def reset(self):
        self.return_history = []

    def compute(self, portfolio_return: float, **kwargs) -> float:
        raise NotImplementedError


# ============================================================
# Raw Return Reward
# ============================================================

class RawReturnReward(BaseReward):
    """Simple log return reward."""

    def compute(self, portfolio_return: float, **kwargs) -> float:
        self.return_history.append(portfolio_return)
        return portfolio_return


# ============================================================
# Differential Sharpe Ratio
# ============================================================

class DifferentialSharpeReward(BaseReward):
    """
    Differential Sharpe ratio (Moody & Saffell, 2001).
    Single-step reward that approximates the Sharpe ratio gradient.

    D_t = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}
    """

    def __init__(self, config: RiskRewardConfig):
        super().__init__(config)
        self.A = 0.0   # EMA of returns
        self.B = 0.0   # EMA of squared returns
        self.eta = config.ema_decay
        self.initialized = False

    def reset(self):
        super().reset()
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def compute(self, portfolio_return: float, **kwargs) -> float:
        self.return_history.append(portfolio_return)
        r = portfolio_return

        if not self.initialized:
            self.A = r
            self.B = r ** 2
            self.initialized = True
            return r

        # Deltas
        delta_A = r - self.A
        delta_B = r ** 2 - self.B

        # Denominator
        denom = (self.B - self.A ** 2) ** 1.5

        if abs(denom) < 1e-10:
            reward = r
        else:
            reward = (self.B * delta_A - 0.5 * self.A * delta_B) / denom

        # Update EMAs
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return float(reward)


# ============================================================
# Rolling Sharpe Reward
# ============================================================

class RollingSharpeReward(BaseReward):
    """Reward based on rolling window Sharpe ratio."""

    def compute(self, portfolio_return: float, **kwargs) -> float:
        self.return_history.append(portfolio_return)
        window = self.config.sharpe_window

        if len(self.return_history) < window:
            return portfolio_return

        recent = np.array(self.return_history[-window:])
        mean_ret = np.mean(recent) - self.config.risk_free_rate
        std_ret = np.std(recent) + 1e-8

        sharpe = mean_ret / std_ret
        return float(sharpe * 0.1)  # Scale for RL


# ============================================================
# Sortino Reward
# ============================================================

class SortinoReward(BaseReward):
    """Reward penalizing only downside deviation."""

    def compute(self, portfolio_return: float, **kwargs) -> float:
        self.return_history.append(portfolio_return)
        r = portfolio_return

        # Downside penalty
        downside = max(0, -r) ** 2
        reward = r - self.config.downside_penalty * downside

        return float(reward)


# ============================================================
# Calmar (Drawdown) Reward
# ============================================================

class CalmarReward(BaseReward):
    """Reward penalizing drawdown from peak."""

    def __init__(self, config: RiskRewardConfig):
        super().__init__(config)
        self.portfolio_value = 1.0
        self.peak_value = 1.0

    def reset(self):
        super().reset()
        self.portfolio_value = 1.0
        self.peak_value = 1.0

    def compute(self, portfolio_return: float, **kwargs) -> float:
        self.return_history.append(portfolio_return)

        self.portfolio_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        reward = portfolio_return - self.config.drawdown_penalty * drawdown

        return float(reward)


# ============================================================
# CVaR-Adjusted Reward
# ============================================================

class CVaRReward(BaseReward):
    """Reward incorporating Conditional Value-at-Risk."""

    def __init__(self, config: RiskRewardConfig, alpha: float = 0.05):
        super().__init__(config)
        self.alpha = alpha

    def compute(self, portfolio_return: float, **kwargs) -> float:
        self.return_history.append(portfolio_return)
        window = self.config.sharpe_window

        if len(self.return_history) < window:
            return portfolio_return

        recent = np.array(self.return_history[-window:])

        # Compute CVaR
        sorted_returns = np.sort(recent)
        n_tail = max(1, int(len(sorted_returns) * self.alpha))
        cvar = np.mean(sorted_returns[:n_tail])

        reward = portfolio_return + self.config.risk_aversion * cvar
        return float(reward)


# ============================================================
# Risk Parity Reward
# ============================================================

class RiskParityReward(BaseReward):
    """Reward encouraging equal risk contribution across assets."""

    def compute(self, portfolio_return: float, weights: np.ndarray = None,
                asset_returns: np.ndarray = None, **kwargs) -> float:
        self.return_history.append(portfolio_return)

        if weights is None or asset_returns is None:
            return portfolio_return

        # Marginal risk contributions
        vol = np.std(asset_returns, axis=0) + 1e-8
        risk_contrib = np.abs(weights) * vol
        total_risk = np.sum(risk_contrib) + 1e-8
        pct_contrib = risk_contrib / total_risk

        # Penalty for unequal risk contribution
        N = len(weights)
        target = 1.0 / N
        rc_penalty = np.sum((pct_contrib - target) ** 2)

        reward = portfolio_return - self.config.risk_aversion * rc_penalty
        return float(reward)


# ============================================================
# Composite Reward
# ============================================================

class CompositeReward:
    """Combine multiple reward components with configurable weights."""

    def __init__(self, components: Dict[str, BaseReward], weights: Dict[str, float]):
        self.components = components
        self.weights = weights

    def reset(self):
        for comp in self.components.values():
            comp.reset()

    def compute(self, portfolio_return: float, **kwargs) -> Dict[str, float]:
        rewards = {}
        total = 0.0
        for name, comp in self.components.items():
            r = comp.compute(portfolio_return, **kwargs)
            w = self.weights.get(name, 1.0)
            rewards[name] = r
            total += w * r

        rewards["total"] = total
        return rewards


# ============================================================
# Demonstration
# ============================================================

def demo_risk_adjusted_rewards():
    """Demonstrate risk-adjusted reward functions."""
    print("=" * 70)
    print("Risk-Adjusted Rewards Demonstration")
    print("=" * 70)

    config = RiskRewardConfig(
        risk_free_rate=0.02 / 252,
        risk_aversion=0.5,
        sharpe_window=20,
        ema_decay=0.05,
    )

    # Generate synthetic return series
    np.random.seed(42)
    T = 200
    returns = np.random.randn(T) * 0.01 + 0.0003  # Slight positive drift

    # Add a drawdown period
    returns[80:100] = np.random.randn(20) * 0.02 - 0.01

    rewards_funcs = {
        "Raw Return": RawReturnReward(config),
        "Diff Sharpe": DifferentialSharpeReward(config),
        "Rolling Sharpe": RollingSharpeReward(config),
        "Sortino": SortinoReward(config),
        "Calmar": CalmarReward(config),
        "CVaR": CVaRReward(config, alpha=0.05),
    }

    results = {name: [] for name in rewards_funcs}

    for name, func in rewards_funcs.items():
        func.reset()
        for t in range(T):
            r = func.compute(returns[t])
            results[name].append(r)

    print(f"\n{'Reward Function':<18} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 62)
    for name, vals in results.items():
        v = np.array(vals)
        print(f"{name:<18} {np.mean(v):>10.6f} {np.std(v):>10.6f} "
              f"{np.min(v):>10.6f} {np.max(v):>10.6f}")

    # Compare drawdown period behavior
    print("\n--- Drawdown Period (steps 80-100) ---")
    for name, vals in results.items():
        dd_vals = np.array(vals[80:100])
        print(f"  {name:<18}: mean={np.mean(dd_vals):>10.6f}, "
              f"sum={np.sum(dd_vals):>10.6f}")

    # Composite reward
    print("\n--- Composite Reward ---")
    composite = CompositeReward(
        components={
            "return": RawReturnReward(config),
            "sharpe": DifferentialSharpeReward(config),
            "drawdown": CalmarReward(config),
        },
        weights={"return": 0.5, "sharpe": 0.3, "drawdown": 0.2},
    )
    composite.reset()

    comp_rewards = []
    for t in range(T):
        r = composite.compute(returns[t])
        comp_rewards.append(r["total"])

    v = np.array(comp_rewards)
    print(f"Composite: mean={np.mean(v):.6f}, std={np.std(v):.6f}")

    # Risk aversion sensitivity
    print("\n--- Risk Aversion Sensitivity ---")
    print(f"{'Lambda':>8} {'Cumulative Reward':>18}")
    print("-" * 28)
    for lam in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        cfg = RiskRewardConfig(risk_aversion=lam, drawdown_penalty=lam)
        calmar = CalmarReward(cfg)
        calmar.reset()
        total = sum(calmar.compute(r) for r in returns)
        print(f"{lam:>8.1f} {total:>17.4f}")


if __name__ == "__main__":
    demo_risk_adjusted_rewards()
