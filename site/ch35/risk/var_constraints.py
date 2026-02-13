"""
Chapter 35.4.3: VaR Constraints
=================================
Value-at-Risk estimation and constrained RL optimization
with parametric, historical, and Monte Carlo VaR.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VaRConfig:
    confidence_level: float = 0.95
    var_limit: float = 0.02       # 2% portfolio value
    lookback: int = 60
    num_mc_simulations: int = 1000
    lambda_var: float = 10.0      # Lagrangian multiplier init
    lambda_lr: float = 0.01       # Dual variable learning rate


# ============================================================
# VaR Estimators
# ============================================================

class ParametricVaR:
    """Gaussian VaR using variance-covariance method."""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        from scipy.stats import norm
        self.z = norm.ppf(1 - confidence)  # Negative

    def compute(self, weights: np.ndarray, returns: np.ndarray,
                portfolio_value: float = 1.0) -> float:
        mu = returns.mean(axis=0)
        cov = np.cov(returns.T)
        port_mu = weights @ mu
        port_var = weights @ cov @ weights
        port_std = np.sqrt(port_var + 1e-8)
        var = -(port_mu + self.z * port_std) * portfolio_value
        return max(0, float(var))


class HistoricalVaR:
    """Historical simulation VaR."""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def compute(self, weights: np.ndarray, returns: np.ndarray,
                portfolio_value: float = 1.0) -> float:
        port_returns = returns @ weights
        alpha = 1 - self.confidence
        var = -np.quantile(port_returns, alpha) * portfolio_value
        return max(0, float(var))


class MonteCarloVaR:
    """Monte Carlo VaR simulation."""

    def __init__(self, confidence: float = 0.95, num_simulations: int = 1000):
        self.confidence = confidence
        self.num_simulations = num_simulations

    def compute(self, weights: np.ndarray, returns: np.ndarray,
                portfolio_value: float = 1.0,
                seed: Optional[int] = None) -> float:
        rng = np.random.RandomState(seed)
        mu = returns.mean(axis=0)
        cov = np.cov(returns.T) + np.eye(len(mu)) * 1e-8

        simulated = rng.multivariate_normal(mu, cov, self.num_simulations)
        port_returns = simulated @ weights

        alpha = 1 - self.confidence
        var = -np.quantile(port_returns, alpha) * portfolio_value
        return max(0, float(var))


# ============================================================
# VaR Constraint Handler
# ============================================================

class VaRConstraintHandler:
    """Manages VaR constraints for RL policies."""

    def __init__(self, config: VaRConfig, var_estimator=None):
        self.config = config
        self.estimator = var_estimator or HistoricalVaR(config.confidence_level)
        self.lambda_var = config.lambda_var
        self.violation_history: List[bool] = []

    def check_constraint(self, weights: np.ndarray, returns: np.ndarray,
                         portfolio_value: float = 1.0) -> Dict:
        var = self.estimator.compute(weights, returns, portfolio_value)
        var_pct = var / (portfolio_value + 1e-8)
        violated = var_pct > self.config.var_limit

        self.violation_history.append(violated)

        return {
            "var": var,
            "var_pct": var_pct,
            "var_limit": self.config.var_limit,
            "violated": violated,
            "excess": max(0, var_pct - self.config.var_limit),
        }

    def compute_penalty(self, var_info: Dict) -> float:
        """Compute reward penalty for VaR violation."""
        if not var_info["violated"]:
            return 0.0
        return self.lambda_var * var_info["excess"]

    def update_lambda(self, var_info: Dict):
        """Dual variable update for Lagrangian relaxation."""
        gradient = var_info["var_pct"] - self.config.var_limit
        self.lambda_var = max(0, self.lambda_var + self.config.lambda_lr * gradient)

    def scale_action(self, weights: np.ndarray, returns: np.ndarray,
                     portfolio_value: float = 1.0) -> np.ndarray:
        """Scale weights to satisfy VaR constraint."""
        var_info = self.check_constraint(weights, returns, portfolio_value)
        if not var_info["violated"]:
            return weights

        # Binary search for maximum feasible scale
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2
            scaled = weights * mid
            v = self.estimator.compute(scaled, returns, portfolio_value)
            if v / (portfolio_value + 1e-8) <= self.config.var_limit:
                lo = mid
            else:
                hi = mid
        return weights * lo


# ============================================================
# Demonstration
# ============================================================

def demo_var_constraints():
    """Demonstrate VaR constraint mechanisms."""
    print("=" * 70)
    print("VaR Constraints Demonstration")
    print("=" * 70)

    np.random.seed(42)
    N = 5
    T = 252
    returns = np.random.randn(T, N) * 0.015 + 0.0003
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    pv = 1_000_000.0

    config = VaRConfig(confidence_level=0.95, var_limit=0.02)

    # Compare estimators
    print("\n--- VaR Estimation Methods ---")
    estimators = {
        "Parametric": ParametricVaR(0.95),
        "Historical": HistoricalVaR(0.95),
        "Monte Carlo": MonteCarloVaR(0.95, num_simulations=10000),
    }
    for name, est in estimators.items():
        if name == "Monte Carlo":
            var = est.compute(weights, returns, pv, seed=42)
        else:
            var = est.compute(weights, returns, pv)
        print(f"  {name:<15}: VaR = ${var:>12,.0f} ({var/pv*100:.2f}%)")

    # Constraint checking
    print("\n--- VaR Constraint Checking ---")
    handler = VaRConstraintHandler(config)
    info = handler.check_constraint(weights, returns, pv)
    print(f"VaR: {info['var_pct']*100:.2f}%, Limit: {info['var_limit']*100:.2f}%")
    print(f"Violated: {info['violated']}, Excess: {info['excess']*100:.2f}%")

    # Action scaling
    print("\n--- Action Scaling ---")
    aggressive_w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    scaled_w = handler.scale_action(aggressive_w, returns, pv)
    var_before = handler.estimator.compute(aggressive_w, returns, pv)
    var_after = handler.estimator.compute(scaled_w, returns, pv)
    print(f"Before: weights={aggressive_w}, VaR=${var_before:,.0f}")
    print(f"After:  weights={np.round(scaled_w, 3)}, VaR=${var_after:,.0f}")

    # Lagrangian dual update
    print("\n--- Lagrangian Dual Update ---")
    print(f"{'Step':>5} {'Lambda':>10} {'VaR%':>8} {'Penalty':>10}")
    print("-" * 36)
    handler.lambda_var = 10.0
    for step in range(10):
        w = weights * (1 + 0.1 * np.random.randn(N))
        w = np.abs(w) / np.sum(np.abs(w))
        info = handler.check_constraint(w, returns, pv)
        penalty = handler.compute_penalty(info)
        handler.update_lambda(info)
        if step % 2 == 0:
            print(f"{step:>5} {handler.lambda_var:>9.3f} "
                  f"{info['var_pct']*100:>7.2f} {penalty:>9.4f}")


if __name__ == "__main__":
    demo_var_constraints()
