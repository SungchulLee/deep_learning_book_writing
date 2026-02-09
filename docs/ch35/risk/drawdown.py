"""
Chapter 35.4.2: Drawdown Control
==================================
Drawdown-aware RL with position scaling, circuit breakers,
and constrained policy optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DrawdownConfig:
    max_drawdown: float = 0.10
    warning_threshold: float = 0.05
    circuit_breaker: float = 0.15
    recovery_threshold: float = 0.03
    drawdown_penalty: float = 2.0
    position_scaling: bool = True


class DrawdownTracker:
    """Track drawdown metrics in real-time."""

    def __init__(self):
        self.peak_value = 1.0
        self.current_value = 1.0
        self.max_drawdown = 0.0
        self.current_dd_duration = 0
        self.max_dd_duration = 0
        self.step = 0
        self.history: List[float] = []

    def reset(self, initial_value: float = 1.0):
        self.peak_value = initial_value
        self.current_value = initial_value
        self.max_drawdown = 0.0
        self.current_dd_duration = 0
        self.max_dd_duration = 0
        self.step = 0
        self.history = []

    def update(self, portfolio_value: float) -> Dict[str, float]:
        self.current_value = portfolio_value
        self.step += 1

        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_dd_duration = 0
        else:
            self.current_dd_duration += 1

        drawdown = (self.peak_value - portfolio_value) / (self.peak_value + 1e-8)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.max_dd_duration = max(self.max_dd_duration, self.current_dd_duration)
        self.history.append(drawdown)

        return {
            "drawdown": float(drawdown),
            "max_drawdown": float(self.max_drawdown),
            "dd_duration": self.current_dd_duration,
            "max_dd_duration": self.max_dd_duration,
            "recovery_ratio": float(portfolio_value / (self.peak_value + 1e-8)),
        }


class DrawdownPositionScaler:
    """Scale positions based on current drawdown level."""

    def __init__(self, config: DrawdownConfig):
        self.config = config

    def compute_scale(self, drawdown: float) -> float:
        if drawdown <= self.config.warning_threshold:
            return 1.0
        elif drawdown >= self.config.max_drawdown:
            return 0.0
        else:
            range_ = self.config.max_drawdown - self.config.warning_threshold
            excess = drawdown - self.config.warning_threshold
            return max(0.0, 1.0 - excess / (range_ + 1e-8))

    def scale_weights(self, weights: np.ndarray, drawdown: float) -> np.ndarray:
        return weights * self.compute_scale(drawdown)


class CircuitBreaker:
    """Hard stop when drawdown exceeds limit."""

    def __init__(self, config: DrawdownConfig):
        self.config = config
        self.triggered = False

    def reset(self):
        self.triggered = False

    def check(self, drawdown: float) -> bool:
        if drawdown >= self.config.circuit_breaker:
            self.triggered = True
        if self.triggered and drawdown <= self.config.recovery_threshold:
            self.triggered = False
        return self.triggered


class DrawdownConstrainedPolicy(nn.Module):
    """Policy network with drawdown state augmentation."""

    def __init__(self, base_state_dim: int, num_assets: int, hidden_dim: int = 128):
        super().__init__()
        # Augmented state: base + (drawdown, dd_duration, recovery_ratio)
        augmented_dim = base_state_dim + 3

        self.encoder = nn.Sequential(
            nn.Linear(augmented_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_assets)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, dd_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        augmented = torch.cat([state, dd_state], dim=-1)
        features = self.encoder(augmented)
        weights = F.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features).squeeze(-1)
        return {"weights": weights, "value": value}


class DrawdownRewardWrapper:
    """Wrapper that adds drawdown penalty to base reward."""

    def __init__(self, config: DrawdownConfig):
        self.config = config
        self.tracker = DrawdownTracker()
        self.scaler = DrawdownPositionScaler(config)
        self.breaker = CircuitBreaker(config)

    def reset(self, initial_value: float = 1.0):
        self.tracker.reset(initial_value)
        self.breaker.reset()

    def compute(self, base_reward: float, portfolio_value: float) -> Tuple[float, Dict]:
        dd_info = self.tracker.update(portfolio_value)
        dd = dd_info["drawdown"]

        # Quadratic penalty beyond threshold
        penalty = 0.0
        if dd > self.config.warning_threshold:
            excess = dd - self.config.warning_threshold
            penalty = self.config.drawdown_penalty * excess ** 2

        adjusted = base_reward - penalty
        circuit = self.breaker.check(dd)

        info = {
            **dd_info,
            "penalty": penalty,
            "circuit_breaker": circuit,
            "position_scale": self.scaler.compute_scale(dd),
        }
        return float(adjusted), info


def demo_drawdown_control():
    """Demonstrate drawdown control mechanisms."""
    print("=" * 70)
    print("Drawdown Control Demonstration")
    print("=" * 70)

    config = DrawdownConfig(
        max_drawdown=0.10, warning_threshold=0.05,
        circuit_breaker=0.15, drawdown_penalty=2.0,
    )

    # Simulate portfolio with a drawdown event
    np.random.seed(42)
    T = 200
    returns = np.random.randn(T) * 0.01 + 0.0003
    returns[70:90] = np.random.randn(20) * 0.015 - 0.008  # Drawdown
    returns[140:155] = np.random.randn(15) * 0.02 - 0.012  # Severe drawdown

    wrapper = DrawdownRewardWrapper(config)
    wrapper.reset(1.0)

    portfolio_value = 1.0
    print(f"\n{'Step':>5} {'Value':>10} {'DD%':>8} {'Scale':>8} {'CB':>4} {'Penalty':>10}")
    print("-" * 50)

    for t in range(T):
        portfolio_value *= (1 + returns[t])
        adj_reward, info = wrapper.compute(returns[t], portfolio_value)

        if t % 20 == 0 or info["circuit_breaker"] or info["drawdown"] > 0.05:
            print(f"{t:>5} {portfolio_value:>9.4f} "
                  f"{info['drawdown']*100:>7.2f}% "
                  f"{info['position_scale']:>7.3f} "
                  f"{'Y' if info['circuit_breaker'] else 'N':>3} "
                  f"{info['penalty']:>9.6f}")

    print(f"\nMax drawdown: {wrapper.tracker.max_drawdown*100:.2f}%")
    print(f"Max DD duration: {wrapper.tracker.max_dd_duration} steps")

    # Position scaling demo
    print("\n--- Position Scaling ---")
    scaler = DrawdownPositionScaler(config)
    for dd in [0.0, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        scale = scaler.compute_scale(dd)
        print(f"  DD={dd*100:5.1f}% -> scale={scale:.3f}")

    # Policy network
    print("\n--- Drawdown-Constrained Policy ---")
    policy = DrawdownConstrainedPolicy(base_state_dim=20, num_assets=5)
    params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {params:,}")

    state = torch.randn(1, 20)
    dd_state = torch.FloatTensor([[0.03, 5.0, 0.97]])
    with torch.no_grad():
        out = policy(state, dd_state)
    print(f"Weights: {out['weights'][0].numpy()}")
    print(f"Value: {out['value'].item():.4f}")


if __name__ == "__main__":
    demo_drawdown_control()
