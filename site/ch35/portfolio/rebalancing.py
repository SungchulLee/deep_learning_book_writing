"""
Chapter 35.2.3: Rebalancing Strategies
=======================================
RL-based dynamic rebalancing including threshold, band-based,
and learned rebalancing policies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass
class RebalancingConfig:
    """Configuration for rebalancing strategies."""
    num_assets: int = 5
    transaction_cost: float = 0.001
    tracking_error_penalty: float = 0.5
    max_turnover: float = 0.5
    calendar_period: int = 20       # Trading days
    threshold: float = 0.05         # 5% drift threshold
    band_width: float = 0.03        # 3% no-trade band
    initial_capital: float = 1_000_000.0


# ============================================================
# Portfolio Drift Simulator
# ============================================================

class PortfolioDriftSimulator:
    """Simulates portfolio weight drift due to market movements."""

    def __init__(self, num_assets: int):
        self.num_assets = num_assets

    def compute_drift(
        self, weights: np.ndarray, returns: np.ndarray
    ) -> np.ndarray:
        """
        Compute drifted weights after market returns.

        Args:
            weights: (N,) current weights
            returns: (N,) single-period returns
        Returns:
            drifted weights: (N,)
        """
        new_values = weights * (1.0 + returns)
        total = np.sum(new_values)
        if total <= 0:
            return np.ones(self.num_assets) / self.num_assets
        return new_values / total

    def simulate_drift_path(
        self,
        initial_weights: np.ndarray,
        returns_series: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate weight drift over multiple periods without rebalancing.

        Args:
            initial_weights: (N,) starting weights
            returns_series: (T, N) returns
        Returns:
            drift_path: (T+1, N) weight path
        """
        T = len(returns_series)
        path = np.zeros((T + 1, self.num_assets))
        path[0] = initial_weights

        for t in range(T):
            path[t + 1] = self.compute_drift(path[t], returns_series[t])

        return path


# ============================================================
# Calendar Rebalancing
# ============================================================

class CalendarRebalancer:
    """Fixed-schedule rebalancing strategy."""

    def __init__(self, config: RebalancingConfig, target_weights: np.ndarray):
        self.config = config
        self.target_weights = target_weights
        self.period = config.calendar_period
        self.step_count = 0

    def reset(self):
        self.step_count = 0

    def should_rebalance(self) -> bool:
        return self.step_count % self.period == 0

    def get_trades(self, current_weights: np.ndarray) -> Tuple[np.ndarray, float]:
        self.step_count += 1
        if self.should_rebalance():
            turnover = float(np.sum(np.abs(self.target_weights - current_weights)))
            return self.target_weights.copy(), turnover
        return current_weights.copy(), 0.0


# ============================================================
# Threshold Rebalancing
# ============================================================

class ThresholdRebalancer:
    """Rebalance when total drift exceeds threshold."""

    def __init__(self, config: RebalancingConfig, target_weights: np.ndarray):
        self.config = config
        self.target_weights = target_weights
        self.threshold = config.threshold

    def should_rebalance(self, current_weights: np.ndarray) -> bool:
        drift = np.sum(np.abs(current_weights - self.target_weights))
        return drift > self.threshold

    def get_trades(self, current_weights: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.should_rebalance(current_weights):
            turnover = float(np.sum(np.abs(self.target_weights - current_weights)))
            return self.target_weights.copy(), turnover
        return current_weights.copy(), 0.0


# ============================================================
# No-Trade Zone (Band) Rebalancing
# ============================================================

class BandRebalancer:
    """
    No-trade zone rebalancing.
    Only trade assets that breach their individual bands.
    """

    def __init__(self, config: RebalancingConfig, target_weights: np.ndarray):
        self.config = config
        self.target_weights = target_weights
        self.band_width = config.band_width

    def get_trades(self, current_weights: np.ndarray) -> Tuple[np.ndarray, float]:
        new_weights = current_weights.copy()

        for i in range(len(current_weights)):
            lower = self.target_weights[i] - self.band_width
            upper = self.target_weights[i] + self.band_width

            if current_weights[i] < lower or current_weights[i] > upper:
                new_weights[i] = self.target_weights[i]

        # Re-normalize
        total = np.sum(new_weights)
        if total > 0:
            new_weights /= total

        turnover = float(np.sum(np.abs(new_weights - current_weights)))
        return new_weights, turnover


# ============================================================
# RL-Based Rebalancing Agent
# ============================================================

class RebalancingPolicyNetwork(nn.Module):
    """
    RL policy that learns WHEN and HOW MUCH to rebalance.

    Outputs:
    - Target weights (what to rebalance toward)
    - Rebalancing intensity alpha in [0, 1] (how aggressively)
    """

    def __init__(self, state_dim: int, num_assets: int, hidden_dim: int = 128):
        super().__init__()
        self.num_assets = num_assets

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Target weight head
        self.target_head = nn.Linear(hidden_dim, num_assets)

        # Rebalancing intensity head
        self.intensity_head = nn.Linear(hidden_dim, 1)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(state)

        target_logits = self.target_head(features)
        target_weights = F.softmax(target_logits, dim=-1)

        intensity = torch.sigmoid(self.intensity_head(features)).squeeze(-1)

        value = self.value_head(features).squeeze(-1)

        return {
            "target_weights": target_weights,
            "intensity": intensity,
            "value": value,
        }

    def get_action(
        self,
        state: torch.Tensor,
        current_weights: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        """
        Get new weights via partial rebalancing.
        new_w = (1 - alpha) * current_w + alpha * target_w
        """
        with torch.no_grad():
            output = self.forward(state)
            alpha = output["intensity"].item()
            target = output["target_weights"].numpy().flatten()

        current = current_weights.numpy().flatten()
        new_weights = (1 - alpha) * current + alpha * target
        turnover = float(np.sum(np.abs(new_weights - current)))

        return new_weights, turnover


# ============================================================
# Rebalancing Environment
# ============================================================

class RebalancingEnv:
    """
    Environment for training rebalancing agents.
    Reward balances tracking error vs. transaction costs.
    """

    def __init__(
        self,
        prices: np.ndarray,
        target_weights: np.ndarray,
        config: RebalancingConfig,
    ):
        self.prices = prices
        self.target_weights = target_weights
        self.config = config
        self.num_assets = config.num_assets
        self.drift_sim = PortfolioDriftSimulator(config.num_assets)

        self.current_step = 0
        self.current_weights = target_weights.copy()
        self.portfolio_value = config.initial_capital
        self.total_cost = 0.0
        self.total_tracking_error = 0.0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.current_weights = self.target_weights.copy()
        self.portfolio_value = self.config.initial_capital
        self.total_cost = 0.0
        self.total_tracking_error = 0.0
        return self._get_state()

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        new_weights = action["new_weights"]
        turnover = action["turnover"]

        tc = self.config.transaction_cost * turnover
        self.total_cost += tc

        idx = self.current_step
        if idx + 1 >= len(self.prices):
            return self._get_state(), 0.0, True, {}

        returns = (self.prices[idx + 1] - self.prices[idx]) / (self.prices[idx] + 1e-8)
        port_return = float(np.dot(new_weights, returns))
        self.portfolio_value *= (1.0 + port_return - tc)

        self.current_weights = self.drift_sim.compute_drift(new_weights, returns)

        te = float(np.sum((self.current_weights - self.target_weights) ** 2))
        self.total_tracking_error += te

        reward = port_return - tc - self.config.tracking_error_penalty * te

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        info = {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "tracking_error": te,
            "transaction_cost": tc,
        }
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        state = np.concatenate([
            self.current_weights,
            self.target_weights,
            self.current_weights - self.target_weights,
            np.array([
                self.portfolio_value / self.config.initial_capital - 1.0,
                self.current_step / max(len(self.prices), 1),
            ]),
        ]).astype(np.float32)
        return state


# ============================================================
# Demonstration
# ============================================================

def generate_synthetic_data(
    num_assets: int = 5, num_steps: int = 500, seed: int = 42
) -> np.ndarray:
    """Generate synthetic prices."""
    np.random.seed(seed)
    returns = np.random.randn(num_steps, num_assets) * 0.01 + 0.0002
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return prices


def demo_rebalancing_strategies():
    """Compare different rebalancing strategies."""
    print("=" * 70)
    print("Rebalancing Strategies Comparison")
    print("=" * 70)

    num_assets = 5
    prices = generate_synthetic_data(num_assets=num_assets, num_steps=500)
    returns = np.diff(prices, axis=0) / (prices[:-1] + 1e-8)

    config = RebalancingConfig(
        num_assets=num_assets,
        transaction_cost=0.001,
        calendar_period=20,
        threshold=0.05,
        band_width=0.03,
    )

    target_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    drift_sim = PortfolioDriftSimulator(num_assets)

    strategies = {
        "No Rebalancing": None,
        "Calendar (Monthly)": CalendarRebalancer(config, target_weights),
        "Threshold (5%)": ThresholdRebalancer(config, target_weights),
        "Band (3%)": BandRebalancer(config, target_weights),
    }

    results = {}
    for name, strategy in strategies.items():
        weights = target_weights.copy()
        portfolio_value = 1_000_000.0
        total_turnover = 0.0
        total_tc = 0.0
        rebalance_count = 0
        max_drift = 0.0

        if isinstance(strategy, CalendarRebalancer):
            strategy.reset()

        for t in range(len(returns)):
            drifted = drift_sim.compute_drift(weights, returns[t])
            drift = float(np.sum(np.abs(drifted - target_weights)))
            max_drift = max(max_drift, drift)

            if strategy is None:
                weights = drifted
                turnover = 0.0
            else:
                weights, turnover = strategy.get_trades(drifted)

            if turnover > 0:
                rebalance_count += 1
                total_turnover += turnover
                tc = config.transaction_cost * turnover
                total_tc += tc

            port_return = float(np.dot(weights, returns[t]))
            portfolio_value *= (1.0 + port_return - config.transaction_cost * turnover)

        total_return = (portfolio_value / 1_000_000.0 - 1) * 100
        results[name] = {
            "return": total_return,
            "rebalances": rebalance_count,
            "total_turnover": total_turnover,
            "total_cost": total_tc,
            "max_drift": max_drift,
            "final_value": portfolio_value,
        }

    print(f"\n{'Strategy':<22} {'Return%':>9} {'#Rebal':>8} "
          f"{'Turnover':>10} {'Cost($)':>10} {'MaxDrift':>10}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:<22} {r['return']:>8.2f}% {r['rebalances']:>7d} "
              f"{r['total_turnover']:>9.3f} {r['total_cost']*1e6:>9.0f} "
              f"{r['max_drift']:>9.4f}")

    # --- Drift simulation ---
    print("\n--- Drift Simulation (No Rebalancing) ---")
    drift_path = drift_sim.simulate_drift_path(target_weights, returns[:60])
    print(f"Initial weights: {drift_path[0]}")
    print(f"After 20 days:   {drift_path[20]}")
    print(f"After 40 days:   {drift_path[40]}")
    print(f"After 60 days:   {drift_path[60]}")
    print(f"Max single-asset drift: "
          f"{np.max(np.abs(drift_path - target_weights)):.4f}")

    # --- RL Rebalancing Policy ---
    print("\n--- RL Rebalancing Policy Architecture ---")
    state_dim = num_assets * 3 + 2  # weights + target + drift + 2
    policy = RebalancingPolicyNetwork(state_dim, num_assets)
    params = sum(p.numel() for p in policy.parameters())
    print(f"State dim: {state_dim}")
    print(f"Parameters: {params:,}")

    # Test forward pass
    state = torch.randn(1, state_dim)
    output = policy(state)
    print(f"Target weights: {output['target_weights'].detach().numpy().flatten()}")
    print(f"Intensity (alpha): {output['intensity'].item():.4f}")
    print(f"Value: {output['value'].item():.4f}")

    # Partial rebalancing demo
    current = torch.FloatTensor([0.35, 0.20, 0.18, 0.17, 0.10])
    new_w, turnover = policy.get_action(state, current)
    print(f"\nPartial rebalancing:")
    print(f"  Current:   {current.numpy()}")
    print(f"  New:       {new_w}")
    print(f"  Turnover:  {turnover:.4f}")

    # --- Rebalancing Environment ---
    print("\n--- Rebalancing Environment ---")
    env = RebalancingEnv(prices[:200], target_weights, config)
    state = env.reset(seed=42)
    print(f"State dim: {len(state)}")

    total_reward = 0.0
    for step in range(100):
        # Use band rebalancer as action source
        band = BandRebalancer(config, target_weights)
        new_w, turnover = band.get_trades(env.current_weights)
        action = {"new_weights": new_w, "turnover": turnover}
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Steps: {step + 1}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final value: ${info['portfolio_value']:,.2f}")
    print(f"Total cost: ${env.total_cost * 1e6:.0f}")


if __name__ == "__main__":
    demo_rebalancing_strategies()
