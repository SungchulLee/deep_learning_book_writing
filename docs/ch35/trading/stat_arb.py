"""
Chapter 35.3.3: Statistical Arbitrage
=======================================
RL-based pairs trading with spread modeling, cointegration testing,
Kalman filter hedge ratios, and multi-pair portfolio management.
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
class StatArbConfig:
    """Configuration for statistical arbitrage."""
    lookback: int = 60
    zscore_window: int = 20
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss: float = 4.0
    max_holding_period: int = 20
    transaction_cost: float = 0.001
    max_position: float = 1.0
    num_actions: int = 5  # -1, -0.5, 0, 0.5, 1


# ============================================================
# Spread Model (Ornstein-Uhlenbeck)
# ============================================================

class OUProcess:
    """
    Ornstein-Uhlenbeck process for spread modeling.
    dz = theta * (mu - z) * dt + sigma * dW
    """

    def __init__(self, theta: float = 5.0, mu: float = 0.0, sigma: float = 0.1):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def simulate(self, z0: float, num_steps: int, dt: float = 1/252,
                 seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        z = np.zeros(num_steps + 1)
        z[0] = z0
        for t in range(num_steps):
            dz = self.theta * (self.mu - z[t]) * dt + self.sigma * np.sqrt(dt) * rng.randn()
            z[t + 1] = z[t] + dz
        return z

    @staticmethod
    def estimate_params(spread: np.ndarray, dt: float = 1/252) -> Dict[str, float]:
        """Estimate OU parameters from spread data using OLS."""
        z = spread[:-1]
        dz = np.diff(spread)

        # OLS: dz = a + b*z + epsilon
        X = np.column_stack([np.ones(len(z)), z])
        beta = np.linalg.lstsq(X, dz, rcond=None)[0]
        a, b = beta

        theta = -b / dt
        mu = a / (theta * dt) if abs(theta * dt) > 1e-8 else 0.0
        residuals = dz - X @ beta
        sigma = np.std(residuals) / np.sqrt(dt)

        half_life = np.log(2) / abs(theta) if abs(theta) > 1e-8 else float('inf')

        return {
            "theta": float(theta),
            "mu": float(mu),
            "sigma": float(sigma),
            "half_life": float(half_life),
        }


# ============================================================
# Cointegration Testing
# ============================================================

class CointegrationTester:
    """Simple cointegration tests for pairs selection."""

    @staticmethod
    def compute_hedge_ratio(prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """OLS hedge ratio: log(A) = alpha + beta * log(B) + epsilon."""
        log_a = np.log(prices_a + 1e-8)
        log_b = np.log(prices_b + 1e-8)
        X = np.column_stack([np.ones(len(log_b)), log_b])
        beta = np.linalg.lstsq(X, log_a, rcond=None)[0]
        return float(beta[1])

    @staticmethod
    def compute_spread(prices_a: np.ndarray, prices_b: np.ndarray,
                       hedge_ratio: float) -> np.ndarray:
        return np.log(prices_a + 1e-8) - hedge_ratio * np.log(prices_b + 1e-8)

    @staticmethod
    def adf_test_simple(spread: np.ndarray) -> Dict:
        """Simplified ADF test (Dickey-Fuller)."""
        y = spread[:-1]
        dy = np.diff(spread)
        X = np.column_stack([np.ones(len(y)), y])
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        residuals = dy - X @ beta
        se = np.sqrt(np.sum(residuals**2) / (len(dy) - 2) /
                     (np.sum((y - y.mean())**2) + 1e-8))
        t_stat = beta[1] / (se + 1e-8)

        # Approximate critical values
        is_stationary = t_stat < -2.86  # 5% level
        return {"t_statistic": float(t_stat), "is_stationary": is_stationary}


# ============================================================
# Kalman Filter Hedge Ratio
# ============================================================

class KalmanHedgeRatio:
    """Online Kalman filter for dynamic hedge ratio estimation."""

    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        self.delta = delta
        self.R = R
        self.theta = np.zeros(2)  # [alpha, beta]
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * delta

    def update(self, price_a: float, price_b: float) -> float:
        """Update hedge ratio with new observation."""
        y = np.log(price_a + 1e-8)
        x = np.array([1.0, np.log(price_b + 1e-8)])

        # Predict
        y_hat = x @ self.theta
        S = x @ self.P @ x + self.R
        K = self.P @ x / (S + 1e-8)

        # Update
        error = y - y_hat
        self.theta += K * error
        self.P = self.P - np.outer(K, x) @ self.P + self.Q

        return float(self.theta[1])  # Return hedge ratio


# ============================================================
# Pairs Trading Environment
# ============================================================

class PairsTradingEnv:
    """
    RL environment for pairs trading.
    State: spread features + position info
    Action: position in spread (-1 to +1)
    """

    def __init__(self, prices_a: np.ndarray, prices_b: np.ndarray,
                 config: StatArbConfig):
        self.prices_a = prices_a
        self.prices_b = prices_b
        self.config = config

        # Compute hedge ratio and spread
        self.hedge_ratio = CointegrationTester.compute_hedge_ratio(prices_a, prices_b)
        self.spread = CointegrationTester.compute_spread(prices_a, prices_b, self.hedge_ratio)

        self.action_values = np.linspace(-config.max_position, config.max_position, config.num_actions)
        self.state_dim = 8
        self.action_dim = config.num_actions

        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.holding_time = 0
        self.total_pnl = 0.0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        start_max = len(self.spread) - self.config.lookback - 100
        if start_max > self.config.lookback:
            self.start_idx = np.random.randint(self.config.lookback, start_max)
        else:
            self.start_idx = self.config.lookback
        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.holding_time = 0
        self.total_pnl = 0.0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        new_position = self.action_values[action]
        idx = self.start_idx + self.current_step

        # Trade
        position_change = abs(new_position - self.position)
        tc = self.config.transaction_cost * position_change

        # PnL from spread change
        if idx + 1 < len(self.spread):
            spread_change = self.spread[idx + 1] - self.spread[idx]
            step_pnl = self.position * spread_change - tc
        else:
            step_pnl = -tc

        self.total_pnl += step_pnl

        # Update position
        if self.position == 0 and new_position != 0:
            self.entry_price = self.spread[idx]
            self.holding_time = 0
        elif new_position == 0:
            self.holding_time = 0
        else:
            self.holding_time += 1

        self.position = new_position
        self.current_step += 1

        done = (self.current_step >= 100 or
                idx + 1 >= len(self.spread) - 1 or
                self.holding_time >= self.config.max_holding_period)

        reward = step_pnl
        info = {
            "total_pnl": self.total_pnl,
            "position": self.position,
            "spread": self.spread[min(idx, len(self.spread) - 1)],
            "holding_time": self.holding_time,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        idx = self.start_idx + self.current_step
        idx = min(idx, len(self.spread) - 1)
        window = self.config.zscore_window

        spread_window = self.spread[max(0, idx - window):idx + 1]
        mu = np.mean(spread_window)
        sigma = np.std(spread_window) + 1e-8
        zscore = (self.spread[idx] - mu) / sigma

        # OU parameter estimation
        if len(spread_window) > 10:
            params = OUProcess.estimate_params(spread_window)
            theta = params["theta"]
            ou_sigma = params["sigma"]
        else:
            theta = 0.0
            ou_sigma = sigma

        return np.array([
            self.spread[idx],
            zscore,
            theta,
            ou_sigma,
            self.position,
            self.total_pnl,
            self.holding_time / max(self.config.max_holding_period, 1),
            self.current_step / 100.0,
        ], dtype=np.float32)


# ============================================================
# Traditional Pairs Trading Strategy
# ============================================================

class ThresholdPairsStrategy:
    """Traditional z-score threshold pairs trading."""

    def __init__(self, config: StatArbConfig):
        self.config = config

    def get_action(self, state: np.ndarray, action_values: np.ndarray) -> int:
        zscore = state[1]
        current_pos = state[4]

        if current_pos == 0:
            if zscore > self.config.entry_threshold:
                target = -self.config.max_position  # Short spread
            elif zscore < -self.config.entry_threshold:
                target = self.config.max_position   # Long spread
            else:
                target = 0.0
        else:
            if abs(zscore) < self.config.exit_threshold:
                target = 0.0
            elif abs(zscore) > self.config.stop_loss:
                target = 0.0
            else:
                target = current_pos

        return int(np.argmin(np.abs(action_values - target)))


# ============================================================
# Stat Arb Policy Network
# ============================================================

class StatArbPolicy(nn.Module):
    """DQN policy for statistical arbitrage."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ============================================================
# Demonstration
# ============================================================

def generate_cointegrated_pair(
    num_steps: int = 1000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a cointegrated pair of price series."""
    np.random.seed(seed)

    # Common factor (random walk)
    common = 100 + np.cumsum(np.random.randn(num_steps) * 0.5)

    # Cointegrated pair
    noise_a = np.cumsum(np.random.randn(num_steps) * 0.1)
    noise_b = np.cumsum(np.random.randn(num_steps) * 0.1)

    # Mean-reverting spread
    ou = OUProcess(theta=5.0, mu=0.0, sigma=0.3)
    spread = ou.simulate(0.0, num_steps - 1, seed=seed)

    prices_a = np.exp(np.log(common) + spread + noise_a * 0.01)
    prices_b = np.exp(np.log(common * 0.8) + noise_b * 0.01)

    return prices_a, prices_b


def demo_stat_arb():
    """Demonstrate statistical arbitrage."""
    print("=" * 70)
    print("Statistical Arbitrage Demonstration")
    print("=" * 70)

    prices_a, prices_b = generate_cointegrated_pair(num_steps=1000)
    print(f"\nAsset A: [{prices_a.min():.2f}, {prices_a.max():.2f}]")
    print(f"Asset B: [{prices_b.min():.2f}, {prices_b.max():.2f}]")

    # --- Cointegration analysis ---
    print("\n--- Cointegration Analysis ---")
    hedge_ratio = CointegrationTester.compute_hedge_ratio(prices_a, prices_b)
    spread = CointegrationTester.compute_spread(prices_a, prices_b, hedge_ratio)
    adf = CointegrationTester.adf_test_simple(spread)
    print(f"Hedge ratio (beta): {hedge_ratio:.4f}")
    print(f"ADF t-statistic: {adf['t_statistic']:.4f}")
    print(f"Stationary: {adf['is_stationary']}")

    # --- OU parameters ---
    print("\n--- OU Process Parameters ---")
    params = OUProcess.estimate_params(spread)
    print(f"Theta (mean reversion speed): {params['theta']:.4f}")
    print(f"Mu (long-run mean): {params['mu']:.4f}")
    print(f"Sigma (volatility): {params['sigma']:.4f}")
    print(f"Half-life: {params['half_life']:.1f} days")

    # --- Kalman filter ---
    print("\n--- Kalman Filter Hedge Ratio ---")
    kf = KalmanHedgeRatio()
    kf_ratios = []
    for t in range(len(prices_a)):
        ratio = kf.update(prices_a[t], prices_b[t])
        kf_ratios.append(ratio)
    print(f"Static hedge ratio: {hedge_ratio:.4f}")
    print(f"Kalman final ratio: {kf_ratios[-1]:.4f}")
    print(f"Kalman ratio range: [{min(kf_ratios):.4f}, {max(kf_ratios):.4f}]")

    # --- Pairs trading strategies ---
    config = StatArbConfig(
        lookback=60, zscore_window=20,
        entry_threshold=2.0, exit_threshold=0.5,
        transaction_cost=0.001,
    )

    env = PairsTradingEnv(prices_a, prices_b, config)
    threshold_strategy = ThresholdPairsStrategy(config)

    print("\n--- Threshold Strategy ---")
    num_trials = 30
    threshold_pnls = []
    for trial in range(num_trials):
        state = env.reset(seed=trial)
        for _ in range(100):
            action = threshold_strategy.get_action(state, env.action_values)
            state, reward, done, info = env.step(action)
            if done:
                break
        threshold_pnls.append(info["total_pnl"])

    print(f"Mean PnL: {np.mean(threshold_pnls):.4f} (std: {np.std(threshold_pnls):.4f})")
    print(f"Win rate: {np.mean(np.array(threshold_pnls) > 0) * 100:.1f}%")

    print("\n--- Random Policy ---")
    random_pnls = []
    for trial in range(num_trials):
        state = env.reset(seed=trial)
        for _ in range(100):
            action = np.random.randint(0, config.num_actions)
            state, reward, done, info = env.step(action)
            if done:
                break
        random_pnls.append(info["total_pnl"])
    print(f"Mean PnL: {np.mean(random_pnls):.4f} (std: {np.std(random_pnls):.4f})")

    # --- Policy Network ---
    print("\n--- RL Policy Network ---")
    policy = StatArbPolicy(state_dim=8, action_dim=config.num_actions)
    params_count = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {params_count:,}")
    print(f"State dim: 8, Action dim: {config.num_actions}")


if __name__ == "__main__":
    demo_stat_arb()
