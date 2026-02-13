"""
Chapter 35.3.4: Momentum Trading
==================================
RL-based momentum trading with time-series and cross-sectional
momentum signals, crash protection, and adaptive lookback.
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
class MomentumConfig:
    """Configuration for momentum trading."""
    num_assets: int = 20
    lookback_periods: List[int] = None  # Multiple lookbacks
    episode_length: int = 252
    transaction_cost: float = 0.001
    max_position: float = 0.10          # Per asset
    rebalance_frequency: int = 20       # Monthly
    top_quantile: float = 0.3           # Long top 30%
    bottom_quantile: float = 0.3        # Short bottom 30%

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m


# ============================================================
# Momentum Signal Generators
# ============================================================

class MomentumSignals:
    """Compute various momentum signals."""

    @staticmethod
    def simple_return(prices: np.ndarray, lookback: int) -> np.ndarray:
        """Simple price momentum: P_t / P_{t-L} - 1."""
        if len(prices) <= lookback:
            return np.zeros(prices.shape[1])
        return prices[-1] / (prices[-lookback - 1] + 1e-8) - 1

    @staticmethod
    def risk_adjusted(prices: np.ndarray, lookback: int) -> np.ndarray:
        """Risk-adjusted momentum: return / volatility."""
        if len(prices) <= lookback:
            return np.zeros(prices.shape[1])
        returns = np.diff(np.log(prices[-lookback - 1:] + 1e-8), axis=0)
        mean_ret = np.mean(returns, axis=0)
        vol = np.std(returns, axis=0) + 1e-8
        return mean_ret / vol

    @staticmethod
    def residual_momentum(
        prices: np.ndarray, market_prices: np.ndarray, lookback: int
    ) -> np.ndarray:
        """Residual momentum: return adjusted for market factor."""
        if len(prices) <= lookback:
            return np.zeros(prices.shape[1])
        returns = np.diff(np.log(prices[-lookback - 1:] + 1e-8), axis=0)
        mkt_returns = np.diff(np.log(market_prices[-lookback - 1:] + 1e-8), axis=0)

        residuals = np.zeros(prices.shape[1])
        for i in range(prices.shape[1]):
            beta = np.cov(returns[:, i], mkt_returns.flatten())[0, 1] / (
                np.var(mkt_returns) + 1e-8
            )
            residual_ret = returns[:, i] - beta * mkt_returns.flatten()
            residuals[i] = np.mean(residual_ret)
        return residuals

    @staticmethod
    def multi_lookback(
        prices: np.ndarray, lookbacks: List[int]
    ) -> np.ndarray:
        """Concatenated signals across multiple lookback periods."""
        signals = []
        for lb in lookbacks:
            sig = MomentumSignals.simple_return(prices, lb)
            signals.append(sig)
        return np.column_stack(signals)  # (N, num_lookbacks)


# ============================================================
# Cross-Sectional Momentum Strategy
# ============================================================

class CrossSectionalMomentum:
    """Traditional long-short cross-sectional momentum."""

    def __init__(self, config: MomentumConfig):
        self.config = config

    def compute_weights(
        self, prices: np.ndarray, lookback: int = 126
    ) -> np.ndarray:
        """
        Compute momentum portfolio weights.
        Long top quantile, short bottom quantile.
        """
        N = prices.shape[1]
        returns = MomentumSignals.simple_return(prices, lookback)

        # Rank assets
        ranks = np.argsort(np.argsort(-returns))  # 0 = best

        weights = np.zeros(N)
        n_long = max(1, int(N * self.config.top_quantile))
        n_short = max(1, int(N * self.config.bottom_quantile))

        # Long top
        long_mask = ranks < n_long
        weights[long_mask] = 1.0 / n_long

        # Short bottom
        short_mask = ranks >= (N - n_short)
        weights[short_mask] = -1.0 / n_short

        # Clip
        weights = np.clip(weights, -self.config.max_position, self.config.max_position)

        return weights


# ============================================================
# Time-Series Momentum Strategy
# ============================================================

class TimeSeriesMomentum:
    """Time-series (trend-following) momentum."""

    def __init__(self, config: MomentumConfig):
        self.config = config

    def compute_weights(
        self, prices: np.ndarray, lookback: int = 126
    ) -> np.ndarray:
        """Position based on own past return."""
        N = prices.shape[1]
        returns = MomentumSignals.simple_return(prices, lookback)
        vol = np.std(
            np.diff(np.log(prices[-lookback:] + 1e-8), axis=0), axis=0
        ) + 1e-8

        # Volatility-scaled positions
        raw_weights = np.sign(returns) * (1.0 / (vol * np.sqrt(252)))
        weights = np.clip(raw_weights, -self.config.max_position, self.config.max_position)

        # Scale to target exposure
        total = np.sum(np.abs(weights))
        if total > 1.0:
            weights /= total

        return weights


# ============================================================
# Momentum Crash Detector
# ============================================================

class CrashDetector:
    """Detect conditions associated with momentum crashes."""

    def __init__(self, vol_threshold: float = 0.025, drawdown_threshold: float = -0.10):
        self.vol_threshold = vol_threshold
        self.drawdown_threshold = drawdown_threshold

    def compute_crash_risk(
        self, market_prices: np.ndarray, lookback: int = 60
    ) -> Dict[str, float]:
        """Estimate momentum crash risk."""
        if len(market_prices) < lookback:
            return {"crash_risk": 0.0, "vol": 0.0, "drawdown": 0.0}

        returns = np.diff(np.log(market_prices[-lookback:] + 1e-8))
        vol = np.std(returns) * np.sqrt(252)

        peak = np.max(market_prices[-lookback:])
        drawdown = (market_prices[-1] / peak) - 1

        # Simple crash risk score
        vol_risk = max(0, (vol - self.vol_threshold) / self.vol_threshold)
        dd_risk = max(0, (self.drawdown_threshold - drawdown) / abs(self.drawdown_threshold))
        crash_risk = min(1.0, (vol_risk + dd_risk) / 2)

        return {
            "crash_risk": float(crash_risk),
            "vol": float(vol),
            "drawdown": float(drawdown),
        }


# ============================================================
# Momentum Trading Environment
# ============================================================

class MomentumTradingEnv:
    """RL environment for momentum trading."""

    def __init__(self, prices: np.ndarray, config: MomentumConfig):
        self.prices = prices
        self.config = config
        self.num_assets = config.num_assets
        self.market_prices = prices.mean(axis=1)  # Equal-weight market

        lookback_max = max(config.lookback_periods)
        self.state_dim = (
            len(config.lookback_periods) * config.num_assets  # momentum signals
            + config.num_assets  # volatilities
            + config.num_assets  # current positions
            + 3  # crash risk, time, portfolio value
        )
        self.action_dim = config.num_assets

        self.current_step = 0
        self.positions = np.zeros(config.num_assets)
        self.portfolio_value = 1.0
        self.crash_detector = CrashDetector()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        lookback_max = max(self.config.lookback_periods)
        max_start = len(self.prices) - self.config.episode_length - lookback_max
        if max_start > lookback_max:
            self.start_idx = np.random.randint(lookback_max, max_start)
        else:
            self.start_idx = lookback_max
        self.current_step = 0
        self.positions = np.zeros(self.num_assets)
        self.portfolio_value = 1.0
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Action: target position per asset in [-max_pos, max_pos]."""
        new_positions = np.clip(action, -self.config.max_position, self.config.max_position)

        # Turnover cost
        turnover = np.sum(np.abs(new_positions - self.positions))
        tc = self.config.transaction_cost * turnover

        # Forward return
        idx = self.start_idx + self.current_step
        if idx + 1 >= len(self.prices):
            return self._get_state(), 0.0, True, {}

        returns = (self.prices[idx + 1] - self.prices[idx]) / (self.prices[idx] + 1e-8)
        port_return = float(np.dot(new_positions, returns)) - tc

        self.portfolio_value *= (1 + port_return)
        self.positions = new_positions
        self.current_step += 1

        done = self.current_step >= self.config.episode_length
        reward = port_return

        info = {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "positions": new_positions.copy(),
        }
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        idx = self.start_idx + self.current_step
        features = []

        # Multi-lookback momentum signals
        for lb in self.config.lookback_periods:
            start = max(0, idx - lb)
            sig = MomentumSignals.simple_return(self.prices[start:idx + 1], lb)
            features.append(sig)

        # Volatilities
        vol_window = self.prices[max(0, idx - 60):idx + 1]
        if len(vol_window) > 1:
            vols = np.std(np.diff(np.log(vol_window + 1e-8), axis=0), axis=0)
        else:
            vols = np.zeros(self.num_assets)
        features.append(vols)

        # Current positions
        features.append(self.positions)

        # Crash risk
        crash = self.crash_detector.compute_crash_risk(self.market_prices[:idx + 1])
        features.append(np.array([
            crash["crash_risk"],
            self.current_step / self.config.episode_length,
            self.portfolio_value - 1.0,
        ]))

        return np.concatenate(features).astype(np.float32)


# ============================================================
# Momentum Policy Network
# ============================================================

class MomentumPolicy(nn.Module):
    """Continuous policy for momentum trading."""

    def __init__(self, state_dim: int, num_assets: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, num_assets)
        self.log_std = nn.Parameter(torch.ones(num_assets) * -1.0)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(state)
        mean = torch.tanh(self.mean_head(features))
        value = self.value_head(features).squeeze(-1)
        return {"mean": mean, "log_std": self.log_std, "value": value}


# ============================================================
# Demonstration
# ============================================================

def generate_momentum_data(
    num_assets: int = 20, num_steps: int = 1500, seed: int = 42
) -> np.ndarray:
    """Generate synthetic data with momentum patterns."""
    np.random.seed(seed)
    prices = np.zeros((num_steps, num_assets))
    prices[0] = 100.0

    # Assets have different drift rates (creates cross-sectional dispersion)
    drifts = np.random.uniform(-0.05, 0.15, num_assets) / 252
    vols = np.random.uniform(0.15, 0.40, num_assets) / np.sqrt(252)

    # Add autocorrelation (momentum effect)
    momentum_strength = 0.02
    prev_returns = np.zeros(num_assets)

    for t in range(1, num_steps):
        noise = np.random.randn(num_assets) * vols
        returns = drifts + momentum_strength * prev_returns + noise
        prices[t] = prices[t - 1] * np.exp(returns)
        prev_returns = returns

    return prices


def demo_momentum_trading():
    """Demonstrate momentum trading strategies."""
    print("=" * 70)
    print("Momentum Trading Demonstration")
    print("=" * 70)

    num_assets = 20
    prices = generate_momentum_data(num_assets=num_assets, num_steps=1500)
    print(f"\nData: {prices.shape[0]} steps, {num_assets} assets")

    config = MomentumConfig(num_assets=num_assets)

    # --- Momentum Signals ---
    print("\n--- Momentum Signals (Current) ---")
    for lb in config.lookback_periods:
        sig = MomentumSignals.simple_return(prices, lb)
        print(f"  {lb:3d}-day return: mean={np.mean(sig):+.4f}, "
              f"std={np.std(sig):.4f}, range=[{sig.min():+.4f}, {sig.max():+.4f}]")

    risk_adj = MomentumSignals.risk_adjusted(prices, 126)
    print(f"  Risk-adj (126d): mean={np.mean(risk_adj):+.4f}")

    # --- Cross-Sectional Momentum ---
    print("\n--- Cross-Sectional Momentum ---")
    cs_mom = CrossSectionalMomentum(config)
    weights = cs_mom.compute_weights(prices, lookback=126)
    long_count = np.sum(weights > 0)
    short_count = np.sum(weights < 0)
    print(f"Long positions: {long_count}, Short positions: {short_count}")
    print(f"Net exposure: {np.sum(weights):.4f}")
    print(f"Gross exposure: {np.sum(np.abs(weights)):.4f}")

    # --- Time-Series Momentum ---
    print("\n--- Time-Series Momentum ---")
    ts_mom = TimeSeriesMomentum(config)
    weights_ts = ts_mom.compute_weights(prices, lookback=126)
    print(f"Long: {np.sum(weights_ts > 0)}, Short: {np.sum(weights_ts < 0)}")
    print(f"Net exposure: {np.sum(weights_ts):.4f}")

    # --- Crash Detection ---
    print("\n--- Crash Risk Assessment ---")
    market = prices.mean(axis=1)
    detector = CrashDetector()
    risk = detector.compute_crash_risk(market)
    print(f"Crash risk: {risk['crash_risk']:.4f}")
    print(f"Market vol: {risk['vol']:.4f}")
    print(f"Market drawdown: {risk['drawdown']:.4f}")

    # --- Backtest comparison ---
    print("\n--- Strategy Backtest (last 252 days) ---")
    test_prices = prices[-312:]  # Extra lookback
    strategies = {"Cross-Sectional": cs_mom, "Time-Series": ts_mom}

    for name, strat in strategies.items():
        portfolio_value = 1.0
        for t in range(252, len(test_prices) - 1):
            w = strat.compute_weights(test_prices[:t + 1], lookback=126)
            ret = (test_prices[t + 1] - test_prices[t]) / (test_prices[t] + 1e-8)
            port_ret = np.dot(w, ret) - 0.001 * 0.1  # Rough TC
            portfolio_value *= (1 + port_ret)

        total_ret = (portfolio_value - 1) * 100
        print(f"  {name:<20}: Return = {total_ret:+.2f}%")

    # --- RL Environment + Policy ---
    print("\n--- RL Momentum Environment ---")
    env = MomentumTradingEnv(prices, config)
    state = env.reset(seed=42)
    print(f"State dim: {len(state)}")

    policy = MomentumPolicy(state_dim=len(state), num_assets=num_assets)
    params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {params:,}")

    with torch.no_grad():
        out = policy(torch.FloatTensor(state).unsqueeze(0))
        print(f"Mean positions: {out['mean'][0][:5].numpy()}... (first 5)")
        print(f"Value estimate: {out['value'].item():.4f}")


if __name__ == "__main__":
    demo_momentum_trading()
