"""
Chapter 35.3.2: Market Making
==============================
RL-based market making with Avellaneda-Stoikov baseline,
inventory management, and fill probability modeling.
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
class MarketMakingConfig:
    """Configuration for market making environment."""
    initial_mid_price: float = 100.0
    tick_size: float = 0.01
    max_inventory: int = 100
    time_horizon: int = 1000

    # Market dynamics
    volatility: float = 0.0002
    drift: float = 0.0
    order_arrival_rate: float = 0.5
    fill_decay: float = 1.5

    # Reward parameters
    inventory_penalty: float = 0.001
    terminal_penalty: float = 0.01

    # Action space
    num_spread_levels: int = 5
    min_spread: float = 0.01
    max_spread: float = 0.10
    num_skew_levels: int = 5


# ============================================================
# Order Book Simulator
# ============================================================

class SimpleOrderBook:
    """Simplified order book for market making simulation."""

    def __init__(self, config: MarketMakingConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.mid_price = config.initial_mid_price
        self.step_count = 0

    def reset(self):
        self.mid_price = self.config.initial_mid_price
        self.step_count = 0

    def simulate_step(self, bid_offset: float, ask_offset: float) -> Dict:
        cfg = self.config

        p_bid_fill = cfg.order_arrival_rate * np.exp(-cfg.fill_decay * bid_offset)
        p_ask_fill = cfg.order_arrival_rate * np.exp(-cfg.fill_decay * ask_offset)

        bid_filled = self.rng.random() < p_bid_fill
        ask_filled = self.rng.random() < p_ask_fill

        bid_price = self.mid_price - bid_offset
        ask_price = self.mid_price + ask_offset

        price_change = self.rng.normal(cfg.drift, cfg.volatility) * self.mid_price
        self.mid_price += price_change
        self.step_count += 1

        return {
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "mid_price": self.mid_price,
            "price_change": price_change,
            "spread": ask_price - bid_price,
        }


# ============================================================
# Market Making Environment
# ============================================================

class MarketMakingEnv:
    """
    RL environment for market making.
    State: (inventory, mid_price_change, volatility, time_frac, pnl, trade_rate)
    Action: (spread_level x skew_level) discretized
    Reward: PnL - inventory_penalty * q^2
    """

    def __init__(self, config: MarketMakingConfig, seed: Optional[int] = None):
        self.config = config
        self.order_book = SimpleOrderBook(config, seed)

        spreads = np.linspace(config.min_spread, config.max_spread, config.num_spread_levels)
        skews = np.linspace(-0.5, 0.5, config.num_skew_levels)
        self.spread_levels = spreads
        self.skew_levels = skews
        self.action_dim = config.num_spread_levels * config.num_skew_levels
        self.state_dim = 6

        self.inventory = 0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.trades = 0
        self.current_step = 0
        self.price_history: List[float] = []
        self.pnl_history: List[float] = []

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.order_book.rng = np.random.RandomState(seed)
        self.order_book.reset()
        self.inventory = 0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.trades = 0
        self.current_step = 0
        self.price_history = [self.config.initial_mid_price]
        self.pnl_history = []
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        spread_idx = action // self.config.num_skew_levels
        skew_idx = action % self.config.num_skew_levels

        half_spread = self.spread_levels[spread_idx]
        skew = self.skew_levels[skew_idx]

        bid_offset = half_spread * (1 + skew)
        ask_offset = half_spread * (1 - skew)

        result = self.order_book.simulate_step(bid_offset, ask_offset)

        if result["bid_filled"] and self.inventory < self.config.max_inventory:
            self.inventory += 1
            self.cash -= result["bid_price"]
            self.trades += 1

        if result["ask_filled"] and self.inventory > -self.config.max_inventory:
            self.inventory -= 1
            self.cash += result["ask_price"]
            self.trades += 1

        mtm = self.cash + self.inventory * result["mid_price"]
        step_pnl = mtm - self.total_pnl
        self.total_pnl = mtm

        self.price_history.append(result["mid_price"])
        self.pnl_history.append(step_pnl)

        inventory_cost = self.config.inventory_penalty * self.inventory ** 2
        reward = step_pnl - inventory_cost

        self.current_step += 1
        done = self.current_step >= self.config.time_horizon

        if done and self.inventory != 0:
            terminal_cost = (self.config.terminal_penalty
                             * abs(self.inventory) * result["mid_price"])
            reward -= terminal_cost

        info = {
            "inventory": self.inventory,
            "cash": self.cash,
            "total_pnl": self.total_pnl,
            "trades": self.trades,
            "mid_price": result["mid_price"],
            "spread": result["spread"],
        }
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        mid = self.price_history[-1]
        price_change = 0.0
        volatility = self.config.volatility
        if len(self.price_history) > 1:
            price_change = (self.price_history[-1] / self.price_history[-2]) - 1
        if len(self.price_history) > 20:
            returns = np.diff(np.log(self.price_history[-20:]))
            volatility = np.std(returns)

        return np.array([
            self.inventory / self.config.max_inventory,
            price_change,
            volatility / self.config.volatility,
            self.current_step / self.config.time_horizon,
            self.total_pnl / (mid + 1e-8),
            self.trades / max(self.current_step, 1),
        ], dtype=np.float32)


# ============================================================
# Avellaneda-Stoikov Baseline
# ============================================================

class AvellanedaStoikov:
    """
    Avellaneda-Stoikov market making model.
    Reservation price: r = mid - q * gamma * sigma^2 * (T-t)
    Optimal spread: delta = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)
    """

    def __init__(self, gamma: float = 0.1, sigma: float = 0.0002,
                 k: float = 1.5, time_horizon: int = 1000):
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.T = time_horizon

    def get_quotes(self, mid_price: float, inventory: int, step: int) -> Tuple[float, float]:
        tau = max(1, self.T - step) / self.T

        reservation_adj = inventory * self.gamma * self.sigma ** 2 * tau
        spread = (self.gamma * self.sigma ** 2 * tau
                  + (2 / self.gamma) * np.log(1 + self.gamma / self.k))

        half_spread = spread / 2
        bid_offset = max(0.01, half_spread + reservation_adj)
        ask_offset = max(0.01, half_spread - reservation_adj)

        return bid_offset, ask_offset


# ============================================================
# Market Making Policy Network
# ============================================================

class MarketMakingPolicy(nn.Module):
    """Policy network for market making."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(0, self.network[-1].out_features)
        with torch.no_grad():
            q_values = self.forward(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax(dim=1).item()


# ============================================================
# Demonstration
# ============================================================

def demo_market_making():
    """Demonstrate market making strategies."""
    print("=" * 70)
    print("Market Making Demonstration")
    print("=" * 70)

    config = MarketMakingConfig(
        initial_mid_price=100.0,
        max_inventory=100,
        time_horizon=500,
        volatility=0.0002,
        inventory_penalty=0.001,
    )

    # --- Avellaneda-Stoikov ---
    print("\n--- Avellaneda-Stoikov Baseline ---")
    as_model = AvellanedaStoikov(
        gamma=0.1, sigma=config.volatility, k=config.fill_decay,
        time_horizon=config.time_horizon,
    )

    num_trials = 20
    as_results = {"pnl": [], "trades": [], "max_inv": []}

    for trial in range(num_trials):
        env = MarketMakingEnv(config, seed=trial)
        state = env.reset()
        max_inv = 0

        for step in range(config.time_horizon):
            bid_off, ask_off = as_model.get_quotes(
                env.price_history[-1], env.inventory, step
            )
            spread_idx = np.argmin(np.abs(env.spread_levels - (bid_off + ask_off) / 2))
            skew_val = (bid_off - ask_off) / (bid_off + ask_off + 1e-8)
            skew_idx = np.argmin(np.abs(env.skew_levels - skew_val))
            action = spread_idx * config.num_skew_levels + skew_idx

            state, reward, done, info = env.step(action)
            max_inv = max(max_inv, abs(env.inventory))
            if done:
                break

        as_results["pnl"].append(info["total_pnl"])
        as_results["trades"].append(info["trades"])
        as_results["max_inv"].append(max_inv)

    print(f"Mean PnL: ${np.mean(as_results['pnl']):.2f} "
          f"(std: ${np.std(as_results['pnl']):.2f})")
    print(f"Mean trades: {np.mean(as_results['trades']):.0f}")
    print(f"Mean max inventory: {np.mean(as_results['max_inv']):.1f}")

    # --- Random policy ---
    print("\n--- Random Policy ---")
    rand_results = {"pnl": [], "trades": [], "max_inv": []}

    for trial in range(num_trials):
        env = MarketMakingEnv(config, seed=trial)
        state = env.reset()
        max_inv = 0

        for step in range(config.time_horizon):
            action = np.random.randint(0, env.action_dim)
            state, reward, done, info = env.step(action)
            max_inv = max(max_inv, abs(env.inventory))
            if done:
                break

        rand_results["pnl"].append(info["total_pnl"])
        rand_results["trades"].append(info["trades"])
        rand_results["max_inv"].append(max_inv)

    print(f"Mean PnL: ${np.mean(rand_results['pnl']):.2f} "
          f"(std: ${np.std(rand_results['pnl']):.2f})")
    print(f"Mean trades: {np.mean(rand_results['trades']):.0f}")
    print(f"Mean max inventory: {np.mean(rand_results['max_inv']):.1f}")

    # --- Policy Network ---
    print("\n--- RL Policy Network ---")
    policy = MarketMakingPolicy(state_dim=6, action_dim=env.action_dim)
    params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {params:,}")
    print(f"Action space: {env.action_dim} "
          f"({config.num_spread_levels} spreads x {config.num_skew_levels} skews)")

    test_state = torch.randn(1, 6)
    q_values = policy(test_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {q_values.argmax(dim=1).item()}")

    # --- Fill probability analysis ---
    print("\n--- Fill Probability Analysis ---")
    print(f"{'Offset':>8} {'Fill Prob':>10}")
    print("-" * 20)
    for offset in [0.01, 0.02, 0.05, 0.10, 0.20]:
        prob = config.order_arrival_rate * np.exp(-config.fill_decay * offset)
        print(f"{offset:>8.2f} {prob:>9.4f}")


if __name__ == "__main__":
    demo_market_making()
