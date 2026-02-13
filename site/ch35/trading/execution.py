"""
Chapter 35.3.1: Execution Algorithms
=====================================
RL-based optimal trade execution with TWAP, VWAP, and Almgren-Chriss
benchmarks. Includes execution environment and DQN-based agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import random


# ============================================================
# Configuration
# ============================================================

@dataclass
class ExecutionConfig:
    """Configuration for execution environment."""
    total_quantity: float = 10000.0     # Shares to execute
    time_horizon: int = 78              # 6.5 hours in 5-min bars
    tick_size: float = 0.01
    initial_price: float = 100.0

    # Market parameters
    daily_volume: float = 1_000_000.0
    daily_volatility: float = 0.02
    spread_mean: float = 0.02          # $0.02 spread
    spread_std: float = 0.005

    # Impact parameters
    permanent_impact: float = 0.0001   # bps per share
    temporary_impact: float = 0.001    # bps per sqrt(share)

    # Penalty
    incomplete_penalty: float = 0.1    # Per remaining share at deadline

    # RL
    num_actions: int = 11              # 0%, 10%, 20%, ..., 100% of remaining


# ============================================================
# Market Simulator for Execution
# ============================================================

class ExecutionMarketSimulator:
    """Simulates market dynamics during order execution."""

    def __init__(self, config: ExecutionConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Per-bar parameters
        self.bar_vol = config.daily_volatility / np.sqrt(config.time_horizon)
        self.bar_volume = config.daily_volume / config.time_horizon

    def reset(self) -> Dict:
        """Reset market state."""
        self.price = self.config.initial_price
        self.arrival_price = self.price
        return self._get_market_state()

    def step(self, trade_volume: float) -> Tuple[float, Dict]:
        """
        Simulate one time bar with a trade.

        Args:
            trade_volume: shares to execute this bar

        Returns:
            execution_price, market_state
        """
        cfg = self.config

        # Market volume this bar (random)
        market_vol = self.bar_volume * self.rng.lognormal(0, 0.3)

        # Spread
        spread = max(cfg.tick_size, self.rng.normal(cfg.spread_mean, cfg.spread_std))

        # Participation rate
        participation = trade_volume / (market_vol + 1e-8)

        # Temporary impact (square root)
        temp_impact = cfg.temporary_impact * np.sqrt(trade_volume) * self.price

        # Permanent impact
        perm_impact = cfg.permanent_impact * trade_volume * self.price

        # Execution price = mid + half_spread + temp_impact + random slippage
        slippage = self.rng.normal(0, self.bar_vol * self.price * 0.1)
        exec_price = self.price + spread / 2 + temp_impact + slippage

        # Update price (permanent impact + random walk)
        random_move = self.rng.normal(0, self.bar_vol * self.price)
        self.price += perm_impact + random_move

        state = self._get_market_state()
        state["market_volume"] = market_vol
        state["spread"] = spread
        state["participation_rate"] = participation

        return exec_price, state

    def _get_market_state(self) -> Dict:
        return {
            "price": self.price,
            "volatility": self.bar_vol,
        }


# ============================================================
# Execution Environment
# ============================================================

class ExecutionEnv:
    """
    RL environment for optimal trade execution.

    State: (remaining_qty_frac, time_frac, price_change, spread, volume, volatility)
    Action: fraction of remaining quantity to execute (discretized)
    Reward: negative implementation shortfall
    """

    def __init__(self, config: ExecutionConfig, seed: Optional[int] = None):
        self.config = config
        self.market = ExecutionMarketSimulator(config, seed)

        # Action mapping: index → fraction of remaining
        self.action_fractions = np.linspace(0, 1, config.num_actions)

        self.state_dim = 6
        self.action_dim = config.num_actions

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.market.rng = np.random.RandomState(seed)

        market_state = self.market.reset()
        self.remaining_qty = self.config.total_quantity
        self.arrival_price = market_state["price"]
        self.current_step = 0
        self.total_cost = 0.0
        self.executed_qty = 0.0
        self.exec_prices: List[float] = []
        self.exec_volumes: List[float] = []

        return self._get_state(market_state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step.

        Args:
            action: index into action_fractions
        """
        fraction = self.action_fractions[action]
        trade_qty = fraction * self.remaining_qty

        # Execute trade
        if trade_qty > 0:
            exec_price, market_state = self.market.step(trade_qty)
            cost = (exec_price - self.arrival_price) * trade_qty
            self.total_cost += cost
            self.remaining_qty -= trade_qty
            self.executed_qty += trade_qty
            self.exec_prices.append(exec_price)
            self.exec_volumes.append(trade_qty)
        else:
            _, market_state = self.market.step(0)
            exec_price = market_state["price"]
            cost = 0.0

        self.current_step += 1

        # Reward: negative cost (want to minimize shortfall)
        reward = -cost / (self.config.total_quantity * self.arrival_price + 1e-8)

        # Termination
        terminated = self.remaining_qty <= 0
        truncated = self.current_step >= self.config.time_horizon

        # Incomplete penalty
        if truncated and self.remaining_qty > 0:
            penalty = (self.config.incomplete_penalty
                       * self.remaining_qty * market_state["price"])
            reward -= penalty / (self.config.total_quantity * self.arrival_price + 1e-8)

        state = self._get_state(market_state)
        info = {
            "remaining_qty": self.remaining_qty,
            "executed_qty": self.executed_qty,
            "total_cost": self.total_cost,
            "implementation_shortfall_bps": self._compute_shortfall_bps(),
            "vwap": self._compute_vwap(),
        }

        return state, reward, terminated, truncated, info

    def _get_state(self, market_state: Dict) -> np.ndarray:
        return np.array([
            self.remaining_qty / self.config.total_quantity,
            self.current_step / self.config.time_horizon,
            (market_state["price"] - self.arrival_price) / self.arrival_price,
            market_state.get("spread", self.config.spread_mean) / self.arrival_price,
            market_state.get("market_volume", self.config.daily_volume / self.config.time_horizon)
            / (self.config.daily_volume / self.config.time_horizon),
            market_state["volatility"] / self.config.daily_volatility * np.sqrt(self.config.time_horizon),
        ], dtype=np.float32)

    def _compute_shortfall_bps(self) -> float:
        if self.executed_qty <= 0:
            return 0.0
        avg_price = self.total_cost / self.executed_qty + self.arrival_price
        return (avg_price / self.arrival_price - 1) * 10000

    def _compute_vwap(self) -> float:
        if not self.exec_volumes or sum(self.exec_volumes) == 0:
            return self.arrival_price
        total_value = sum(p * v for p, v in zip(self.exec_prices, self.exec_volumes))
        return total_value / sum(self.exec_volumes)


# ============================================================
# Benchmark Strategies
# ============================================================

class TWAPStrategy:
    """Time-Weighted Average Price execution."""

    def __init__(self, total_qty: float, time_horizon: int):
        self.qty_per_bar = total_qty / time_horizon
        self.time_horizon = time_horizon

    def get_action(self, state: np.ndarray, action_fractions: np.ndarray) -> int:
        remaining_frac = state[0]
        time_frac = state[1]
        remaining_steps = max(1, int((1 - time_frac) * self.time_horizon))
        target_frac = 1.0 / remaining_steps
        return int(np.argmin(np.abs(action_fractions - target_frac)))


class VWAPStrategy:
    """Volume-Weighted Average Price execution."""

    def __init__(self, volume_profile: np.ndarray):
        """
        Args:
            volume_profile: (T,) expected volume per bar (normalized to sum=1)
        """
        self.profile = volume_profile / volume_profile.sum()
        self.cumulative = np.cumsum(self.profile)

    def get_action(self, state: np.ndarray, action_fractions: np.ndarray, step: int) -> int:
        remaining_frac = state[0]
        if step >= len(self.profile):
            return len(action_fractions) - 1  # Execute all remaining

        target_cum = self.cumulative[step]
        executed_frac = 1.0 - remaining_frac
        target_this_bar = max(0, target_cum - executed_frac)
        frac = target_this_bar / (remaining_frac + 1e-8)
        frac = np.clip(frac, 0, 1)
        return int(np.argmin(np.abs(action_fractions - frac)))


class AlmgrenChrissStrategy:
    """Almgren-Chriss optimal execution trajectory."""

    def __init__(
        self,
        total_qty: float,
        time_horizon: int,
        volatility: float,
        risk_aversion: float = 1e-6,
        temp_impact: float = 0.001,
    ):
        self.total_qty = total_qty
        self.T = time_horizon
        self.kappa = np.sqrt(risk_aversion * volatility ** 2 / (temp_impact + 1e-8))

        # Precompute optimal trajectory
        self.trajectory = np.zeros(time_horizon)
        for t in range(time_horizon):
            self.trajectory[t] = (
                total_qty
                * np.sinh(self.kappa * (time_horizon - t))
                / np.sinh(self.kappa * time_horizon + 1e-8)
            )

    def get_action(self, state: np.ndarray, action_fractions: np.ndarray, step: int) -> int:
        remaining_frac = state[0]
        remaining_qty = remaining_frac * self.total_qty

        if step >= self.T:
            return len(action_fractions) - 1

        target_remaining = self.trajectory[step]
        trade_qty = max(0, remaining_qty - target_remaining)
        frac = trade_qty / (remaining_qty + 1e-8)
        frac = np.clip(frac, 0, 1)
        return int(np.argmin(np.abs(action_fractions - frac)))


# ============================================================
# DQN Execution Agent
# ============================================================

class ExecutionDQN(nn.Module):
    """DQN network for execution decisions."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ExecutionDQNAgent:
    """DQN agent for optimal execution."""

    def __init__(self, config: ExecutionConfig, lr: float = 1e-3):
        self.config = config
        self.state_dim = 6
        self.num_actions = config.num_actions
        self.action_fractions = np.linspace(0, 1, config.num_actions)

        self.q_net = ExecutionDQN(self.state_dim, self.num_actions)
        self.target_net = ExecutionDQN(self.state_dim, self.num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 100
        self.step_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)

        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()


# ============================================================
# Demonstration
# ============================================================

def demo_execution_algorithms():
    """Demonstrate execution algorithms."""
    print("=" * 70)
    print("Execution Algorithms Demonstration")
    print("=" * 70)

    config = ExecutionConfig(
        total_quantity=10000,
        time_horizon=78,
        initial_price=100.0,
        daily_volume=1_000_000,
    )

    # --- Run benchmark strategies ---
    strategies = {}

    # TWAP
    twap = TWAPStrategy(config.total_quantity, config.time_horizon)
    strategies["TWAP"] = twap

    # VWAP (U-shaped volume profile)
    t = np.linspace(0, 1, config.time_horizon)
    volume_profile = 1.0 + 0.5 * np.cos(2 * np.pi * t)
    vwap = VWAPStrategy(volume_profile)
    strategies["VWAP"] = vwap

    # Almgren-Chriss
    ac = AlmgrenChrissStrategy(
        config.total_quantity, config.time_horizon,
        volatility=config.daily_volatility, risk_aversion=1e-6,
    )
    strategies["Almgren-Chriss"] = ac

    results = {}
    num_trials = 50

    for name, strategy in strategies.items():
        shortfalls = []
        for trial in range(num_trials):
            env = ExecutionEnv(config, seed=trial)
            state = env.reset()
            total_reward = 0.0

            for step in range(config.time_horizon):
                if name == "TWAP":
                    action = strategy.get_action(state, env.action_fractions)
                elif name == "VWAP":
                    action = strategy.get_action(state, env.action_fractions, step)
                else:
                    action = strategy.get_action(state, env.action_fractions, step)

                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            shortfalls.append(info["implementation_shortfall_bps"])

        results[name] = {
            "mean_shortfall": np.mean(shortfalls),
            "std_shortfall": np.std(shortfalls),
            "median_shortfall": np.median(shortfalls),
        }

    print(f"\n{'Strategy':<18} {'Mean IS (bps)':>14} {'Std (bps)':>12} {'Median (bps)':>14}")
    print("-" * 62)
    for name, r in results.items():
        print(f"{name:<18} {r['mean_shortfall']:>13.2f} {r['std_shortfall']:>11.2f} "
              f"{r['median_shortfall']:>13.2f}")

    # --- DQN Agent ---
    print("\n--- DQN Execution Agent ---")
    agent = ExecutionDQNAgent(config, lr=1e-3)
    params = sum(p.numel() for p in agent.q_net.parameters())
    print(f"Q-network parameters: {params:,}")
    print(f"Action space: {config.num_actions} discrete actions")
    print(f"State dim: {agent.state_dim}")

    # Quick training loop
    print("\nTraining (100 episodes)...")
    for episode in range(100):
        env = ExecutionEnv(config, seed=episode + 1000)
        state = env.reset()
        episode_reward = 0.0

        for step in range(config.time_horizon):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            episode_reward += reward
            if done:
                break

        if (episode + 1) % 25 == 0:
            print(f"  Episode {episode+1}: reward={episode_reward:.4f}, "
                  f"IS={info['implementation_shortfall_bps']:.2f}bps, "
                  f"eps={agent.epsilon:.3f}")

    # Evaluate trained agent
    print("\nEvaluating trained agent (50 trials)...")
    dqn_shortfalls = []
    for trial in range(50):
        env = ExecutionEnv(config, seed=trial + 5000)
        state = env.reset()
        for step in range(config.time_horizon):
            action = agent.select_action(state, training=False)
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        dqn_shortfalls.append(info["implementation_shortfall_bps"])

    print(f"DQN Agent:  Mean IS = {np.mean(dqn_shortfalls):.2f} bps, "
          f"Std = {np.std(dqn_shortfalls):.2f} bps")


if __name__ == "__main__":
    demo_execution_algorithms()
