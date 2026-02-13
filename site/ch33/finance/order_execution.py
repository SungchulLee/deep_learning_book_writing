"""
33.7.1 Order Execution
========================

DQN-based optimal order execution with simulated market impact.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
import random


# ---------------------------------------------------------------------------
# Order Execution Environment
# ---------------------------------------------------------------------------

class OrderExecutionEnv:
    """Simulated order execution environment with market impact.
    
    The agent must liquidate Q shares over T periods.
    """

    def __init__(self, total_shares: int = 10000, n_periods: int = 20,
                 initial_price: float = 100.0, volatility: float = 0.02,
                 temp_impact: float = 0.1, perm_impact: float = 0.01,
                 spread: float = 0.01, n_actions: int = 11):
        self.Q = total_shares
        self.T = n_periods
        self.p0 = initial_price
        self.sigma = volatility
        self.eta = temp_impact      # Temporary impact coefficient
        self.gamma_imp = perm_impact  # Permanent impact coefficient
        self.spread = spread
        self.n_actions = n_actions  # Discretized fractions: 0, 0.1, 0.2, ..., 1.0

        self.state_dim = 5  # (inventory_frac, time_frac, price_return, volume, volatility)
        self.action_dim = n_actions

    def reset(self) -> np.ndarray:
        self.inventory = self.Q
        self.time_step = 0
        self.price = self.p0
        self.arrival_price = self.p0
        self.total_cost = 0.0
        self.execution_log = []
        self.volume_history = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        inv_frac = self.inventory / self.Q
        time_frac = self.time_step / self.T
        price_ret = (self.price - self.arrival_price) / self.arrival_price
        volume = np.random.lognormal(0, 0.3)  # Random volume
        vol = self.sigma * np.sqrt(1 + 0.5 * np.random.randn())
        self.volume_history.append(volume)
        return np.array([inv_frac, time_frac, price_ret, volume, vol], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Convert action to fraction of remaining inventory
        fraction = action / (self.n_actions - 1)

        # Determine shares to trade
        if self.time_step >= self.T - 1:
            # Force liquidation at terminal time
            n_shares = self.inventory
        else:
            n_shares = int(fraction * self.inventory)
            n_shares = max(0, min(n_shares, self.inventory))

        # Market impact
        volume = max(self.volume_history[-1] * self.Q * 0.1, 1)
        participation = n_shares / volume if volume > 0 else 0

        temp_cost = self.eta * participation * self.price * n_shares
        perm_shift = self.gamma_imp * n_shares / self.Q * self.price
        spread_cost = self.spread * self.price * n_shares * 0.5

        # Execute
        exec_price = self.price + temp_cost / max(n_shares, 1) + self.spread * 0.5
        total_exec_cost = temp_cost + spread_cost

        # Permanent impact on price
        self.price -= perm_shift

        # Random price evolution
        self.price *= np.exp(self.sigma * np.random.randn())

        # Update state
        self.inventory -= n_shares
        self.time_step += 1
        self.total_cost += total_exec_cost

        # Reward: negative execution cost (implementation shortfall)
        reward = -total_exec_cost / (self.Q * self.arrival_price) * 1000  # Scale

        # Penalty for holding inventory (urgency)
        if self.inventory > 0:
            holding_penalty = -0.1 * (self.inventory / self.Q) ** 2
            reward += holding_penalty

        done = (self.time_step >= self.T) or (self.inventory <= 0)

        self.execution_log.append({
            'step': self.time_step, 'shares': n_shares, 'price': exec_price,
            'cost': total_exec_cost, 'remaining': self.inventory,
        })

        info = {'exec_cost': total_exec_cost, 'shares_traded': n_shares,
                'inventory': self.inventory, 'participation': participation}

        return self._get_state(), reward, done, info


# ---------------------------------------------------------------------------
# TWAP and VWAP Baselines
# ---------------------------------------------------------------------------

def twap_policy(env: OrderExecutionEnv) -> List[Dict]:
    """Time-Weighted Average Price: equal shares per period."""
    state = env.reset()
    shares_per_period = env.Q // env.T
    total_reward = 0
    done = False
    while not done:
        n = min(shares_per_period, env.inventory)
        action = int(round(n / max(env.inventory, 1) * (env.n_actions - 1)))
        action = max(0, min(action, env.n_actions - 1))
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward, env.execution_log


# ---------------------------------------------------------------------------
# DQN Agent for Order Execution
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


class OrderExecDQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 batch_size=64, buf_cap=50000, target_freq=200,
                 eps_end=0.05, eps_decay=5000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_freq = target_freq
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.online = QNet(state_dim, action_dim)
        self.target = QNet(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)

        self.buf = deque(maxlen=buf_cap)
        self.step = 0
        self.updates = 0

    @property
    def epsilon(self):
        return max(self.eps_end, 1.0 - (1.0 - self.eps_end) * self.step / self.eps_decay)

    def act(self, state, training=True):
        if training:
            self.step += 1
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.online(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

    def store(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, float(d)))

    def update(self):
        if len(self.buf) < 500:
            return
        batch = random.sample(self.buf, self.batch_size)
        s = torch.FloatTensor([t[0] for t in batch])
        a = torch.LongTensor([t[1] for t in batch])
        r = torch.FloatTensor([t[2] for t in batch])
        ns = torch.FloatTensor([t[3] for t in batch])
        d = torch.FloatTensor([t[4] for t in batch])

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a = self.online(ns).argmax(1)
            nq = self.target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            tgt = r + (1 - d) * self.gamma * nq
        loss = nn.functional.smooth_l1_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


def demo_order_execution():
    print("=" * 60)
    print("Order Execution Demo")
    print("=" * 60)

    env = OrderExecutionEnv(total_shares=10000, n_periods=20)
    print(f"\nProblem: Liquidate {env.Q} shares over {env.T} periods")
    print(f"State dim: {env.state_dim}, Actions: {env.action_dim}")

    # TWAP baseline
    print("\n--- TWAP Baseline (10 trials) ---")
    twap_rewards = []
    for _ in range(10):
        r, _ = twap_policy(env)
        twap_rewards.append(r)
    print(f"  TWAP reward: {np.mean(twap_rewards):.2f} ± {np.std(twap_rewards):.2f}")

    # Train DQN
    print("\n--- DQN Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    agent = OrderExecDQN(env.state_dim, env.action_dim, lr=1e-3, eps_decay=3000)

    rewards_hist = []
    for ep in range(500):
        s = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s)
            ns, r, done, info = env.step(a)
            agent.store(s, a, r, ns, done)
            agent.update()
            s = ns; total += r
        rewards_hist.append(total)
        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}: avg100={np.mean(rewards_hist[-100:]):.2f}")

    # Evaluate DQN
    print("\n--- DQN Evaluation (50 trials) ---")
    dqn_rewards = []
    for _ in range(50):
        s = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s, training=False)
            s, r, done, _ = env.step(a)
            total += r
        dqn_rewards.append(total)
    print(f"  DQN reward: {np.mean(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
    print(f"  TWAP reward: {np.mean(twap_rewards):.2f} ± {np.std(twap_rewards):.2f}")
    improvement = (np.mean(dqn_rewards) - np.mean(twap_rewards))
    print(f"  Improvement over TWAP: {improvement:+.2f}")

    print("\nOrder execution demo complete!")


if __name__ == "__main__":
    demo_order_execution()
