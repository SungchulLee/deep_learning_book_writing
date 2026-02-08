"""
33.7.2 Market Making
=====================

DQN-based market making with inventory management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Tuple, List
import random


class MarketMakingEnv:
    """Simplified market making environment."""

    def __init__(self, n_steps: int = 200, mid_price: float = 100.0,
                 volatility: float = 0.001, tick_size: float = 0.01,
                 fill_prob_base: float = 0.5, max_inventory: int = 50,
                 inventory_penalty: float = 0.001):
        self.n_steps = n_steps
        self.p0 = mid_price
        self.sigma = volatility
        self.tick = tick_size
        self.fill_base = fill_prob_base
        self.max_inv = max_inventory
        self.inv_penalty = inventory_penalty

        # Action space: (bid_offset, ask_offset) in ticks
        # 0: (1,1), 1: (1,2), 2: (2,1), 3: (2,2), 4: (1,3), 5: (3,1),
        # 6: (2,3), 7: (3,2), 8: (3,3)
        self.action_map = [
            (1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)]
        self.action_dim = len(self.action_map)
        self.state_dim = 5  # inventory_norm, price_return, volatility, spread, time_frac

    def reset(self):
        self.price = self.p0
        self.inventory = 0
        self.step_count = 0
        self.pnl = 0.0
        self.trades = 0
        return self._state()

    def _state(self):
        inv_norm = self.inventory / self.max_inv
        price_ret = (self.price - self.p0) / self.p0
        vol = self.sigma + 0.0005 * np.random.randn()
        spread = self.tick * 2 / self.price
        time_frac = self.step_count / self.n_steps
        return np.array([inv_norm, price_ret, vol, spread, time_frac], dtype=np.float32)

    def step(self, action: int):
        bid_off, ask_off = self.action_map[action]
        bid = self.price - bid_off * self.tick
        ask = self.price + ask_off * self.tick

        # Fill probability (decreases with offset, adjusted for imbalance)
        p_bid_fill = self.fill_base * np.exp(-0.5 * (bid_off - 1))
        p_ask_fill = self.fill_base * np.exp(-0.5 * (ask_off - 1))

        reward = 0.0

        # Bid fill (we buy)
        if np.random.random() < p_bid_fill and self.inventory < self.max_inv:
            self.inventory += 1
            self.pnl -= bid
            reward += self.price - bid  # Spread capture
            self.trades += 1

        # Ask fill (we sell)
        if np.random.random() < p_ask_fill and self.inventory > -self.max_inv:
            self.inventory -= 1
            self.pnl += ask
            reward += ask - self.price  # Spread capture
            self.trades += 1

        # Inventory penalty
        reward -= self.inv_penalty * self.inventory ** 2

        # Mark-to-market from price change
        old_price = self.price
        self.price *= np.exp(self.sigma * np.random.randn())
        mtm = self.inventory * (self.price - old_price)
        reward += mtm

        self.step_count += 1
        done = self.step_count >= self.n_steps

        # Unwind at terminal
        if done and self.inventory != 0:
            unwind_cost = abs(self.inventory) * self.tick * 2  # Pay 2 ticks spread
            reward -= unwind_cost

        return self._state(), reward, done, {
            'inventory': self.inventory, 'trades': self.trades,
            'pnl': self.pnl + self.inventory * self.price}


# ---------------------------------------------------------------------------
# DQN Market Maker
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


class MarketMakerDQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 batch_size=64, buf_cap=50000, target_freq=200,
                 eps_end=0.05, eps_decay=5000):
        self.gamma = gamma; self.batch_size = batch_size
        self.action_dim = action_dim; self.target_freq = target_freq
        self.eps_end = eps_end; self.eps_decay = eps_decay

        self.online = QNet(state_dim, action_dim)
        self.target = QNet(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = deque(maxlen=buf_cap)
        self.step = 0; self.updates = 0

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
        if len(self.buf) < 500: return
        batch = random.sample(self.buf, self.batch_size)
        s = torch.FloatTensor([t[0] for t in batch])
        a = torch.LongTensor([t[1] for t in batch])
        r = torch.FloatTensor([t[2] for t in batch])
        ns = torch.FloatTensor([t[3] for t in batch])
        d = torch.FloatTensor([t[4] for t in batch])
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            ba = self.online(ns).argmax(1)
            nq = self.target(ns).gather(1, ba.unsqueeze(1)).squeeze(1)
            tgt = r + (1 - d) * self.gamma * nq
        loss = nn.functional.smooth_l1_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


def demo_market_making():
    print("=" * 60)
    print("Market Making Demo")
    print("=" * 60)

    env = MarketMakingEnv()
    print(f"\nState dim: {env.state_dim}, Actions: {env.action_dim}")
    print(f"Action map (bid_off, ask_off): {env.action_map}")

    # Baseline: always tight spread
    print("\n--- Tight Spread Baseline ---")
    baseline_pnls = []
    for _ in range(50):
        s = env.reset(); total = 0; done = False
        while not done:
            s, r, done, info = env.step(0)  # Always (1,1)
            total += r
        baseline_pnls.append(total)
    print(f"  Reward: {np.mean(baseline_pnls):.3f} ± {np.std(baseline_pnls):.3f}")

    # Train DQN
    print("\n--- DQN Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    agent = MarketMakerDQN(env.state_dim, env.action_dim, lr=1e-3, eps_decay=3000)

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
            print(f"  Episode {ep+1}: avg100={np.mean(rewards_hist[-100:]):.3f}")

    # Evaluate
    print("\n--- DQN Evaluation ---")
    dqn_pnls = []
    for _ in range(50):
        s = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s, training=False)
            s, r, done, info = env.step(a)
            total += r
        dqn_pnls.append(total)

    print(f"  DQN reward: {np.mean(dqn_pnls):.3f} ± {np.std(dqn_pnls):.3f}")
    print(f"  Baseline:   {np.mean(baseline_pnls):.3f} ± {np.std(baseline_pnls):.3f}")

    print("\nMarket making demo complete!")


if __name__ == "__main__":
    demo_market_making()
