"""
Chapter 34.7.2: Continuous Trading with Policy-Based RL
========================================================
SAC-based continuous trading agent with position management,
transaction costs, and risk constraints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque


class TradingEnv:
    """
    Simplified continuous trading environment.
    Agent controls position size in [-1, 1] at each timestep.
    """
    
    def __init__(self, prices, window=30, max_position=1.0,
                 spread_cost=0.0002, impact_cost=0.0001):
        self.prices = prices
        self.window = window
        self.max_position = max_position
        self.spread_cost = spread_cost
        self.impact_cost = impact_cost
        
        self.returns = np.diff(prices) / prices[:-1]
        self.obs_dim = window + 3  # returns window + position + unrealized_pnl + volatility
        self.reset()
    
    def reset(self):
        self.t = self.window
        self.position = 0.0
        self.entry_price = self.prices[self.t]
        self.pnl = 0.0
        self.trades = 0
        return self._get_obs()
    
    def _get_obs(self):
        ret_window = self.returns[self.t - self.window:self.t]
        unrealized = self.position * (self.prices[self.t] - self.entry_price) / self.entry_price
        vol = ret_window.std()
        return np.concatenate([ret_window, [self.position, unrealized, vol]]).astype(np.float32)
    
    def step(self, target_position):
        target_position = np.clip(target_position, -self.max_position, self.max_position)
        
        # Trade execution
        trade_size = abs(target_position - self.position)
        cost = trade_size * (self.spread_cost + self.impact_cost * trade_size)
        
        # Position change
        old_position = self.position
        self.position = target_position
        if abs(target_position - old_position) > 0.01:
            self.entry_price = self.prices[self.t]
            self.trades += 1
        
        # Move to next timestep
        self.t += 1
        done = self.t >= len(self.prices) - 1
        
        if not done:
            price_return = self.returns[self.t - 1]
            reward = self.position * price_return - cost
            self.pnl += reward
        else:
            reward = 0.0
        
        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, {"pnl": self.pnl, "trades": self.trades}


class TradingActor(nn.Module):
    """Squashed Gaussian actor for position sizing."""
    
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
    
    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)
        return mu, log_std
    
    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        return action.squeeze(-1), log_prob.squeeze(-1)


class TradingCritic(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, obs, action):
        sa = torch.cat([obs, action.unsqueeze(-1) if action.dim() == 1 else action], -1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class TradingSAC:
    """SAC agent for continuous trading."""
    
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=50000, batch_size=128, warmup=1000):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.warmup = warmup
        
        obs_dim = env.obs_dim
        self.actor = TradingActor(obs_dim)
        self.critic = TradingCritic(obs_dim)
        self.critic_target = TradingCritic(obs_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Simple buffer
        self.buffer = {"s": [], "a": [], "r": [], "s2": [], "d": []}
        self.buffer_size = buffer_size
    
    def _store(self, s, a, r, s2, d):
        if len(self.buffer["s"]) >= self.buffer_size:
            for k in self.buffer:
                self.buffer[k] = self.buffer[k][-self.buffer_size//2:]
        for k, v in zip(["s", "a", "r", "s2", "d"], [s, a, r, s2, float(d)]):
            self.buffer[k].append(v)
    
    def _sample(self):
        n = len(self.buffer["s"])
        idx = np.random.randint(0, n, self.batch_size)
        return tuple(torch.FloatTensor(np.array([self.buffer[k][i] for i in idx]))
                     for k in ["s", "a", "r", "s2", "d"])
    
    def update(self):
        if len(self.buffer["s"]) < self.batch_size:
            return
        
        s, a, r, s2, d = self._sample()
        
        with torch.no_grad():
            a2, lp2 = self.actor.sample(s2)
            q1t, q2t = self.critic_target(s2, a2)
            target = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - self.alpha * lp2)
        
        q1, q2 = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        new_a, new_lp = self.actor.sample(s)
        q1_new, _ = self.critic(s, new_a)
        actor_loss = (self.alpha * new_lp - q1_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
    
    def train(self, n_episodes=100, print_interval=10):
        results = []
        
        for ep in range(1, n_episodes + 1):
            obs = self.env.reset()
            total_reward = 0
            step = 0
            
            done = False
            while not done:
                if len(self.buffer["s"]) < self.warmup:
                    action = np.random.uniform(-1, 1)
                else:
                    with torch.no_grad():
                        action, _ = self.actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                    action = action.item()
                
                next_obs, reward, done, info = self.env.step(action)
                self._store(obs, [action], reward, next_obs, done)
                
                if len(self.buffer["s"]) >= self.warmup:
                    self.update()
                
                total_reward += reward
                obs = next_obs
                step += 1
            
            results.append({"pnl": info["pnl"], "trades": info["trades"]})
            
            if ep % print_interval == 0:
                recent_pnl = [r["pnl"] for r in results[-print_interval:]]
                print(f"Episode {ep:>4d} | PnL: {info['pnl']*100:>7.3f}% | "
                      f"Trades: {info['trades']:>4d} | Avg PnL: {np.mean(recent_pnl)*100:>7.3f}%")
        
        return results


def generate_price_series(n_days=500, seed=42):
    np.random.seed(seed)
    returns = np.random.normal(0.0001, 0.015, n_days)
    # Add momentum regime
    for i in range(100, 200):
        returns[i] += 0.002
    for i in range(300, 400):
        returns[i] -= 0.002
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


def demo_trading():
    print("=" * 60)
    print("Continuous Trading with SAC")
    print("=" * 60)
    
    prices = generate_price_series(n_days=500)
    env = TradingEnv(prices, window=30)
    agent = TradingSAC(env, lr=3e-4, warmup=500)
    results = agent.train(n_episodes=80, print_interval=20)
    
    buy_hold = (prices[-1] / prices[30] - 1) * 100
    print(f"\nBuy-and-hold: {buy_hold:.2f}%")


if __name__ == "__main__":
    demo_trading()
