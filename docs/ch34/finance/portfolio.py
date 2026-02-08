"""
Chapter 34.7.1: Portfolio Optimization with Policy-Based RL
=============================================================
PPO-based portfolio optimization with transaction costs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


class PortfolioEnv:
    """Multi-asset portfolio management environment."""
    
    def __init__(self, returns_data, window=20, transaction_cost=0.001, risk_penalty=0.5):
        self.returns = returns_data
        self.n_assets = returns_data.shape[1]
        self.window = window
        self.tc = transaction_cost
        self.risk_penalty = risk_penalty
        self.features_per_asset = 4
        self.obs_dim = self.n_assets * self.features_per_asset + self.n_assets
        self.reset()
    
    def reset(self, start_idx=None):
        self.t = start_idx if start_idx else self.window
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.values_history = [1.0]
        return self._get_obs()
    
    def _get_obs(self):
        features = []
        for i in range(self.n_assets):
            rw = self.returns[self.t - self.window:self.t, i]
            features.extend([self.returns[self.t-1, i], rw[-5:].mean(), rw.mean(), rw.std()])
        features.extend(self.weights.tolist())
        return np.array(features, dtype=np.float32)
    
    def step(self, target_weights):
        turnover = np.abs(target_weights - self.weights).sum()
        tc_cost = self.tc * turnover
        self.weights = target_weights.copy()
        
        asset_returns = self.returns[self.t] if self.t < len(self.returns) else np.zeros(self.n_assets)
        port_return = np.dot(self.weights, asset_returns) - tc_cost
        self.portfolio_value *= (1 + port_return)
        self.values_history.append(self.portfolio_value)
        
        risk = 0.0
        if len(self.values_history) > 5:
            recent = np.diff(self.values_history[-6:]) / np.array(self.values_history[-6:-1])
            risk = np.var(recent)
        
        reward = port_return - self.risk_penalty * risk
        self.t += 1
        done = self.t >= len(self.returns)
        
        if not done:
            w_drift = self.weights * (1 + asset_returns)
            self.weights = w_drift / (w_drift.sum() + 1e-8)
        
        obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, {"portfolio_value": self.portfolio_value, "turnover": turnover}


class PortfolioActorCritic(nn.Module):
    """Actor-critic for portfolio allocation with softmax output."""
    
    def __init__(self, obs_dim, n_assets, hidden=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.weight_head = nn.Linear(hidden, n_assets)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((n_assets,), -1.0))
    
    def forward(self, obs):
        f = self.features(obs)
        logits = self.weight_head(f)
        value = self.value_head(f).squeeze(-1)
        return logits, value
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        
        # Mean weights via softmax
        mean_weights = F.softmax(logits, dim=-1)
        std = self.log_std.exp().expand_as(mean_weights)
        
        # Add exploration noise in logit space
        noise = torch.randn_like(logits) * std
        noisy_logits = logits + noise if action is None else logits
        weights = F.softmax(noisy_logits, dim=-1)
        
        if action is None:
            action = weights
        
        # Approximate log_prob (Gaussian on logits)
        dist = torch.distributions.Normal(logits, std)
        log_prob = dist.log_prob(noisy_logits if action is None else logits).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        return action, log_prob, entropy, value


class PortfolioPPO:
    """PPO for portfolio optimization."""
    
    def __init__(self, env, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip=0.2, epochs=4, hidden=128):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        
        self.network = PortfolioActorCritic(env.obs_dim, env.n_assets, hidden)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def train(self, n_episodes=200, print_interval=20):
        results = []
        
        for ep in range(1, n_episodes + 1):
            obs = self.env.reset()
            
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            done = False
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, lp, _, value = self.network.get_action_and_value(obs_t)
                
                weights = action.numpy().flatten()
                weights = np.maximum(weights, 0)
                weights /= weights.sum() + 1e-8
                
                next_obs, reward, done, info = self.env.step(weights)
                
                states.append(obs)
                actions.append(weights)
                rewards.append(reward)
                log_probs.append(lp.item())
                values.append(value.item())
                dones.append(float(done))
                
                obs = next_obs
            
            # Compute GAE
            advantages, returns = self._compute_gae(rewards, values, dones)
            
            # PPO update
            self._update(states, actions, log_probs, advantages, returns)
            
            final_value = self.env.portfolio_value
            total_return = (final_value - 1.0) * 100
            results.append({"return": total_return, "value": final_value})
            
            if ep % print_interval == 0:
                recent = [r["return"] for r in results[-print_interval:]]
                print(f"Episode {ep:>4d} | Return: {total_return:>7.2f}% | Avg: {np.mean(recent):>7.2f}%")
        
        return results
    
    def _compute_gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
        returns = advantages + np.array(values, dtype=np.float32)
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
    
    def _update(self, states, actions, old_log_probs, advantages, returns):
        s = torch.FloatTensor(np.array(states))
        old_lp = torch.FloatTensor(old_log_probs)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            _, new_lp, entropy, values = self.network.get_action_and_value(s)
            
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()


def generate_synthetic_returns(n_days=500, n_assets=5, seed=42):
    """Generate synthetic multi-asset return data."""
    np.random.seed(seed)
    means = np.random.uniform(0.0002, 0.001, n_assets)
    vols = np.random.uniform(0.01, 0.03, n_assets)
    
    # Correlated returns
    corr = np.eye(n_assets) * 0.5 + 0.5
    L = np.linalg.cholesky(corr)
    
    returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        z = np.random.randn(n_assets)
        returns[t] = means + vols * (L @ z)
    
    return returns


def demo_portfolio_optimization():
    print("=" * 60)
    print("Portfolio Optimization with PPO")
    print("=" * 60)
    
    returns = generate_synthetic_returns(n_days=500, n_assets=5)
    print(f"Data: {returns.shape[0]} days, {returns.shape[1]} assets")
    print(f"Mean daily returns: {returns.mean(axis=0).round(5)}")
    
    env = PortfolioEnv(returns, window=20, transaction_cost=0.001, risk_penalty=0.5)
    agent = PortfolioPPO(env, lr=3e-4, gamma=0.99, epochs=4)
    
    results = agent.train(n_episodes=100, print_interval=20)
    
    # Compare with equal-weight benchmark
    env.reset()
    equal_weights = np.ones(env.n_assets) / env.n_assets
    bm_value = 1.0
    for t in range(env.window, len(returns)):
        bm_value *= (1 + np.dot(equal_weights, returns[t]))
    
    print(f"\nEqual-weight benchmark: {(bm_value - 1) * 100:.2f}%")
    print(f"PPO final episode:     {results[-1]['return']:.2f}%")


if __name__ == "__main__":
    demo_portfolio_optimization()
