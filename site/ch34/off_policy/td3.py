"""
Chapter 34.4.2: Twin Delayed DDPG (TD3)
=========================================
TD3 implementation with twin critics, delayed policy updates,
and target policy smoothing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        self.s[self.idx], self.a[self.idx] = s, a
        self.r[self.idx], self.s2[self.idx], self.d[self.idx] = r, s2, float(d)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.s[idx]), torch.FloatTensor(self.a[idx]),
                torch.FloatTensor(self.r[idx]), torch.FloatTensor(self.s2[idx]),
                torch.FloatTensor(self.d[idx]))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
    
    def forward(self, obs):
        return self.max_action * self.net(obs)


class TwinCritic(nn.Module):
    """Two independent Q-networks for clipped double-Q learning."""
    
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)
    
    def q1_forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1)


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient.
    
    Three key improvements over DDPG:
    1. Twin critics (clipped double-Q)
    2. Delayed policy updates
    3. Target policy smoothing
    """
    
    def __init__(
        self,
        env: gym.Env,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
        explore_noise=0.1,
        hidden_dim=256,
        buffer_size=1000000,
        batch_size=256,
        warmup_steps=25000,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.explore_noise = explore_noise
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        self.actor = Actor(obs_dim, act_dim, hidden_dim, self.max_action)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = TwinCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target = TwinCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)
        self.total_updates = 0
    
    def select_action(self, obs, noise=True):
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(obs).unsqueeze(0)).numpy().flatten()
        if noise:
            action += np.random.normal(0, self.max_action * self.explore_noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)
    
    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
    
    def update(self):
        if self.buffer.size < self.batch_size:
            return {}
        
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        self.total_updates += 1
        
        # === Critic Update ===
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(a) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(s2) + noise).clamp(-self.max_action, self.max_action)
            
            # Clipped double-Q: take the minimum
            target_q1, target_q2 = self.critic_target(s2, next_action)
            target_q = r + self.gamma * (1 - d) * torch.min(target_q1, target_q2)
        
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        metrics = {"critic_loss": critic_loss.item()}
        
        # === Delayed Actor Update ===
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(s, self.actor(s)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
            
            metrics["actor_loss"] = actor_loss.item()
        
        return metrics
    
    def train(self, total_steps=200000, print_interval=10000):
        obs, _ = self.env.reset()
        ep_rewards, recent = [], deque(maxlen=100)
        ep_r = 0.0
        
        for step in range(1, total_steps + 1):
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs)
            
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.buffer.add(obs, action, reward, next_obs, done)
            ep_r += reward
            
            if step >= self.warmup_steps:
                self.update()
            
            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                recent.append(ep_r)
                ep_r = 0.0
                obs, _ = self.env.reset()
            
            if step % print_interval == 0 and recent:
                print(f"Step {step:>8d} | Avg(100): {np.mean(recent):>8.1f}")
        
        return ep_rewards


def demo_td3():
    print("=" * 60)
    print("TD3 on Pendulum-v1")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    agent = TD3(env=env, warmup_steps=5000, batch_size=256)
    rewards = agent.train(total_steps=100000, print_interval=10000)
    env.close()
    
    if len(rewards) >= 50:
        print(f"\nFinal avg (last 50): {np.mean(rewards[-50:]):.1f}")


if __name__ == "__main__":
    demo_td3()
