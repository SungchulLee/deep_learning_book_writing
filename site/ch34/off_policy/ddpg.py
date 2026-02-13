"""
Chapter 34.4.1: Deep Deterministic Policy Gradient (DDPG)
==========================================================
Complete DDPG implementation with replay buffer, target networks,
and Ornstein-Uhlenbeck exploration noise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
from typing import Tuple


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size experience replay buffer."""
    
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
        )


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Deterministic actor: maps states to actions."""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh(),
        )
    
    def forward(self, obs):
        return self.max_action * self.net(obs)


class Critic(nn.Module):
    """Q-function critic: maps (state, action) to Q-value."""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Noise
# ---------------------------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.astype(np.float32)


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPG:
    """
    Deep Deterministic Policy Gradient agent.
    
    Parameters
    ----------
    env : gym.Env
        Continuous action environment.
    actor_lr : float
        Actor learning rate.
    critic_lr : float
        Critic learning rate.
    gamma : float
        Discount factor.
    tau : float
        Soft update coefficient.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Training batch size.
    noise_sigma : float
        Gaussian exploration noise std.
    warmup_steps : int
        Random actions before training starts.
    """
    
    def __init__(
        self,
        env: gym.Env,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        hidden_dim=256,
        buffer_size=1000000,
        batch_size=256,
        noise_sigma=0.1,
        warmup_steps=25000,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.noise_sigma = noise_sigma
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # Networks
        self.actor = Actor(obs_dim, act_dim, hidden_dim, max_action)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(obs_dim, act_dim, hidden_dim)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)
        self.max_action = max_action
    
    def select_action(self, obs, noise=True):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).numpy().flatten()
        if noise:
            action += np.random.normal(0, self.max_action * self.noise_sigma, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)
    
    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
    
    def update(self):
        if self.buffer.size < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
    
    def train(self, total_steps=200000, print_interval=10000):
        obs, _ = self.env.reset()
        episode_rewards = []
        recent = deque(maxlen=100)
        ep_reward = 0.0
        
        for step in range(1, total_steps + 1):
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.buffer.add(obs, action, reward, next_obs, done)
            ep_reward += reward
            
            if step >= self.warmup_steps:
                self.update()
            
            obs = next_obs
            if done:
                episode_rewards.append(ep_reward)
                recent.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()
            
            if step % print_interval == 0 and len(recent) > 0:
                print(f"Step {step:>8d} | Avg(100): {np.mean(recent):>8.1f} | Episodes: {len(episode_rewards)}")
        
        return episode_rewards


def demo_ddpg():
    print("=" * 60)
    print("DDPG on Pendulum-v1")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    agent = DDPG(
        env=env, actor_lr=1e-4, critic_lr=1e-3,
        gamma=0.99, tau=0.005, batch_size=256,
        noise_sigma=0.1, warmup_steps=10000,
    )
    rewards = agent.train(total_steps=100000, print_interval=10000)
    env.close()
    
    if len(rewards) >= 50:
        print(f"\nFinal avg (last 50): {np.mean(rewards[-50:]):.1f}")


if __name__ == "__main__":
    demo_ddpg()
