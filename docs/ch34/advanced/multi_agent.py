"""
Chapter 34.5.3: Multi-Agent Reinforcement Learning
====================================================
Simple MARL implementations: independent learners,
centralized critic (MADDPG-style), and cooperative MAPPO.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict
from collections import deque


# ---------------------------------------------------------------------------
# Simple Multi-Agent Environment: Cooperative Navigation
# ---------------------------------------------------------------------------

class CooperativeNavigation:
    """
    N agents must navigate to N landmarks. Each agent observes all
    positions. Reward is negative total distance from each agent
    to its nearest unoccupied landmark.
    """
    
    def __init__(self, n_agents=3, world_size=2.0):
        self.n_agents = n_agents
        self.world_size = world_size
        self.n_actions = 5  # stay, up, down, left, right
        self.obs_dim = 2 * n_agents + 2 * n_agents  # agent + landmark positions
        self.reset()
    
    def reset(self):
        self.agent_pos = np.random.uniform(-1, 1, (self.n_agents, 2)).astype(np.float32)
        self.landmark_pos = np.random.uniform(-1, 1, (self.n_agents, 2)).astype(np.float32)
        return self._get_obs()
    
    def _get_obs(self):
        """Each agent sees all positions."""
        obs = np.concatenate([self.agent_pos.flatten(), self.landmark_pos.flatten()])
        return [obs.copy() for _ in range(self.n_agents)]
    
    def step(self, actions):
        """Actions: list of ints for each agent."""
        # Movement
        moves = np.array([[0,0], [0,0.1], [0,-0.1], [-0.1,0], [0.1,0]], dtype=np.float32)
        for i, a in enumerate(actions):
            self.agent_pos[i] += moves[a]
            self.agent_pos[i] = np.clip(self.agent_pos[i], -self.world_size, self.world_size)
        
        # Reward: negative sum of distances to nearest landmark
        total_dist = 0
        for i in range(self.n_agents):
            dists = np.linalg.norm(self.agent_pos[i] - self.landmark_pos, axis=1)
            total_dist += dists.min()
        
        reward = -total_dist
        obs = self._get_obs()
        done = total_dist < 0.1 * self.n_agents
        
        return obs, [reward] * self.n_agents, done


# ---------------------------------------------------------------------------
# Independent Learner
# ---------------------------------------------------------------------------

class IndependentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    
    def forward(self, obs):
        return Categorical(logits=self.net(obs))


class IndependentLearners:
    """Each agent learns independently with its own policy."""
    
    def __init__(self, n_agents, obs_dim, act_dim, lr=1e-3, gamma=0.99):
        self.n_agents = n_agents
        self.gamma = gamma
        self.policies = [IndependentPolicy(obs_dim, act_dim) for _ in range(n_agents)]
        self.optimizers = [optim.Adam(p.parameters(), lr=lr) for p in self.policies]
    
    def get_actions(self, observations):
        actions, log_probs = [], []
        for i, obs in enumerate(observations):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.policies[i](obs_t)
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        return actions, log_probs
    
    def update(self, agent_log_probs, agent_rewards):
        """Update each agent independently."""
        for i in range(self.n_agents):
            returns = []
            G = 0
            for r in reversed(agent_rewards[i]):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            log_probs = torch.stack(agent_log_probs[i])
            loss = -(log_probs * returns).mean()
            
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()


# ---------------------------------------------------------------------------
# MAPPO: Multi-Agent PPO
# ---------------------------------------------------------------------------

class MAPPOAgent:
    """
    Multi-Agent PPO with shared policy and centralized value function.
    
    All agents share the same policy network (parameter sharing),
    and use a centralized value function that sees global state.
    """
    
    def __init__(self, obs_dim, global_state_dim, act_dim, n_agents,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 epochs=4, hidden=64):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        
        # Shared policy (parameter sharing)
        self.policy = IndependentPolicy(obs_dim, act_dim, hidden)
        
        # Centralized value function
        self.value_fn = nn.Sequential(
            nn.Linear(global_state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        
        all_params = list(self.policy.parameters()) + list(self.value_fn.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)
    
    def get_actions(self, observations):
        actions, log_probs = [], []
        for obs in observations:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.policy(obs_t)
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        return actions, log_probs
    
    def get_value(self, global_state):
        state_t = torch.FloatTensor(global_state).unsqueeze(0)
        return self.value_fn(state_t).squeeze()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_cooperative(n_episodes=500, print_interval=100):
    print("=" * 60)
    print("Multi-Agent Cooperative Navigation")
    print("=" * 60)
    
    n_agents = 3
    env = CooperativeNavigation(n_agents=n_agents)
    
    # Independent learners
    agents = IndependentLearners(n_agents, env.obs_dim, env.n_actions, lr=1e-3)
    
    recent = deque(maxlen=100)
    
    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        agent_log_probs = [[] for _ in range(n_agents)]
        agent_rewards = [[] for _ in range(n_agents)]
        total_reward = 0.0
        
        for step in range(50):
            actions, log_probs = agents.get_actions(obs)
            next_obs, rewards, done = env.step(actions)
            
            for i in range(n_agents):
                agent_log_probs[i].append(log_probs[i])
                agent_rewards[i].append(rewards[i])
            
            total_reward += rewards[0]
            obs = next_obs
            
            if done:
                break
        
        agents.update(agent_log_probs, agent_rewards)
        recent.append(total_reward)
        
        if ep % print_interval == 0:
            print(f"Episode {ep:>5d} | Avg(100): {np.mean(recent):>8.2f}")
    
    return list(recent)


def demo_paradigm_comparison():
    """Show the effect of independent vs shared learning."""
    print("\n" + "=" * 60)
    print("MARL Paradigm Comparison")
    print("=" * 60)
    
    print("\nTraining independent learners...")
    rewards = train_cooperative(n_episodes=300, print_interval=300)
    print(f"Independent learners final avg: {np.mean(rewards[-50:]):.2f}")


if __name__ == "__main__":
    train_cooperative()
