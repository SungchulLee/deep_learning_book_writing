"""
Chapter 34.2.1: Actor-Critic Fundamentals
==========================================
Implementation of basic actor-critic algorithms including
online (step-by-step) and batch actor-critic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List
from collections import deque


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic network.
    
    Architecture:
        shared features → actor head (logits)
                        → critic head (value)
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def forward(self, obs):
        features = self.features(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_value(self, obs):
        _, value = self.forward(obs)
        return value


# ---------------------------------------------------------------------------
# Online Actor-Critic (updates every step)
# ---------------------------------------------------------------------------

class OnlineActorCritic:
    """
    Online (step-by-step) actor-critic using TD(0) advantage.
    
    Updates both actor and critic at every environment step.
    The TD error δ = r + γV(s') - V(s) serves as a one-step
    advantage estimate.
    """
    
    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.ac = ActorCritic(obs_dim, act_dim, hidden_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
    
    def train(self, n_episodes: int = 1000, print_interval: int = 100) -> List[float]:
        episode_rewards = []
        recent_rewards = deque(maxlen=100)
        
        for episode in range(1, n_episodes + 1):
            obs, _ = self.env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                
                # Get action and value
                action, log_prob, entropy, value = self.ac.get_action_and_value(obs_t)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                total_reward += reward
                
                # Compute TD error (one-step advantage)
                with torch.no_grad():
                    next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0)
                    next_value = self.ac.get_value(next_obs_t) if not done else torch.tensor(0.0)
                
                td_target = reward + self.gamma * next_value * (1 - float(done))
                td_error = td_target - value
                
                # Losses
                actor_loss = -(log_prob * td_error.detach())
                critic_loss = td_error.pow(2)
                entropy_loss = -entropy
                
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                obs = next_obs
            
            episode_rewards.append(total_reward)
            recent_rewards.append(total_reward)
            
            if episode % print_interval == 0:
                print(
                    f"Episode {episode:>5d} | "
                    f"Reward: {total_reward:>7.1f} | "
                    f"Avg(100): {np.mean(recent_rewards):>7.1f}"
                )
        
        return episode_rewards


# ---------------------------------------------------------------------------
# Batch Actor-Critic (n-step updates)
# ---------------------------------------------------------------------------

class BatchActorCritic:
    """
    Batch actor-critic with n-step returns.
    
    Collects rollouts of fixed length, computes n-step advantages,
    and performs batched gradient updates.
    
    Parameters
    ----------
    env : gym.Env
        Environment.
    n_steps : int
        Number of steps to collect before each update.
    n_step_return : int
        Number of steps for n-step return targets.
    """
    
    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 128,
        n_step_return: int = 5,
        hidden_dim: int = 128,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.env = env
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_step_return = n_step_return
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.ac = ActorCritic(obs_dim, act_dim, hidden_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
    
    def collect_rollout(self, obs):
        """Collect n_steps of experience."""
        states, actions, rewards, dones, log_probs, values, entropies = (
            [], [], [], [], [], [], []
        )
        
        for _ in range(self.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, entropy, value = self.ac.get_action_and_value(obs_t)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            entropies.append(entropy.item())
            
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
        
        return obs, states, actions, rewards, dones, log_probs, values
    
    def compute_nstep_returns(self, rewards, dones, values, last_obs):
        """Compute n-step return targets."""
        n = self.n_step_return
        T = len(rewards)
        
        with torch.no_grad():
            last_value = self.ac.get_value(
                torch.FloatTensor(last_obs).unsqueeze(0)
            ).item()
        
        # Extend values with bootstrapped last value
        extended_values = values + [last_value]
        
        returns = np.zeros(T)
        for t in range(T):
            G = 0.0
            for k in range(min(n, T - t)):
                G += self.gamma ** k * rewards[t + k]
                if dones[t + k]:
                    break
            else:
                # Bootstrap if no terminal state within n steps
                if t + n < T:
                    G += self.gamma ** n * extended_values[t + n]
                else:
                    G += self.gamma ** (T - t) * last_value
            returns[t] = G
        
        return torch.FloatTensor(returns)
    
    def update(self, states, actions, returns):
        """Perform batched actor-critic update."""
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.tensor(actions, dtype=torch.long)
        
        # Forward pass
        _, log_probs, entropies, values = self.ac.get_action_and_value(states_t, actions_t)
        
        # Advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropies.mean().item()
    
    def train(self, total_steps: int = 100000, print_interval: int = 5000) -> List[float]:
        obs, _ = self.env.reset()
        episode_rewards = []
        current_reward = 0.0
        recent_rewards = deque(maxlen=100)
        steps = 0
        
        while steps < total_steps:
            # Collect rollout
            next_obs, states, actions, rewards, dones, log_probs, values = \
                self.collect_rollout(obs)
            
            # Track episode rewards
            for r, d in zip(rewards, dones):
                current_reward += r
                if d:
                    episode_rewards.append(current_reward)
                    recent_rewards.append(current_reward)
                    current_reward = 0.0
            
            # Compute returns and update
            returns = self.compute_nstep_returns(rewards, dones, values, next_obs)
            actor_loss, critic_loss, avg_entropy = self.update(states, actions, returns)
            
            obs = next_obs
            steps += self.n_steps
            
            if steps % print_interval < self.n_steps and len(recent_rewards) > 0:
                print(
                    f"Step {steps:>7d} | "
                    f"Avg(100): {np.mean(recent_rewards):>7.1f} | "
                    f"Actor: {actor_loss:>7.4f} | "
                    f"Critic: {critic_loss:>7.4f} | "
                    f"H: {avg_entropy:>5.3f}"
                )
        
        return episode_rewards


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_online_actor_critic():
    """Train online actor-critic on CartPole."""
    print("=" * 60)
    print("Online Actor-Critic on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    agent = OnlineActorCritic(env=env, lr=1e-3, gamma=0.99, entropy_coef=0.01)
    rewards = agent.train(n_episodes=1000, print_interval=200)
    env.close()
    
    print(f"\nFinal avg reward (last 100): {np.mean(rewards[-100:]):.1f}")
    return rewards


def demo_batch_actor_critic():
    """Train batch actor-critic on CartPole."""
    print("\n" + "=" * 60)
    print("Batch Actor-Critic on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    agent = BatchActorCritic(
        env=env, lr=3e-4, gamma=0.99,
        n_steps=128, n_step_return=5,
        entropy_coef=0.01, value_coef=0.5,
    )
    rewards = agent.train(total_steps=100000, print_interval=10000)
    env.close()
    
    if len(rewards) >= 100:
        print(f"\nFinal avg reward (last 100): {np.mean(rewards[-100:]):.1f}")
    return rewards


def demo_architecture_comparison():
    """Compare shared vs separate network architectures."""
    print("\n" + "=" * 60)
    print("Architecture Comparison")
    print("=" * 60)
    
    obs_dim, act_dim = 4, 2
    
    # Shared backbone
    shared = ActorCritic(obs_dim, act_dim, hidden_dim=128)
    shared_params = sum(p.numel() for p in shared.parameters())
    
    # Separate networks (simulated)
    actor = nn.Sequential(
        nn.Linear(obs_dim, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, act_dim),
    )
    critic = nn.Sequential(
        nn.Linear(obs_dim, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )
    separate_params = sum(p.numel() for p in actor.parameters()) + \
                      sum(p.numel() for p in critic.parameters())
    
    print(f"Shared backbone parameters:   {shared_params:,}")
    print(f"Separate networks parameters: {separate_params:,}")
    print(f"Parameter ratio (sep/shared): {separate_params / shared_params:.2f}x")
    
    # Test forward pass
    obs = torch.randn(32, obs_dim)
    action, log_prob, entropy, value = shared.get_action_and_value(obs)
    print(f"\nBatch forward pass (shared):")
    print(f"  Actions: {action.shape}")
    print(f"  Log probs: {log_prob.shape}")
    print(f"  Entropy: {entropy.mean().item():.4f}")
    print(f"  Values: {value.shape}, mean={value.mean().item():.4f}")


if __name__ == "__main__":
    demo_online_actor_critic()
    demo_batch_actor_critic()
    demo_architecture_comparison()
