"""
Chapter 34.1.3: REINFORCE Algorithm
=====================================
Complete REINFORCE implementation with variants:
- Vanilla REINFORCE
- REINFORCE with reward-to-go
- REINFORCE with return normalization
- Training on CartPole and continuous control
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Optional
from collections import deque


# ---------------------------------------------------------------------------
# Policy Networks
# ---------------------------------------------------------------------------

class DiscretePolicyNetwork(nn.Module):
    """Policy network for discrete actions."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous actions."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs: torch.Tensor) -> Normal:
        features = self.net(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


# ---------------------------------------------------------------------------
# REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCE:
    """
    REINFORCE policy gradient agent.
    
    Parameters
    ----------
    env : gym.Env
        Gymnasium environment.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    hidden_dim : int
        Hidden layer size.
    use_reward_to_go : bool
        If True, use causality (reward-to-go). Otherwise total return.
    normalize_returns : bool
        If True, normalize returns to zero mean unit variance.
    entropy_coef : float
        Coefficient for entropy bonus (encourages exploration).
    """
    
    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        use_reward_to_go: bool = True,
        normalize_returns: bool = True,
        entropy_coef: float = 0.01,
    ):
        self.env = env
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.normalize_returns = normalize_returns
        self.entropy_coef = entropy_coef
        
        # Determine action space type
        obs_dim = env.observation_space.shape[0]
        self.continuous = isinstance(env.action_space, gym.spaces.Box)
        
        if self.continuous:
            act_dim = env.action_space.shape[0]
            self.policy = ContinuousPolicyNetwork(obs_dim, act_dim, hidden_dim)
        else:
            act_dim = env.action_space.n
            self.policy = DiscretePolicyNetwork(obs_dim, act_dim, hidden_dim)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select action from the current policy."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = self.policy(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
            return action.detach().numpy().flatten(), log_prob, entropy
        else:
            return action.item(), log_prob, entropy
    
    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns."""
        if self.use_reward_to_go:
            # Reward-to-go: G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
        else:
            # Total trajectory return: all timesteps weighted by R(τ)
            R = sum(self.gamma ** t * r for t, r in enumerate(rewards))
            returns = torch.full((len(rewards),), R, dtype=torch.float32)
        
        if self.normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def collect_episode(self) -> Tuple[List, List, List, float]:
        """Collect a complete episode."""
        obs, _ = self.env.reset()
        log_probs, entropies, rewards = [], [], []
        
        done = False
        while not done:
            action, log_prob, entropy = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            
            obs = next_obs
            done = terminated or truncated
        
        return log_probs, entropies, rewards, sum(rewards)
    
    def update(self, log_probs: List, entropies: List, rewards: List):
        """Perform a single REINFORCE update."""
        returns = self.compute_returns(rewards)
        
        # Stack log probs and entropies
        log_probs_t = torch.stack(log_probs).squeeze()
        entropies_t = torch.stack(entropies).squeeze()
        
        # Policy gradient loss: -E[log π(a|s) · G_t]
        policy_loss = -(log_probs_t * returns).mean()
        
        # Entropy bonus for exploration
        entropy_loss = -entropies_t.mean()
        
        # Total loss
        loss = policy_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return policy_loss.item(), entropies_t.mean().item()
    
    def train(
        self,
        n_episodes: int = 1000,
        print_interval: int = 100,
        solved_reward: Optional[float] = None,
    ) -> List[float]:
        """
        Train the agent using REINFORCE.
        
        Parameters
        ----------
        n_episodes : int
            Number of training episodes.
        print_interval : int
            How often to print progress.
        solved_reward : float, optional
            If provided, stop training when average reward exceeds this.
        
        Returns
        -------
        episode_rewards : list of float
            Reward history.
        """
        episode_rewards = []
        recent_rewards = deque(maxlen=100)
        
        for episode in range(1, n_episodes + 1):
            log_probs, entropies, rewards, total_reward = self.collect_episode()
            policy_loss, avg_entropy = self.update(log_probs, entropies, rewards)
            
            episode_rewards.append(total_reward)
            recent_rewards.append(total_reward)
            avg_reward = np.mean(recent_rewards)
            
            if episode % print_interval == 0:
                print(
                    f"Episode {episode:>5d} | "
                    f"Reward: {total_reward:>7.1f} | "
                    f"Avg(100): {avg_reward:>7.1f} | "
                    f"Loss: {policy_loss:>8.4f} | "
                    f"Entropy: {avg_entropy:>6.3f}"
                )
            
            if solved_reward is not None and avg_reward >= solved_reward:
                print(f"\nSolved in {episode} episodes! Avg reward: {avg_reward:.1f}")
                break
        
        return episode_rewards


# ---------------------------------------------------------------------------
# Batch REINFORCE (multiple episodes per update)
# ---------------------------------------------------------------------------

class BatchREINFORCE(REINFORCE):
    """
    REINFORCE with batch updates for reduced variance.
    
    Collects multiple episodes before performing a gradient update,
    averaging gradients across the batch.
    """
    
    def __init__(self, *args, batch_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
    
    def train(
        self,
        n_episodes: int = 1000,
        print_interval: int = 100,
        solved_reward: Optional[float] = None,
    ) -> List[float]:
        episode_rewards = []
        recent_rewards = deque(maxlen=100)
        
        episode = 0
        while episode < n_episodes:
            # Collect batch of episodes
            batch_log_probs = []
            batch_entropies = []
            batch_returns = []
            batch_rewards = []
            
            for _ in range(self.batch_size):
                log_probs, entropies, rewards, total_reward = self.collect_episode()
                returns = self.compute_returns(rewards)
                
                batch_log_probs.extend(log_probs)
                batch_entropies.extend(entropies)
                batch_returns.append(returns)
                batch_rewards.append(total_reward)
                
                episode += 1
                episode_rewards.append(total_reward)
                recent_rewards.append(total_reward)
            
            # Concatenate all transitions
            all_log_probs = torch.stack(batch_log_probs).squeeze()
            all_entropies = torch.stack(batch_entropies).squeeze()
            all_returns = torch.cat(batch_returns)
            
            # Normalize returns across entire batch
            if self.normalize_returns:
                all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
            
            # Single gradient update
            policy_loss = -(all_log_probs * all_returns).mean()
            entropy_loss = -all_entropies.mean()
            loss = policy_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            avg_reward = np.mean(recent_rewards)
            if episode % print_interval < self.batch_size:
                print(
                    f"Episode {episode:>5d} | "
                    f"Batch Avg: {np.mean(batch_rewards):>7.1f} | "
                    f"Avg(100): {avg_reward:>7.1f}"
                )
            
            if solved_reward is not None and avg_reward >= solved_reward:
                print(f"\nSolved in {episode} episodes! Avg reward: {avg_reward:.1f}")
                break
        
        return episode_rewards


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def train_cartpole():
    """Train REINFORCE on CartPole-v1."""
    print("=" * 60)
    print("REINFORCE on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    
    agent = REINFORCE(
        env=env,
        lr=1e-3,
        gamma=0.99,
        hidden_dim=128,
        use_reward_to_go=True,
        normalize_returns=True,
        entropy_coef=0.01,
    )
    
    rewards = agent.train(n_episodes=1000, print_interval=100, solved_reward=475.0)
    env.close()
    return rewards


def train_cartpole_batch():
    """Train Batch REINFORCE on CartPole-v1."""
    print("\n" + "=" * 60)
    print("Batch REINFORCE on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    
    agent = BatchREINFORCE(
        env=env,
        lr=1e-3,
        gamma=0.99,
        hidden_dim=128,
        use_reward_to_go=True,
        normalize_returns=True,
        entropy_coef=0.01,
        batch_size=10,
    )
    
    rewards = agent.train(n_episodes=1000, print_interval=100, solved_reward=475.0)
    env.close()
    return rewards


def compare_variants():
    """Compare REINFORCE variants on CartPole."""
    print("\n" + "=" * 60)
    print("Comparing REINFORCE Variants")
    print("=" * 60)
    
    variants = {
        "Total Return": {"use_reward_to_go": False, "normalize_returns": False},
        "Reward-to-Go": {"use_reward_to_go": True, "normalize_returns": False},
        "RTG + Normalize": {"use_reward_to_go": True, "normalize_returns": True},
    }
    
    n_episodes = 500
    n_trials = 3
    
    for name, kwargs in variants.items():
        trial_rewards = []
        for trial in range(n_trials):
            env = gym.make("CartPole-v1")
            torch.manual_seed(trial)
            np.random.seed(trial)
            
            agent = REINFORCE(
                env=env, lr=1e-3, gamma=0.99, hidden_dim=128,
                entropy_coef=0.01, **kwargs
            )
            
            rewards = agent.train(n_episodes=n_episodes, print_interval=n_episodes + 1)
            trial_rewards.append(np.mean(rewards[-100:]))
            env.close()
        
        avg = np.mean(trial_rewards)
        std = np.std(trial_rewards)
        print(f"{name:<20}: Final Avg Reward = {avg:.1f} ± {std:.1f}")


if __name__ == "__main__":
    train_cartpole()
    train_cartpole_batch()
    compare_variants()
