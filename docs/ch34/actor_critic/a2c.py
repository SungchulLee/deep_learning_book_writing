"""
Chapter 34.2.2: Advantage Actor-Critic (A2C)
=============================================
Full A2C implementation with vectorized environments,
n-step returns, and combined actor-critic training.
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


class A2CNetwork(nn.Module):
    """Shared actor-critic network for A2C."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def forward(self, obs):
        features = self.features(obs)
        return self.actor(features), self.critic(features).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_value(self, obs):
        return self.forward(obs)[1]


# ---------------------------------------------------------------------------
# Vectorized Environment Wrapper
# ---------------------------------------------------------------------------

class VecEnv:
    """
    Simple synchronous vectorized environment.
    
    Runs N environments in parallel, stepping them synchronously.
    """
    
    def __init__(self, env_id: str, n_envs: int, seed: int = 0):
        self.envs = [gym.make(env_id) for _ in range(n_envs)]
        self.n_envs = n_envs
        for i, env in enumerate(self.envs):
            env.reset(seed=seed + i)
    
    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.array(obs_list)
    
    def step(self, actions):
        obs_list, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Auto-reset on done
            if done:
                info["terminal_obs"] = obs
                info["episode_reward"] = info.get("episode", {}).get("r", None)
                obs, _ = env.reset()
            
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.array(obs_list), np.array(rewards), np.array(dones), infos
    
    @property
    def observation_space(self):
        return self.envs[0].observation_space
    
    @property
    def action_space(self):
        return self.envs[0].action_space
    
    def close(self):
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# A2C Agent
# ---------------------------------------------------------------------------

class A2C:
    """
    Advantage Actor-Critic (A2C) with synchronous vectorized environments.
    
    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    n_envs : int
        Number of parallel environments.
    n_steps : int
        Rollout length per environment.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    value_coef : float
        Value loss coefficient.
    entropy_coef : float
        Entropy bonus coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    use_gae : bool
        Whether to use GAE (True) or n-step returns (False).
    gae_lambda : float
        GAE lambda parameter.
    """
    
    def __init__(
        self,
        env_id: str = "CartPole-v1",
        n_envs: int = 8,
        n_steps: int = 5,
        lr: float = 7e-4,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_gae: bool = False,
        gae_lambda: float = 0.95,
        seed: int = 0,
    ):
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        
        # Setup environments
        self.envs = VecEnv(env_id, n_envs, seed=seed)
        obs_dim = self.envs.observation_space.shape[0]
        act_dim = self.envs.action_space.n
        
        # Setup network
        self.network = A2CNetwork(obs_dim, act_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
    
    def collect_rollout(self, obs):
        """
        Collect n_steps of experience from all environments.
        
        Returns
        -------
        obs : ndarray (next observation for continuing)
        rollout : dict of tensors
        episode_rewards : list of completed episode rewards
        """
        # Storage
        mb_obs = np.zeros((self.n_steps, self.n_envs) + self.envs.observation_space.shape)
        mb_actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        mb_rewards = np.zeros((self.n_steps, self.n_envs))
        mb_dones = np.zeros((self.n_steps, self.n_envs))
        mb_values = np.zeros((self.n_steps, self.n_envs))
        mb_log_probs = np.zeros((self.n_steps, self.n_envs))
        
        episode_rewards = []
        
        for step in range(self.n_steps):
            mb_obs[step] = obs
            
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs)
                action, log_prob, _, value = self.network.get_action_and_value(obs_t)
            
            mb_actions[step] = action.numpy()
            mb_values[step] = value.numpy()
            mb_log_probs[step] = log_prob.numpy()
            
            obs, rewards, dones, infos = self.envs.step(action.numpy())
            mb_rewards[step] = rewards
            mb_dones[step] = dones
            
            # Track episode completions
            for i, done in enumerate(dones):
                if done:
                    # Compute episode reward from stored rewards
                    ep_reward = sum(mb_rewards[s, i] for s in range(step + 1))
                    episode_rewards.append(ep_reward)
        
        rollout = {
            "obs": torch.FloatTensor(mb_obs.reshape(-1, *self.envs.observation_space.shape)),
            "actions": torch.LongTensor(mb_actions.reshape(-1)),
            "rewards": mb_rewards,
            "dones": mb_dones,
            "values": mb_values,
            "log_probs": mb_log_probs,
        }
        
        return obs, rollout, episode_rewards
    
    def compute_returns_and_advantages(self, rollout, last_obs):
        """Compute return targets and advantages."""
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]
        
        with torch.no_grad():
            last_value = self.network.get_value(
                torch.FloatTensor(last_obs)
            ).numpy()
        
        if self.use_gae:
            # GAE computation
            advantages = np.zeros_like(rewards)
            last_gae = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae = \
                    delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
            returns = advantages + values
        else:
            # N-step returns (standard A2C)
            returns = np.zeros_like(rewards)
            R = last_value
            for t in reversed(range(self.n_steps)):
                R = rewards[t] + self.gamma * R * (1.0 - dones[t])
                returns[t] = R
            advantages = returns - values
        
        returns = torch.FloatTensor(returns.reshape(-1))
        advantages = torch.FloatTensor(advantages.reshape(-1))
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, rollout, returns, advantages):
        """Perform A2C update."""
        obs = rollout["obs"]
        actions = rollout["actions"]
        
        # Forward pass
        _, new_log_probs, entropy, new_values = \
            self.network.get_action_and_value(obs, actions)
        
        # Policy loss
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(new_values, returns)
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
        }
    
    def train(self, total_steps: int = 200000, print_interval: int = 10000) -> List[float]:
        """Train A2C agent."""
        obs = self.envs.reset()
        all_rewards = []
        recent_rewards = deque(maxlen=100)
        steps = 0
        updates = 0
        
        while steps < total_steps:
            obs, rollout, episode_rewards = self.collect_rollout(obs)
            returns, advantages = self.compute_returns_and_advantages(rollout, obs)
            metrics = self.update(rollout, returns, advantages)
            
            steps += self.n_steps * self.n_envs
            updates += 1
            
            for r in episode_rewards:
                all_rewards.append(r)
                recent_rewards.append(r)
            
            if steps % print_interval < self.n_steps * self.n_envs and len(recent_rewards) > 0:
                print(
                    f"Step {steps:>8d} | "
                    f"Avg(100): {np.mean(recent_rewards):>7.1f} | "
                    f"Policy: {metrics['policy_loss']:>7.4f} | "
                    f"Value: {metrics['value_loss']:>7.4f} | "
                    f"H: {metrics['entropy']:>5.3f}"
                )
        
        self.envs.close()
        return all_rewards
    
    def evaluate(self, env_id: str, n_episodes: int = 10) -> float:
        """Evaluate the trained policy."""
        env = gym.make(env_id)
        rewards = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    logits, _ = self.network(obs_t)
                    action = logits.argmax(dim=-1).item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            rewards.append(total_reward)
        
        env.close()
        return np.mean(rewards)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_a2c():
    """Train A2C on CartPole."""
    print("=" * 60)
    print("A2C on CartPole-v1")
    print("=" * 60)
    
    agent = A2C(
        env_id="CartPole-v1",
        n_envs=8,
        n_steps=5,
        lr=7e-4,
        gamma=0.99,
        hidden_dim=64,
        value_coef=0.5,
        entropy_coef=0.01,
        use_gae=False,
        seed=42,
    )
    
    rewards = agent.train(total_steps=200000, print_interval=20000)
    
    # Evaluate
    eval_reward = agent.evaluate("CartPole-v1", n_episodes=20)
    print(f"\nEvaluation reward (20 episodes): {eval_reward:.1f}")
    
    return rewards


def demo_a2c_with_gae():
    """Train A2C with GAE on CartPole."""
    print("\n" + "=" * 60)
    print("A2C with GAE on CartPole-v1")
    print("=" * 60)
    
    agent = A2C(
        env_id="CartPole-v1",
        n_envs=8,
        n_steps=128,
        lr=2.5e-4,
        gamma=0.99,
        hidden_dim=64,
        value_coef=0.5,
        entropy_coef=0.01,
        use_gae=True,
        gae_lambda=0.95,
        seed=42,
    )
    
    rewards = agent.train(total_steps=200000, print_interval=20000)
    
    eval_reward = agent.evaluate("CartPole-v1", n_episodes=20)
    print(f"\nEvaluation reward (20 episodes): {eval_reward:.1f}")
    
    return rewards


if __name__ == "__main__":
    demo_a2c()
    demo_a2c_with_gae()
