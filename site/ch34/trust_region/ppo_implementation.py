"""
Chapter 34.3.4: PPO Implementation
====================================
Complete PPO implementation following best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from typing import List
from collections import deque


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class VecEnv:
    def __init__(self, env_id, n_envs, seed=0):
        self.envs = [gym.make(env_id) for _ in range(n_envs)]
        self.n_envs = n_envs
        for i, env in enumerate(self.envs):
            env.reset(seed=seed + i)
    
    @property
    def obs_shape(self): return self.envs[0].observation_space.shape
    
    @property
    def act_space(self): return self.envs[0].action_space
    
    def reset(self):
        return np.array([env.reset()[0] for env in self.envs])
    
    def step(self, actions):
        obs_list, rewards, dones = [], [], []
        for env, a in zip(self.envs, actions):
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if done:
                o = env.reset()[0]
            obs_list.append(o)
            rewards.append(r)
            dones.append(float(done))
        return np.array(obs_list), np.array(rewards), np.array(dones)
    
    def close(self):
        for env in self.envs:
            env.close()


class PPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, continuous=False):
        super().__init__()
        self.continuous = continuous
        self.features = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
        )
        if continuous:
            self.actor_mean = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.actor = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
    
    def get_value(self, obs):
        return self.critic(self.features(obs)).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        features = self.features(obs)
        value = self.critic(features).squeeze(-1)
        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            dist = Categorical(logits=self.actor(features))
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action, log_prob, entropy, value


class PPO:
    """Complete PPO-Clip implementation."""
    
    def __init__(
        self,
        env_id="CartPole-v1",
        n_envs=8,
        n_steps=128,
        n_epochs=4,
        n_minibatches=4,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        anneal_lr=True,
        seed=42,
    ):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.initial_lr = lr
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.envs = VecEnv(env_id, n_envs, seed=seed)
        obs_dim = self.envs.obs_shape[0]
        
        continuous = isinstance(self.envs.act_space, gym.spaces.Box)
        act_dim = self.envs.act_space.shape[0] if continuous else self.envs.act_space.n
        
        self.network = PPONetwork(obs_dim, act_dim, continuous=continuous)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.batch_size = n_envs * n_steps
        self.minibatch_size = self.batch_size // n_minibatches
    
    def compute_gae(self, rewards, values, dones, last_value):
        """Compute GAE advantages and returns."""
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        lastgae = np.zeros(N, dtype=np.float32)
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - dones[t]
                next_values = last_value
            else:
                next_nonterminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_nonterminal - values[t]
            advantages[t] = lastgae = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgae
        
        returns = advantages + values
        return advantages, returns
    
    def train(self, total_timesteps=200000, print_interval=10000) -> dict:
        """
        Train PPO agent.
        
        Returns dict with training metrics.
        """
        obs = self.envs.reset()
        num_updates = total_timesteps // self.batch_size
        
        # Tracking
        episode_rewards = []
        recent_rewards = deque(maxlen=100)
        current_rewards = np.zeros(self.n_envs)
        global_step = 0
        
        # Storage
        mb_obs = np.zeros((self.n_steps, self.n_envs) + self.envs.obs_shape, dtype=np.float32)
        mb_actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64 
                              if isinstance(self.envs.act_space, gym.spaces.Discrete) else np.float32)
        mb_logprobs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        mb_rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        mb_dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        mb_values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        
        for update in range(1, num_updates + 1):
            # Learning rate annealing
            if self.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lr = self.initial_lr * frac
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
            
            # === Rollout Phase ===
            for step in range(self.n_steps):
                global_step += self.n_envs
                mb_obs[step] = obs
                
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs)
                    action, logprob, _, value = self.network.get_action_and_value(obs_t)
                
                if isinstance(self.envs.act_space, gym.spaces.Discrete):
                    mb_actions[step] = action.numpy()
                    step_actions = action.numpy()
                else:
                    mb_actions[step] = action.numpy()
                    step_actions = action.numpy()
                
                mb_logprobs[step] = logprob.numpy()
                mb_values[step] = value.numpy()
                
                obs, rewards, dones = self.envs.step(step_actions)
                mb_rewards[step] = rewards
                mb_dones[step] = dones
                
                # Track episodes
                current_rewards += rewards
                for i in range(self.n_envs):
                    if dones[i]:
                        episode_rewards.append(current_rewards[i])
                        recent_rewards.append(current_rewards[i])
                        current_rewards[i] = 0.0
            
            # === Compute GAE ===
            with torch.no_grad():
                last_value = self.network.get_value(torch.FloatTensor(obs)).numpy()
            
            advantages, returns = self.compute_gae(
                mb_rewards, mb_values, mb_dones, last_value
            )
            
            # Flatten batch
            b_obs = torch.FloatTensor(mb_obs.reshape(-1, *self.envs.obs_shape))
            b_actions = torch.LongTensor(mb_actions.reshape(-1)) if isinstance(
                self.envs.act_space, gym.spaces.Discrete
            ) else torch.FloatTensor(mb_actions.reshape(-1, self.envs.act_space.shape[0]))
            b_logprobs = torch.FloatTensor(mb_logprobs.reshape(-1))
            b_advantages = torch.FloatTensor(advantages.reshape(-1))
            b_returns = torch.FloatTensor(returns.reshape(-1))
            b_values = torch.FloatTensor(mb_values.reshape(-1))
            
            # === Optimization Phase ===
            clipfracs = []
            
            for epoch in range(self.n_epochs):
                # Random permutation for minibatches
                indices = np.random.permutation(self.batch_size)
                
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = indices[start:end]
                    
                    _, new_logprob, entropy, new_value = self.network.get_action_and_value(
                        b_obs[mb_idx], b_actions[mb_idx]
                    )
                    
                    # Ratio
                    logratio = new_logprob - b_logprobs[mb_idx]
                    ratio = logratio.exp()
                    
                    # Debug: approx KL
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        )
                    
                    # Normalize advantages per minibatch
                    mb_adv = b_advantages[mb_idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    
                    # Policy loss (clipped surrogate)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    if self.clip_vloss:
                        v_loss_unclipped = (new_value - b_returns[mb_idx]).pow(2)
                        v_clipped = b_values[mb_idx] + torch.clamp(
                            new_value - b_values[mb_idx],
                            -self.clip_coef, self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_idx]).pow(2)
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * (new_value - b_returns[mb_idx]).pow(2).mean()
                    
                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    # Total loss
                    loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            
            # === Logging ===
            if global_step % print_interval < self.batch_size and len(recent_rewards) > 0:
                print(
                    f"Step {global_step:>8d} | "
                    f"Avg(100): {np.mean(recent_rewards):>7.1f} | "
                    f"Policy: {pg_loss.item():>7.4f} | "
                    f"Value: {v_loss.item():>7.4f} | "
                    f"Entropy: {entropy_loss.item():>5.3f} | "
                    f"ClipFrac: {np.mean(clipfracs):>5.3f} | "
                    f"KL: {approx_kl.item():>7.5f}"
                )
        
        self.envs.close()
        return {
            "episode_rewards": episode_rewards,
            "final_avg": np.mean(list(recent_rewards)) if recent_rewards else 0,
        }
    
    def evaluate(self, env_id: str, n_episodes: int = 10) -> float:
        env = gym.make(env_id)
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_r = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _, _ = self.network.get_action_and_value(obs_t)
                if isinstance(env.action_space, gym.spaces.Discrete):
                    obs, r, term, trunc, _ = env.step(action.item())
                else:
                    obs, r, term, trunc, _ = env.step(action.numpy().flatten())
                total_r += r
                done = term or trunc
            rewards.append(total_r)
        env.close()
        return np.mean(rewards)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_ppo_cartpole():
    print("=" * 60)
    print("PPO on CartPole-v1")
    print("=" * 60)
    
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=8, n_steps=128, n_epochs=4, n_minibatches=4,
        lr=2.5e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
        ent_coef=0.01, vf_coef=0.5, anneal_lr=True, seed=42,
    )
    
    result = agent.train(total_timesteps=200000, print_interval=25000)
    eval_reward = agent.evaluate("CartPole-v1", n_episodes=20)
    
    print(f"\nTraining avg (last 100): {result['final_avg']:.1f}")
    print(f"Evaluation reward (20 ep): {eval_reward:.1f}")


def demo_ppo_hyperparameter_sweep():
    """Quick sweep of clip coefficient."""
    print("\n" + "=" * 60)
    print("PPO Clip Coefficient Sweep")
    print("=" * 60)
    
    clip_values = [0.1, 0.2, 0.3]
    
    for clip in clip_values:
        agent = PPO(
            env_id="CartPole-v1",
            n_envs=4, n_steps=128, n_epochs=4,
            clip_coef=clip, seed=42,
        )
        result = agent.train(total_timesteps=100000, print_interval=200000)
        print(f"clip={clip}: Final avg = {result['final_avg']:.1f}")


if __name__ == "__main__":
    demo_ppo_cartpole()
    demo_ppo_hyperparameter_sweep()
