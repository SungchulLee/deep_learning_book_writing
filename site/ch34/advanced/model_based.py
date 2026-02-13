"""
Chapter 34.5.4: Model-Based Reinforcement Learning
====================================================
Dyna-style model-based RL, learned dynamics model with
ensemble uncertainty, and MBPO-style training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Dynamics Model
# ---------------------------------------------------------------------------

class DynamicsModel(nn.Module):
    """
    Probabilistic dynamics model: predicts Gaussian over (next_state, reward).
    """
    def __init__(self, obs_dim, act_dim, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        output_dim = obs_dim + 1  # next_state + reward
        self.mu_head = nn.Linear(hidden, output_dim)
        self.log_var_head = nn.Linear(hidden, output_dim)
        self.obs_dim = obs_dim
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        h = self.net(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h).clamp(-10, 0.5)
        return mu, log_var
    
    def predict(self, obs, action, deterministic=False):
        mu, log_var = self.forward(obs, action)
        if deterministic:
            output = mu
        else:
            std = (log_var * 0.5).exp()
            output = mu + std * torch.randn_like(std)
        
        next_obs = obs + output[:, :self.obs_dim]  # Predict delta
        reward = output[:, self.obs_dim]
        return next_obs, reward
    
    def loss(self, obs, action, next_obs, reward):
        mu, log_var = self.forward(obs, action)
        target = torch.cat([next_obs - obs, reward.unsqueeze(-1)], dim=-1)
        
        inv_var = (-log_var).exp()
        mse = ((mu - target) ** 2) * inv_var
        nll = (mse + log_var).mean()
        return nll


class EnsembleDynamics:
    """Ensemble of dynamics models for uncertainty estimation."""
    
    def __init__(self, obs_dim, act_dim, n_models=5, hidden=200, lr=1e-3):
        self.models = [DynamicsModel(obs_dim, act_dim, hidden) for _ in range(n_models)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self.n_models = n_models
    
    def train_step(self, obs, action, next_obs, reward, n_epochs=5):
        """Train all models on the same data."""
        losses = []
        for model, opt in zip(self.models, self.optimizers):
            for _ in range(n_epochs):
                loss = model.loss(obs, action, next_obs, reward)
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(loss.item())
        return np.mean(losses)
    
    def predict(self, obs, action):
        """Predict with random model from ensemble (TS1 sampling)."""
        idx = np.random.randint(self.n_models)
        return self.models[idx].predict(obs, action)
    
    def get_uncertainty(self, obs, action):
        """Ensemble disagreement as uncertainty measure."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                next_obs, _ = model.predict(obs, action, deterministic=True)
            predictions.append(next_obs)
        predictions = torch.stack(predictions)
        return predictions.std(dim=0).mean(dim=-1)


# ---------------------------------------------------------------------------
# Dyna-Style Agent
# ---------------------------------------------------------------------------

class DynaAgent:
    """
    Dyna-style model-based RL agent.
    
    Interleaves real environment interaction with model-based
    rollouts for improved sample efficiency.
    """
    
    def __init__(
        self, env, lr_policy=1e-3, lr_model=1e-3, gamma=0.99,
        model_rollouts=10, rollout_length=1, hidden=128,
    ):
        self.env = env
        self.gamma = gamma
        self.model_rollouts = model_rollouts
        self.rollout_length = rollout_length
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Dynamics model
        self.dynamics = EnsembleDynamics(obs_dim, act_dim, n_models=3, lr=lr_model)
        
        # Simple actor (deterministic + noise)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.max_action = float(env.action_space.high[0])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_policy)
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_model)
        
        # Real data buffer
        self.real_buffer = {"s": [], "a": [], "r": [], "s2": [], "d": []}
    
    def select_action(self, obs, noise=0.1):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).numpy().flatten() * self.max_action
        action += np.random.normal(0, noise * self.max_action, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)
    
    def store_real(self, s, a, r, s2, d):
        for key, val in zip(["s", "a", "r", "s2", "d"], [s, a, r, s2, float(d)]):
            self.real_buffer[key].append(val)
    
    def train_model(self, batch_size=256):
        """Train dynamics model on real data."""
        n = len(self.real_buffer["s"])
        if n < batch_size:
            return 0.0
        
        idx = np.random.choice(n, batch_size, replace=False)
        s = torch.FloatTensor(np.array([self.real_buffer["s"][i] for i in idx]))
        a = torch.FloatTensor(np.array([self.real_buffer["a"][i] for i in idx]))
        r = torch.FloatTensor(np.array([self.real_buffer["r"][i] for i in idx]))
        s2 = torch.FloatTensor(np.array([self.real_buffer["s2"][i] for i in idx]))
        
        loss = self.dynamics.train_step(s, a, s2, r, n_epochs=5)
        return loss
    
    def model_rollout(self, start_states, length=1):
        """Generate synthetic data from learned model."""
        synthetic = {"s": [], "a": [], "r": [], "s2": []}
        
        obs = start_states.clone()
        for _ in range(length):
            with torch.no_grad():
                actions = self.actor(obs) * self.max_action
                next_obs, rewards = self.dynamics.predict(obs, actions)
            
            synthetic["s"].append(obs)
            synthetic["a"].append(actions)
            synthetic["r"].append(rewards)
            synthetic["s2"].append(next_obs)
            
            obs = next_obs
        
        return {k: torch.cat(v) for k, v in synthetic.items()}
    
    def update_policy(self, states, actions, rewards, next_states):
        """Update actor and critic on data."""
        # Critic update
        with torch.no_grad():
            next_a = self.actor(next_states) * self.max_action
            target_q = rewards + self.gamma * self.critic(
                torch.cat([next_states, next_a], -1)
            ).squeeze(-1)
        
        current_q = self.critic(torch.cat([states, actions], -1)).squeeze(-1)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        pred_a = self.actor(states) * self.max_action
        actor_loss = -self.critic(torch.cat([states, pred_a], -1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def train(self, total_steps=50000, print_interval=5000):
        obs, _ = self.env.reset()
        ep_rewards, recent = [], deque(maxlen=50)
        ep_r = 0.0
        
        for step in range(1, total_steps + 1):
            action = self.select_action(obs)
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            
            self.store_real(obs, action, reward, next_obs, done)
            ep_r += reward
            
            # Train model periodically
            if step % 250 == 0 and len(self.real_buffer["s"]) > 500:
                self.train_model()
                
                # Model rollouts
                n = len(self.real_buffer["s"])
                idx = np.random.choice(n, min(self.model_rollouts, n), replace=False)
                starts = torch.FloatTensor(
                    np.array([self.real_buffer["s"][i] for i in idx])
                )
                syn = self.model_rollout(starts, self.rollout_length)
                self.update_policy(syn["s"], syn["a"], syn["r"], syn["s2"])
            
            # Also train on real data
            if len(self.real_buffer["s"]) > 256 and step % 10 == 0:
                idx = np.random.choice(len(self.real_buffer["s"]), 256, replace=False)
                s = torch.FloatTensor(np.array([self.real_buffer["s"][i] for i in idx]))
                a = torch.FloatTensor(np.array([self.real_buffer["a"][i] for i in idx]))
                r = torch.FloatTensor(np.array([self.real_buffer["r"][i] for i in idx]))
                s2 = torch.FloatTensor(np.array([self.real_buffer["s2"][i] for i in idx]))
                self.update_policy(s, a, r, s2)
            
            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                recent.append(ep_r)
                ep_r = 0.0
                obs, _ = self.env.reset()
            
            if step % print_interval == 0 and recent:
                print(f"Step {step:>7d} | Avg(50): {np.mean(recent):>8.1f} | Buffer: {len(self.real_buffer['s'])}")
        
        return ep_rewards


def demo_dyna():
    print("=" * 60)
    print("Dyna-Style Model-Based RL on Pendulum-v1")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    agent = DynaAgent(env, model_rollouts=20, rollout_length=1)
    rewards = agent.train(total_steps=30000, print_interval=5000)
    env.close()
    
    if len(rewards) >= 30:
        print(f"\nFinal avg (last 30): {np.mean(rewards[-30:]):.1f}")


if __name__ == "__main__":
    demo_dyna()
