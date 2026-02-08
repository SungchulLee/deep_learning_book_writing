"""
Chapter 34.4.4: SAC Complete Implementation
=============================================
Production-quality SAC with automatic temperature tuning,
squashed Gaussian policy, and twin critics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from collections import deque


LOG_STD_MIN, LOG_STD_MAX = -20, 2
EPS = 1e-6


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.cap = capacity
        self.idx = self.size = 0
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        self.s[self.idx], self.a[self.idx] = s, a
        self.r[self.idx], self.s2[self.idx], self.d[self.idx] = r, s2, float(d)
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
    
    def sample(self, n):
        i = np.random.randint(0, self.size, n)
        return (torch.FloatTensor(self.s[i]), torch.FloatTensor(self.a[i]),
                torch.FloatTensor(self.r[i]), torch.FloatTensor(self.s2[i]),
                torch.FloatTensor(self.d[i]))


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class SquashedGaussianActor(nn.Module):
    """SAC stochastic actor with squashed Gaussian policy."""
    
    def __init__(self, obs_dim, act_dim, hidden=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
    
    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    
    def sample(self, obs):
        """
        Sample action with reparameterization and compute log-prob.
        
        Returns: action, log_prob, mean_action
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # Reparameterized sample
        u = dist.rsample()
        action = torch.tanh(u) * self.max_action
        
        # Log probability with tanh correction
        # log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - (action / self.max_action).pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1)
        
        mean_action = torch.tanh(mu) * self.max_action
        
        return action, log_prob, mean_action


class TwinQCritic(nn.Module):
    """Twin Q-networks for SAC."""
    
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
        sa = torch.cat([obs, action], -1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SAC:
    """
    Soft Actor-Critic with automatic entropy tuning.
    
    Parameters
    ----------
    env : gym.Env
        Continuous action environment.
    lr : float
        Learning rate for all networks.
    gamma : float
        Discount factor.
    tau : float
        Soft target update coefficient.
    alpha_lr : float
        Temperature learning rate.
    init_alpha : float
        Initial temperature value.
    buffer_size : int
        Replay buffer size.
    batch_size : int
        Training minibatch size.
    warmup_steps : int
        Random exploration steps before training.
    auto_alpha : bool
        Whether to automatically tune temperature.
    """
    
    def __init__(
        self,
        env: gym.Env,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha_lr=3e-4,
        init_alpha=0.2,
        hidden_dim=256,
        buffer_size=1000000,
        batch_size=256,
        warmup_steps=5000,
        auto_alpha=True,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.auto_alpha = auto_alpha
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # Actor
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Twin Critics
        self.critic = TwinQCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target = TwinQCritic(obs_dim, act_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Temperature (alpha)
        if auto_alpha:
            self.target_entropy = -act_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = init_alpha
        
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)
    
    def select_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(obs_t)
        if deterministic:
            return mean_action.numpy().flatten()
        return action.numpy().flatten()
    
    def _soft_update(self):
        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
    
    def update(self):
        if self.buffer.size < self.batch_size:
            return {}
        
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        
        # === Critic Update ===
        with torch.no_grad():
            next_a, next_log_prob, _ = self.actor.sample(s2)
            target_q1, target_q2 = self.critic_target(s2, next_a)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = r + self.gamma * (1 - d) * target_q
        
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # === Actor Update ===
        new_a, log_prob, _ = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # === Temperature Update ===
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        
        # Soft update targets
        self._soft_update()
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": -log_prob.mean().item(),
        }
    
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
                metrics = self.update()
            
            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                recent.append(ep_r)
                ep_r = 0.0
                obs, _ = self.env.reset()
            
            if step % print_interval == 0 and recent:
                m = metrics if step >= self.warmup_steps else {}
                print(
                    f"Step {step:>8d} | "
                    f"Avg(100): {np.mean(recent):>8.1f} | "
                    f"α: {self.alpha:>6.4f} | "
                    f"H: {m.get('entropy', 0):>6.3f}"
                )
        
        return ep_rewards
    
    def evaluate(self, n_episodes=10):
        env = gym.make(self.env.spec.id)
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_r, done = 0.0, False
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                done = term or trunc
            rewards.append(total_r)
        env.close()
        return np.mean(rewards), np.std(rewards)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_sac():
    print("=" * 60)
    print("SAC on Pendulum-v1")
    print("=" * 60)
    
    env = gym.make("Pendulum-v1")
    agent = SAC(
        env=env, lr=3e-4, gamma=0.99, tau=0.005,
        auto_alpha=True, warmup_steps=5000,
        batch_size=256, hidden_dim=256,
    )
    
    rewards = agent.train(total_steps=100000, print_interval=10000)
    
    mean_r, std_r = agent.evaluate(n_episodes=20)
    print(f"\nEvaluation: {mean_r:.1f} ± {std_r:.1f}")
    
    env.close()
    return rewards


def demo_sac_alpha_comparison():
    """Compare auto vs fixed temperature."""
    print("\n" + "=" * 60)
    print("SAC: Auto vs Fixed Temperature")
    print("=" * 60)
    
    for auto, alpha in [(True, 0.2), (False, 0.2), (False, 1.0)]:
        env = gym.make("Pendulum-v1")
        agent = SAC(env=env, auto_alpha=auto, init_alpha=alpha, warmup_steps=5000)
        rewards = agent.train(total_steps=50000, print_interval=100000)
        env.close()
        
        label = f"auto (init={alpha})" if auto else f"fixed={alpha}"
        final = np.mean(rewards[-30:]) if len(rewards) >= 30 else 0
        print(f"  α={label:<20}: final avg = {final:.1f}")


if __name__ == "__main__":
    demo_sac()
