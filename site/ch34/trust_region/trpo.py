"""
Chapter 34.3.1: Trust Region Policy Optimization (TRPO)
========================================================
Implementation of TRPO with conjugate gradient optimization
and backtracking line search.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class TRPOPolicy(nn.Module):
    """Policy network for TRPO (separate from value network)."""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
    
    def forward(self, obs):
        return Categorical(logits=self.net(obs))
    
    def get_log_prob(self, obs, actions):
        dist = self.forward(obs)
        return dist.log_prob(actions)


class ValueNetwork(nn.Module):
    """Value network (trained separately from policy)."""
    
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Conjugate Gradient
# ---------------------------------------------------------------------------

def conjugate_gradient(
    fvp_fn,
    b: torch.Tensor,
    n_steps: int = 10,
    residual_tol: float = 1e-10,
    damping: float = 0.1,
) -> torch.Tensor:
    """
    Solve F x = b using conjugate gradient, where F is the Fisher
    information matrix accessed only through Fisher-vector products.
    
    Parameters
    ----------
    fvp_fn : callable
        Function computing Fisher-vector product F @ v.
    b : Tensor
        Right-hand side (policy gradient).
    n_steps : int
        Maximum CG iterations.
    damping : float
        Tikhonov regularization (F + damping * I).
    
    Returns
    -------
    x : Tensor
        Approximate solution to F x = b.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)
    
    for _ in range(n_steps):
        fvp = fvp_fn(p) + damping * p  # (F + damping * I) @ p
        alpha = rdotr / (p.dot(fvp) + 1e-8)
        x += alpha * p
        r -= alpha * fvp
        new_rdotr = r.dot(r)
        
        if new_rdotr < residual_tol:
            break
        
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    
    return x


# ---------------------------------------------------------------------------
# TRPO Agent
# ---------------------------------------------------------------------------

class TRPO:
    """
    Trust Region Policy Optimization.
    
    Parameters
    ----------
    env : gym.Env
        Environment.
    max_kl : float
        KL divergence constraint (trust region radius).
    gamma : float
        Discount factor.
    lam : float
        GAE lambda.
    cg_iters : int
        Conjugate gradient iterations.
    cg_damping : float
        CG damping coefficient.
    line_search_steps : int
        Backtracking line search steps.
    backtrack_coeff : float
        Backtracking step decay.
    """
    
    def __init__(
        self,
        env: gym.Env,
        max_kl: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.97,
        hidden_dim: int = 64,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        line_search_steps: int = 10,
        backtrack_coeff: float = 0.5,
        value_lr: float = 1e-3,
        value_epochs: int = 5,
    ):
        self.env = env
        self.max_kl = max_kl
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_steps = line_search_steps
        self.backtrack_coeff = backtrack_coeff
        self.value_epochs = value_epochs
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.policy = TRPOPolicy(obs_dim, act_dim, hidden_dim)
        self.value_fn = ValueNetwork(obs_dim, hidden_dim)
        self.value_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=value_lr)
    
    def _get_flat_params(self, model):
        return torch.cat([p.data.reshape(-1) for p in model.parameters()])
    
    def _set_flat_params(self, model, flat_params):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx:idx + n].reshape(p.shape))
            idx += n
    
    def _get_flat_grad(self, loss, model, retain_graph=False):
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph)
        return torch.cat([g.reshape(-1) for g in grads])
    
    def collect_trajectories(self, n_steps=2048):
        """Collect rollout data."""
        states, actions, rewards, dones, log_probs = [], [], [], [], []
        obs, _ = self.env.reset()
        episode_rewards = []
        ep_reward = 0.0
        
        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = self.policy(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.item())
            
            ep_reward += reward
            obs = next_obs
            
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = self.env.reset()
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            torch.FloatTensor(log_probs),
            episode_rewards,
        )
    
    def compute_gae(self, rewards, dones, values, last_value):
        """Compute GAE advantages and returns."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value
            else:
                next_val = values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * non_terminal * last_gae
        
        returns = advantages + values
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
    
    def fisher_vector_product(self, obs, v):
        """Compute Fisher-vector product F @ v."""
        dist = self.policy(obs)
        kl = torch.distributions.kl_divergence(dist, dist).mean()
        
        kl_grad = self._get_flat_grad(kl, self.policy, retain_graph=True)
        kl_v = kl_grad.dot(v)
        fvp = self._get_flat_grad(kl_v, self.policy)
        
        return fvp
    
    def update_policy(self, obs, actions, advantages, old_log_probs):
        """TRPO policy update with CG and line search."""
        # Surrogate loss
        new_log_probs = self.policy.get_log_prob(obs, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = (ratio * advantages).mean()
        
        # Policy gradient
        policy_grad = self._get_flat_grad(surrogate, self.policy, retain_graph=True)
        
        if policy_grad.norm() < 1e-8:
            return 0.0, 0.0
        
        # Conjugate gradient: compute s = F^{-1} g
        fvp_fn = lambda v: self.fisher_vector_product(obs, v)
        step_dir = conjugate_gradient(
            fvp_fn, policy_grad,
            n_steps=self.cg_iters,
            damping=self.cg_damping,
        )
        
        # Compute max step size: sqrt(2δ / s^T F s)
        sFs = step_dir.dot(fvp_fn(step_dir))
        if sFs <= 0:
            return 0.0, 0.0
        
        max_step = torch.sqrt(2 * self.max_kl / (sFs + 1e-8))
        full_step = max_step * step_dir
        
        # Backtracking line search
        old_params = self._get_flat_params(self.policy)
        expected_improve = policy_grad.dot(full_step).item()
        
        for k in range(self.line_search_steps):
            coeff = self.backtrack_coeff ** k
            new_params = old_params + coeff * full_step
            self._set_flat_params(self.policy, new_params)
            
            with torch.no_grad():
                new_log_probs = self.policy.get_log_prob(obs, actions)
                new_ratio = torch.exp(new_log_probs - old_log_probs)
                new_surrogate = (new_ratio * advantages).mean().item()
                
                new_dist = self.policy(obs)
                old_dist = Categorical(logits=self.policy.net(obs).detach())
                kl = torch.distributions.kl_divergence(
                    Categorical(logits=old_dist.logits), new_dist
                ).mean().item()
            
            actual_improve = new_surrogate - surrogate.item()
            
            if kl <= self.max_kl and actual_improve > 0:
                return actual_improve, kl
        
        # Line search failed, revert
        self._set_flat_params(self.policy, old_params)
        return 0.0, 0.0
    
    def update_value(self, obs, returns):
        """Update value function via MSE regression."""
        for _ in range(self.value_epochs):
            values = self.value_fn(obs)
            loss = nn.functional.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
        return loss.item()
    
    def train(self, n_iterations=100, steps_per_iter=2048, print_interval=10):
        all_rewards = []
        recent = deque(maxlen=100)
        
        for iteration in range(1, n_iterations + 1):
            states, actions, rewards, dones, old_log_probs, ep_rewards = \
                self.collect_trajectories(steps_per_iter)
            
            with torch.no_grad():
                values = self.value_fn(states).numpy()
                last_value = self.value_fn(states[-1:]).item()
            
            advantages, returns = self.compute_gae(rewards, dones, values, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # TRPO update
            improve, kl = self.update_policy(states, actions, advantages, old_log_probs)
            value_loss = self.update_value(states, returns)
            
            for r in ep_rewards:
                all_rewards.append(r)
                recent.append(r)
            
            if iteration % print_interval == 0 and len(recent) > 0:
                print(
                    f"Iter {iteration:>4d} | "
                    f"Avg(100): {np.mean(recent):>7.1f} | "
                    f"Improve: {improve:>8.5f} | "
                    f"KL: {kl:>7.5f} | "
                    f"VLoss: {value_loss:>7.4f}"
                )
        
        return all_rewards


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_trpo():
    print("=" * 60)
    print("TRPO on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    agent = TRPO(
        env=env, max_kl=0.01, gamma=0.99, lam=0.97,
        cg_iters=10, cg_damping=0.1,
    )
    rewards = agent.train(n_iterations=100, steps_per_iter=2048, print_interval=10)
    env.close()
    
    if len(rewards) >= 100:
        print(f"\nFinal avg (last 100): {np.mean(rewards[-100:]):.1f}")
    return rewards


if __name__ == "__main__":
    demo_trpo()
