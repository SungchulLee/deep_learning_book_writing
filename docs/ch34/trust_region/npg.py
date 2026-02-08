"""
Chapter 34.3.2: Natural Policy Gradient
=========================================
Implementation of Natural Policy Gradient with Fisher information
matrix computation and comparison with vanilla policy gradient.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import List
from collections import deque


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, obs):
        return Categorical(logits=self.net(obs))


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Fisher Information Utilities
# ---------------------------------------------------------------------------

def compute_fisher_vector_product(policy, obs, v, damping=0.1):
    """
    Compute Fisher-vector product Fv using double backprop.
    
    F = E[∇log π ∇log π^T]
    Fv = ∇(∇KL · v)
    """
    dist = policy(obs)
    # KL divergence of distribution with itself (for Hessian computation)
    log_probs = dist.logits - dist.logits.logsumexp(dim=-1, keepdim=True)
    probs = dist.probs
    kl = (probs * log_probs).sum(-1).mean()
    
    params = list(policy.parameters())
    kl_grad = torch.autograd.grad(kl, params, create_graph=True)
    kl_grad_flat = torch.cat([g.reshape(-1) for g in kl_grad])
    
    kl_v = kl_grad_flat.dot(v)
    fvp_grads = torch.autograd.grad(kl_v, params)
    fvp = torch.cat([g.reshape(-1) for g in fvp_grads])
    
    return fvp + damping * v


def conjugate_gradient(fvp_fn, b, n_steps=10, residual_tol=1e-10):
    """Solve Fx = b using conjugate gradient."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)
    
    for _ in range(n_steps):
        Fp = fvp_fn(p)
        alpha = rdotr / (p.dot(Fp) + 1e-8)
        x += alpha * p
        r -= alpha * Fp
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / (rdotr + 1e-8)) * p
        rdotr = new_rdotr
    
    return x


def compute_empirical_fisher(policy, obs, actions, n_samples=None):
    """
    Compute empirical Fisher matrix (for small networks).
    
    F = (1/N) Σ ∇log π(a|s) ∇log π(a|s)^T
    
    Only feasible for small parameter counts.
    """
    if n_samples is None:
        n_samples = len(obs)
    
    indices = np.random.choice(len(obs), min(n_samples, len(obs)), replace=False)
    
    grads = []
    for i in indices:
        policy.zero_grad()
        dist = policy(obs[i:i+1])
        log_prob = dist.log_prob(actions[i:i+1])
        log_prob.backward()
        
        grad = torch.cat([p.grad.reshape(-1) for p in policy.parameters()])
        grads.append(grad)
    
    grads = torch.stack(grads)
    fisher = grads.T @ grads / len(grads)
    
    return fisher


# ---------------------------------------------------------------------------
# Natural Policy Gradient Agent
# ---------------------------------------------------------------------------

class NaturalPolicyGradient:
    """
    Natural Policy Gradient agent.
    
    Uses Fisher information matrix to compute natural gradient
    direction, providing parameterization-invariant updates.
    """
    
    def __init__(
        self,
        env: gym.Env,
        step_size: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.97,
        hidden_dim: int = 64,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        value_lr: float = 1e-3,
        value_epochs: int = 5,
    ):
        self.env = env
        self.step_size = step_size
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.value_epochs = value_epochs
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        self.policy = PolicyNet(obs_dim, act_dim, hidden_dim)
        self.value_fn = ValueNet(obs_dim, hidden_dim)
        self.value_opt = torch.optim.Adam(self.value_fn.parameters(), lr=value_lr)
    
    def _flat_params(self):
        return torch.cat([p.data.reshape(-1) for p in self.policy.parameters()])
    
    def _set_flat_params(self, flat):
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx + n].reshape(p.shape))
            idx += n
    
    def collect_data(self, n_steps=2048):
        states, actions, rewards, dones = [], [], [], []
        obs, _ = self.env.reset()
        ep_rewards = []
        ep_r = 0.0
        
        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = self.policy(obs_t)
                action = dist.sample()
            
            next_obs, reward, term, trunc, _ = self.env.step(action.item())
            done = term or trunc
            
            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            ep_r += reward
            
            obs = next_obs
            if done:
                ep_rewards.append(ep_r)
                ep_r = 0.0
                obs, _ = self.env.reset()
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            ep_rewards,
        )
    
    def compute_gae(self, rewards, dones, values, last_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv = last_value if t == T - 1 else values[t + 1]
            nt = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nv * nt - values[t]
            advantages[t] = gae = delta + self.gamma * self.lam * nt * gae
        return torch.FloatTensor(advantages), torch.FloatTensor(advantages + values)
    
    def update_policy(self, obs, actions, advantages):
        """Natural policy gradient update."""
        # Compute policy gradient
        dist = self.policy(obs)
        log_probs = dist.log_prob(actions)
        surrogate = (log_probs * advantages).mean()
        
        grads = torch.autograd.grad(surrogate, self.policy.parameters())
        pg = torch.cat([g.reshape(-1) for g in grads])
        
        if pg.norm() < 1e-8:
            return 0.0
        
        # Compute natural gradient via CG
        fvp_fn = lambda v: compute_fisher_vector_product(
            self.policy, obs, v, self.cg_damping
        )
        natural_grad = conjugate_gradient(fvp_fn, pg, self.cg_iters)
        
        # Step size based on KL constraint: α = sqrt(2δ / g^T F^{-1} g)
        sFs = pg.dot(natural_grad)
        if sFs <= 0:
            return 0.0
        
        alpha = torch.sqrt(2 * self.step_size / (sFs + 1e-8))
        
        # Update parameters
        old_params = self._flat_params()
        new_params = old_params + alpha * natural_grad
        self._set_flat_params(new_params)
        
        return (alpha * natural_grad).norm().item()
    
    def update_value(self, obs, returns):
        for _ in range(self.value_epochs):
            pred = self.value_fn(obs)
            loss = nn.functional.mse_loss(pred, returns)
            self.value_opt.zero_grad()
            loss.backward()
            self.value_opt.step()
    
    def train(self, n_iters=100, steps_per_iter=2048, print_interval=10):
        all_rewards = []
        recent = deque(maxlen=100)
        
        for it in range(1, n_iters + 1):
            obs, actions, rewards, dones, ep_rewards = self.collect_data(steps_per_iter)
            
            with torch.no_grad():
                values = self.value_fn(obs).numpy()
                last_val = self.value_fn(obs[-1:]).item()
            
            advantages, returns = self.compute_gae(rewards, dones, values, last_val)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            step_norm = self.update_policy(obs, actions, advantages)
            self.update_value(obs, returns)
            
            for r in ep_rewards:
                all_rewards.append(r)
                recent.append(r)
            
            if it % print_interval == 0 and len(recent) > 0:
                print(
                    f"Iter {it:>4d} | Avg(100): {np.mean(recent):>7.1f} | "
                    f"Step: {step_norm:>8.5f}"
                )
        
        return all_rewards


# ---------------------------------------------------------------------------
# Comparison: Standard PG vs Natural PG
# ---------------------------------------------------------------------------

class VanillaPG(NaturalPolicyGradient):
    """Standard policy gradient for comparison."""
    
    def __init__(self, env, lr=1e-3, **kwargs):
        super().__init__(env, **kwargs)
        self.pg_lr = lr
    
    def update_policy(self, obs, actions, advantages):
        dist = self.policy(obs)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * advantages).mean()
        
        # Standard gradient update
        grads = torch.autograd.grad(loss, self.policy.parameters())
        with torch.no_grad():
            for p, g in zip(self.policy.parameters(), grads):
                p.data -= self.pg_lr * g
        
        return sum(g.norm().item() for g in grads)


def demo_npg():
    print("=" * 60)
    print("Natural Policy Gradient on CartPole-v1")
    print("=" * 60)
    
    env = gym.make("CartPole-v1")
    agent = NaturalPolicyGradient(env=env, step_size=0.01, gamma=0.99, lam=0.97)
    rewards = agent.train(n_iters=80, steps_per_iter=2048, print_interval=10)
    env.close()
    
    if len(rewards) >= 50:
        print(f"\nFinal avg (last 50): {np.mean(rewards[-50:]):.1f}")


def demo_comparison():
    print("\n" + "=" * 60)
    print("Standard PG vs Natural PG Comparison")
    print("=" * 60)
    
    n_iters = 50
    n_trials = 3
    
    for name, AgentClass, kwargs in [
        ("Vanilla PG", VanillaPG, {"lr": 1e-3}),
        ("Natural PG", NaturalPolicyGradient, {"step_size": 0.01}),
    ]:
        trial_results = []
        for trial in range(n_trials):
            torch.manual_seed(trial)
            env = gym.make("CartPole-v1")
            agent = AgentClass(env=env, gamma=0.99, lam=0.97, **kwargs)
            rewards = agent.train(n_iters=n_iters, steps_per_iter=2048, print_interval=n_iters + 1)
            env.close()
            trial_results.append(np.mean(rewards[-30:]) if len(rewards) >= 30 else 0)
        
        print(f"{name:<15}: {np.mean(trial_results):.1f} ± {np.std(trial_results):.1f}")


if __name__ == "__main__":
    demo_npg()
    demo_comparison()
