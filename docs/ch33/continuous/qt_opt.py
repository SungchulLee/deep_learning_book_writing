"""
33.4.2 QT-Opt
===============

Q-learning for continuous actions using Cross-Entropy Method (CEM).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Tuple
import random


# ---------------------------------------------------------------------------
# Q-Network (state-action input)
# ---------------------------------------------------------------------------

class ContinuousQNetwork(nn.Module):
    """Q-network that takes (state, action) as input → scalar Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Cross-Entropy Method for Action Optimization
# ---------------------------------------------------------------------------

class CEM:
    """Cross-Entropy Method for continuous action optimization."""

    def __init__(self, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 n_samples: int = 64, n_elite: int = 6, n_iterations: int = 3):
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations

    def optimize(self, q_network: nn.Module, state: torch.Tensor) -> np.ndarray:
        """Find action that maximizes Q(state, action) using CEM.
        
        Args:
            q_network: Q-network
            state: single state tensor, shape (state_dim,) or (1, state_dim)
            
        Returns:
            Best action found, shape (action_dim,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Initialize distribution
        mu = np.zeros(self.action_dim)
        sigma = np.ones(self.action_dim)

        best_action = mu.copy()
        best_q = -float('inf')

        for _ in range(self.n_iterations):
            # Sample actions from Gaussian
            actions = np.random.normal(mu, sigma, size=(self.n_samples, self.action_dim))
            actions = np.clip(actions, self.action_low, self.action_high)

            # Evaluate Q-values
            actions_t = torch.FloatTensor(actions)
            states_t = state.expand(self.n_samples, -1)
            with torch.no_grad():
                q_values = q_network(states_t, actions_t).numpy()

            # Select elite samples
            elite_idx = np.argsort(q_values)[-self.n_elite:]
            elite_actions = actions[elite_idx]

            # Track best overall
            if q_values[elite_idx[-1]] > best_q:
                best_q = q_values[elite_idx[-1]]
                best_action = actions[elite_idx[-1]].copy()

            # Refit distribution
            mu = elite_actions.mean(axis=0)
            sigma = elite_actions.std(axis=0) + 1e-6

        return best_action

    def optimize_batch(self, q_network: nn.Module, states: torch.Tensor) -> torch.Tensor:
        """Optimize actions for a batch of states."""
        actions = []
        for i in range(states.shape[0]):
            a = self.optimize(q_network, states[i])
            actions.append(a)
        return torch.FloatTensor(np.array(actions))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap, sd, ad):
        self.cap=cap;self.sz=0;self.p=0
        self.s=np.zeros((cap,sd),np.float32);self.a=np.zeros((cap,ad),np.float32)
        self.r=np.zeros(cap,np.float32);self.ns=np.zeros((cap,sd),np.float32)
        self.d=np.zeros(cap,np.float32)
    def push(self,s,a,r,ns,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r;self.ns[self.p]=ns;self.d[self.p]=float(d)
        self.p=(self.p+1)%self.cap;self.sz=min(self.sz+1,self.cap)
    def sample(self,n):
        i=np.random.randint(0,self.sz,n)
        return (torch.FloatTensor(self.s[i]),torch.FloatTensor(self.a[i]),
                torch.FloatTensor(self.r[i]),torch.FloatTensor(self.ns[i]),torch.FloatTensor(self.d[i]))
    def __len__(self): return self.sz


# ---------------------------------------------------------------------------
# QT-Opt Agent
# ---------------------------------------------------------------------------

class QTOptAgent:
    """QT-Opt: Q-learning + CEM for continuous actions."""

    def __init__(self, state_dim, action_dim, action_low, action_high,
                 lr=1e-3, gamma=0.99, tau=0.005, batch_size=128,
                 buf_cap=100000, cem_samples=64, cem_elite=6, cem_iters=3,
                 noise_std=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_low = action_low
        self.action_high = action_high
        self.noise_std = noise_std

        self.online = ContinuousQNetwork(state_dim, action_dim)
        self.target = ContinuousQNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(buf_cap, state_dim, action_dim)

        self.cem = CEM(action_dim, action_low, action_high,
                       cem_samples, cem_elite, cem_iters)

    def act(self, state, training=True):
        s = torch.FloatTensor(state)
        if training:
            # Use CEM with exploration noise
            action = self.cem.optimize(self.online, s)
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, self.action_low, self.action_high)
        else:
            action = self.cem.optimize(self.online, s)
        return action

    def store(self, s, a, r, ns, d):
        self.buf.push(s, a, r, ns, d)

    def update(self):
        if len(self.buf) < self.batch_size:
            return 0.0
        s, a, r, ns, d = self.buf.sample(self.batch_size)

        # Current Q
        q = self.online(s, a)

        # Target: r + γ max_a' Q_target(s', a')
        # Use CEM to find max_a' for each next_state
        with torch.no_grad():
            best_next_a = self.cem.optimize_batch(self.target, ns)
            next_q = self.target(ns, best_next_a)
            targets = r + (1 - d) * self.gamma * next_q

        loss = nn.functional.mse_loss(q, targets)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)
        return loss.item()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_qt_opt():
    print("=" * 60)
    print("QT-Opt Demo")
    print("=" * 60)

    # --- CEM optimization ---
    print("\n--- CEM Action Optimization ---")
    sd, ad = 3, 1
    q_net = ContinuousQNetwork(sd, ad)
    cem = CEM(ad, np.array([-2.0]), np.array([2.0]),
              n_samples=64, n_elite=6, n_iterations=3)

    state = torch.randn(1, sd)
    best_a = cem.optimize(q_net, state)
    print(f"  State: {state.numpy().round(3)}")
    print(f"  Best action (CEM): {best_a.round(3)}")

    # Verify by grid search
    grid = torch.linspace(-2, 2, 100).unsqueeze(1)
    with torch.no_grad():
        grid_q = q_net(state.expand(100, -1), grid).numpy()
    grid_best = grid[np.argmax(grid_q)].item()
    print(f"  Best action (grid): {grid_best:.3f}")
    print(f"  CEM vs grid gap: {abs(best_a[0] - grid_best):.4f}")

    # --- Training on Pendulum ---
    print("\n--- QT-Opt Training on Pendulum-v1 ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    env = gym.make('Pendulum-v1')
    agent = QTOptAgent(3, 1, np.array([-2.0]), np.array([2.0]),
                       lr=1e-3, noise_std=0.3,
                       cem_samples=32, cem_elite=4, cem_iters=2)
    rewards = []

    for ep in range(150):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(s, a, r, ns, done)
            agent.update()
            s = ns; total += r
        rewards.append(total)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: avg50={np.mean(rewards[-50:]):.1f}")

    env.close()
    print(f"\n  Final avg(50): {np.mean(rewards[-50:]):.1f}")
    print("\nQT-Opt demo complete!")


if __name__ == "__main__":
    demo_qt_opt()
