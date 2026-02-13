"""
33.2.6 Rainbow DQN
===================

Simplified Rainbow combining: Double DQN + Dueling + PER + N-step + C51 + NoisyNets.
Also includes a practical "Mini-Rainbow" with only the top-3 components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import math
import random
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# NoisyLinear (from noisy_networks.py)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.mu_w = nn.Parameter(torch.empty(out_f, in_f))
        self.sigma_w = nn.Parameter(torch.empty(out_f, in_f))
        self.mu_b = nn.Parameter(torch.empty(out_f))
        self.sigma_b = nn.Parameter(torch.empty(out_f))
        self.register_buffer('eps_in', torch.zeros(in_f))
        self.register_buffer('eps_out', torch.zeros(out_f))
        bound = 1 / math.sqrt(in_f)
        self.mu_w.data.uniform_(-bound, bound)
        self.mu_b.data.uniform_(-bound, bound)
        self.sigma_w.data.fill_(sigma_init / math.sqrt(in_f))
        self.sigma_b.data.fill_(sigma_init / math.sqrt(out_f))
        self.reset_noise()

    @staticmethod
    def _f(x): return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in.copy_(self._f(torch.randn(self.in_f)))
        self.eps_out.copy_(self._f(torch.randn(self.out_f)))

    def forward(self, x):
        if self.training:
            w = self.mu_w + self.sigma_w * (self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0))
            b = self.mu_b + self.sigma_b * self.eps_out
        else:
            w, b = self.mu_w, self.mu_b
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# Rainbow Network: Dueling + Distributional + Noisy
# ---------------------------------------------------------------------------

class RainbowNetwork(nn.Module):
    """Full Rainbow architecture combining Dueling + C51 + NoisyNets."""

    def __init__(self, state_dim: int, action_dim: int, n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0, hidden: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.register_buffer('atoms', torch.linspace(v_min, v_max, n_atoms))
        self.v_min, self.v_max = v_min, v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Shared feature
        self.feature = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        # Value stream (noisy)
        self.v_noisy1 = NoisyLinear(hidden, hidden)
        self.v_noisy2 = NoisyLinear(hidden, n_atoms)
        # Advantage stream (noisy)
        self.a_noisy1 = NoisyLinear(hidden, hidden)
        self.a_noisy2 = NoisyLinear(hidden, action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-probs: (batch, action_dim, n_atoms)."""
        feat = self.feature(x)
        v = F.relu(self.v_noisy1(feat))
        v = self.v_noisy2(v).unsqueeze(1)  # (batch, 1, atoms)
        a = F.relu(self.a_noisy1(feat))
        a = self.a_noisy2(a).view(-1, self.action_dim, self.n_atoms)
        q_atoms = v + a - a.mean(dim=1, keepdim=True)
        return F.log_softmax(q_atoms, dim=2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward(x).exp()
        return (probs * self.atoms.unsqueeze(0).unsqueeze(0)).sum(2)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ---------------------------------------------------------------------------
# N-step Replay Buffer with Priorities
# ---------------------------------------------------------------------------

class NStepPERBuffer:
    """N-step replay with prioritized sampling (simplified, list-based)."""

    def __init__(self, capacity: int, state_dim: int, n_step: int = 3,
                 gamma: float = 0.99, alpha: float = 0.5):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha

        self.states = np.zeros((capacity, state_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)  # n-step reward
        self.next_states = np.zeros((capacity, state_dim), np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.priorities = np.ones(capacity, np.float64)

        self.ptr = 0
        self.size = 0
        self.n_step_buffer: List = []

    def _compute_nstep(self) -> Optional[Tuple]:
        """Compute n-step return from the n-step buffer."""
        if len(self.n_step_buffer) < self.n_step:
            return None
        R = sum(self.gamma ** i * t[2] for i, t in enumerate(self.n_step_buffer))
        s0 = self.n_step_buffer[0][0]
        a0 = self.n_step_buffer[0][1]
        sn = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        return (s0, a0, R, sn, done)

    def push(self, s, a, r, ns, done):
        self.n_step_buffer.append((s, a, r, ns, done))
        if done:
            while self.n_step_buffer:
                R = sum(self.gamma**i * t[2] for i, t in enumerate(self.n_step_buffer))
                entry = (self.n_step_buffer[0][0], self.n_step_buffer[0][1],
                         R, ns, True)
                self._store(*entry)
                self.n_step_buffer.pop(0)
        elif len(self.n_step_buffer) >= self.n_step:
            entry = self._compute_nstep()
            if entry:
                self._store(*entry)
            self.n_step_buffer.pop(0)

    def _store(self, s, a, r, ns, d):
        max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.dones[self.ptr] = float(d)
        self.priorities[self.ptr] = max_p
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        priors = self.priorities[:self.size] ** self.alpha
        probs = priors / priors.sum()
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
            torch.FloatTensor(weights),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Rainbow Agent
# ---------------------------------------------------------------------------

class RainbowAgent:
    def __init__(self, state_dim, action_dim, n_atoms=51, v_min=0, v_max=200,
                 n_step=3, gamma=0.99, lr=6.25e-5, batch_size=32,
                 buf_cap=50000, target_freq=500, per_alpha=0.5, per_beta=0.4):
        self.gamma = gamma
        self.gamma_n = gamma ** n_step
        self.batch_size = batch_size
        self.target_freq = target_freq
        self.per_beta_start = per_beta
        self.action_dim = action_dim

        self.online = RainbowNetwork(state_dim, action_dim, n_atoms, v_min, v_max)
        self.target = RainbowNetwork(state_dim, action_dim, n_atoms, v_min, v_max)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr, eps=1.5e-4)
        self.buf = NStepPERBuffer(buf_cap, state_dim, n_step, gamma, per_alpha)
        self.step = 0

    def act(self, state, training=True):
        self.step += 1
        if training:
            self.online.reset_noise()
        st = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if training:
                self.online.train()
            else:
                self.online.eval()
            return self.online.q_values(st).argmax(1).item()

    def store(self, s, a, r, ns, d):
        self.buf.push(s, a, r, ns, d)

    def update(self):
        if len(self.buf) < 500:
            return
        beta = min(1.0, self.per_beta_start + self.step * (1.0 - self.per_beta_start) / 100000)
        s, a, r, ns, d, w, idx = self.buf.sample(self.batch_size, beta)

        self.online.reset_noise()
        self.target.reset_noise()

        # C51 projection with n-step gamma and Double DQN action selection
        log_p = self.online(s)
        a_idx = a.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.online.n_atoms)
        log_p_a = log_p.gather(1, a_idx).squeeze(1)

        with torch.no_grad():
            # Double: online selects, target evaluates
            next_a = self.online.q_values(ns).argmax(1)
            next_log_p = self.target(ns)
            na_idx = next_a.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.target.n_atoms)
            next_p = next_log_p.exp().gather(1, na_idx).squeeze(1)

            atoms = self.target.atoms
            tz = r.unsqueeze(1) + (1 - d.unsqueeze(1)) * self.gamma_n * atoms
            tz = tz.clamp(self.target.v_min, self.target.v_max)
            b = (tz - self.target.v_min) / self.target.delta_z
            l = b.floor().long().clamp(0, self.target.n_atoms - 1)
            u = b.ceil().long().clamp(0, self.target.n_atoms - 1)

            m = torch.zeros_like(next_p)
            offset = torch.arange(s.size(0)).unsqueeze(1) * self.target.n_atoms
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_p * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_p * (b - l.float())).view(-1))

        # Weighted cross-entropy loss
        element_loss = -(m * log_p_a).sum(1)
        loss = (w * element_loss).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # Update priorities
        self.buf.update_priorities(idx, element_loss.detach().numpy())

        if self.step % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_rainbow():
    print("=" * 60)
    print("Rainbow DQN Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    print(f"\nEnvironment: CartPole-v1 (state={sd}, actions={ad})")
    print("Components: Double + Dueling + PER + N-step + C51 + NoisyNets")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    agent = RainbowAgent(sd, ad, n_atoms=51, v_min=0, v_max=500,
                         n_step=3, lr=1e-3, batch_size=32, target_freq=200)

    params = sum(p.numel() for p in agent.online.parameters())
    print(f"Network parameters: {params:,}")

    rewards = []
    for ep in range(300):
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
            avg = np.mean(rewards[-50:])
            print(f"  Episode {ep+1}: avg50={avg:.1f}")

    env.close()
    print(f"\nFinal avg (last 50): {np.mean(rewards[-50:]):.1f}")
    print(f"Best episode: {max(rewards):.0f}")
    print("\nRainbow demo complete!")


if __name__ == "__main__":
    demo_rainbow()
