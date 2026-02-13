"""
33.2.2 Dueling DQN
===================

Dueling network architecture with value and advantage streams.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
from typing import Tuple, List
import random


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network: separate value and advantage streams.
    
    Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        # Mean-centering for identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def value_and_advantage(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return separate V and A for analysis."""
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage


class DuelingConvQNetwork(nn.Module):
    """Dueling architecture for Atari-style image observations."""

    def __init__(self, action_dim: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        feat = self.conv(x).view(x.size(0), -1)
        v = self.value_stream(feat)
        a = self.advantage_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap, sd):
        self.cap=cap; self.sz=0; self.p=0
        self.s=np.zeros((cap,sd),np.float32); self.a=np.zeros(cap,np.int64)
        self.r=np.zeros(cap,np.float32); self.ns=np.zeros((cap,sd),np.float32)
        self.d=np.zeros(cap,np.float32)
    def push(self,s,a,r,ns,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r
        self.ns[self.p]=ns;self.d[self.p]=float(d)
        self.p=(self.p+1)%self.cap;self.sz=min(self.sz+1,self.cap)
    def sample(self,n):
        i=np.random.randint(0,self.sz,n)
        return (torch.FloatTensor(self.s[i]),torch.LongTensor(self.a[i]),
                torch.FloatTensor(self.r[i]),torch.FloatTensor(self.ns[i]),
                torch.FloatTensor(self.d[i]))
    def __len__(self): return self.sz


# ---------------------------------------------------------------------------
# Training with Dueling + Double DQN
# ---------------------------------------------------------------------------

class DuelingDQNAgent:
    """Agent combining Dueling architecture with Double DQN."""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 batch_size=64, buf_cap=50000, target_freq=200,
                 eps_end=0.01, eps_decay=5000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.target_freq = target_freq
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.online = DuelingQNetwork(state_dim, action_dim)
        self.target = DuelingQNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(buf_cap, state_dim)

        self.step = 0
        self.updates = 0
        self.v_history = []
        self.a_history = []

    @property
    def epsilon(self):
        return max(self.eps_end, 1.0 - (1.0 - self.eps_end) * self.step / self.eps_decay)

    def act(self, state, training=True):
        if training:
            self.step += 1
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.online(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()

    def store(self, s, a, r, ns, d):
        self.buf.push(s, a, r, ns, d)

    def update(self):
        if len(self.buf) < 500:
            return
        s, a, r, ns, d = self.buf.sample(self.batch_size)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Track V and A decomposition
        with torch.no_grad():
            v, adv = self.online.value_and_advantage(s)
            self.v_history.append(v.mean().item())
            self.a_history.append(adv.abs().mean().item())

        # Double DQN target with dueling architecture
        with torch.no_grad():
            best_a = self.online(ns).argmax(1)
            next_q = self.target(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            targets = r + (1 - d) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q, targets)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_dueling_dqn():
    print("=" * 60)
    print("Dueling DQN Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n
    env.close()

    # --- Architecture comparison ---
    print("\n--- Architecture Comparison ---")
    standard = nn.Sequential(
        nn.Linear(sd, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ad))
    dueling = DuelingQNetwork(sd, ad)

    std_params = sum(p.numel() for p in standard.parameters())
    duel_params = sum(p.numel() for p in dueling.parameters())
    print(f"  Standard Q-Net params: {std_params:,}")
    print(f"  Dueling Q-Net params:  {duel_params:,}")

    # --- V/A decomposition ---
    print("\n--- Value/Advantage Decomposition ---")
    test_states = torch.randn(5, sd)
    q_vals = dueling(test_states)
    v_vals, a_vals = dueling.value_and_advantage(test_states)

    for i in range(5):
        print(f"  State {i}: V={v_vals[i].item():.3f}, "
              f"A={a_vals[i].detach().numpy().round(3)}, "
              f"Q={q_vals[i].detach().numpy().round(3)}")

    # --- Training comparison ---
    print("\n--- Training: Dueling+Double DQN on CartPole ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    agent = DuelingDQNAgent(sd, ad)
    env = gym.make('CartPole-v1')
    rewards = []

    for ep in range(250):
        s, _ = env.reset(); total = 0; done = False
        while not done:
            a = agent.act(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(s, a, r, ns, float(done))
            agent.update()
            s = ns; total += r
        rewards.append(total)
        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"  Episode {ep+1}: avg50={avg:.1f}, ε={agent.epsilon:.3f}")
    env.close()

    # --- Analyze V/A streams ---
    print("\n--- V/A Stream Analysis ---")
    if agent.v_history:
        v_early = np.mean(agent.v_history[:200])
        v_late = np.mean(agent.v_history[-200:])
        a_early = np.mean(agent.a_history[:200])
        a_late = np.mean(agent.a_history[-200:])
        print(f"  V stream — early: {v_early:.3f}, late: {v_late:.3f}")
        print(f"  A stream (|A|) — early: {a_early:.3f}, late: {a_late:.3f}")
        print(f"  V/|A| ratio — early: {v_early/(a_early+1e-8):.2f}, "
              f"late: {v_late/(a_late+1e-8):.2f}")

    # --- Conv dueling shape check ---
    print("\n--- Conv Dueling Architecture ---")
    conv_duel = DuelingConvQNetwork(action_dim=4)
    dummy = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    q_out = conv_duel(dummy.float())
    print(f"  Input: {dummy.shape} → Output: {q_out.shape}")
    print(f"  Params: {sum(p.numel() for p in conv_duel.parameters()):,}")

    print("\nDueling DQN demo complete!")


if __name__ == "__main__":
    demo_dueling_dqn()
