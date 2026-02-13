"""
33.4.1 Normalized Advantage Functions (NAF)
=============================================

Value-based RL for continuous action spaces using quadratic advantage.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Tuple, List
import random


class NAFNetwork(nn.Module):
    """NAF: Q(s,a) = V(s) - 0.5*(a-μ)^T P (a-μ), P = LL^T."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_low: float = -1.0, action_high: float = 1.0):
        super().__init__()
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.tril_size = action_dim * (action_dim + 1) // 2

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.value_head = nn.Linear(hidden_dim, 1)
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())
        self.l_head = nn.Linear(hidden_dim, self.tril_size)

    def forward(self, state, action=None):
        feat = self.feature(state)
        V = self.value_head(feat).squeeze(-1)
        mu = self.mu_head(feat)
        mu = self.action_low + (mu + 1) * 0.5 * (self.action_high - self.action_low)

        if action is None:
            return V, mu

        l_entries = self.l_head(feat)
        L = self._construct_L(l_entries)
        diff = (action - mu).unsqueeze(-1)
        LT_diff = torch.bmm(L.transpose(1, 2), diff)
        advantage = -0.5 * LT_diff.pow(2).sum(dim=(1, 2))
        Q = V + advantage
        return Q, V, mu

    def _construct_L(self, l_entries):
        batch = l_entries.shape[0]
        L = torch.zeros(batch, self.action_dim, self.action_dim, device=l_entries.device)
        idx = 0
        for i in range(self.action_dim):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = torch.exp(l_entries[:, idx])
                else:
                    L[:, i, j] = l_entries[:, idx]
                idx += 1
        return L

    def get_action(self, state):
        _, mu = self.forward(state)
        return mu


class ReplayBuffer:
    def __init__(self, cap, sd, ad):
        self.cap=cap; self.sz=0; self.p=0
        self.s=np.zeros((cap,sd),np.float32); self.a=np.zeros((cap,ad),np.float32)
        self.r=np.zeros(cap,np.float32); self.ns=np.zeros((cap,sd),np.float32)
        self.d=np.zeros(cap,np.float32)
    def push(self,s,a,r,ns,d):
        self.s[self.p]=s;self.a[self.p]=a;self.r[self.p]=r;self.ns[self.p]=ns;self.d[self.p]=float(d)
        self.p=(self.p+1)%self.cap;self.sz=min(self.sz+1,self.cap)
    def sample(self,n):
        i=np.random.randint(0,self.sz,n)
        return (torch.FloatTensor(self.s[i]),torch.FloatTensor(self.a[i]),
                torch.FloatTensor(self.r[i]),torch.FloatTensor(self.ns[i]),torch.FloatTensor(self.d[i]))
    def __len__(self): return self.sz


class NAFAgent:
    def __init__(self, sd, ad, a_low=-1.0, a_high=1.0, lr=1e-3, gamma=0.99,
                 tau=0.005, batch_size=128, buf_cap=100000, noise_std=0.1):
        self.gamma=gamma; self.tau=tau; self.batch_size=batch_size
        self.noise_std=noise_std; self.a_low=a_low; self.a_high=a_high

        self.online = NAFNetwork(sd, ad, action_low=a_low, action_high=a_high)
        self.target = NAFNetwork(sd, ad, action_low=a_low, action_high=a_high)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buf = ReplayBuffer(buf_cap, sd, ad)

    def act(self, state, training=True):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu = self.online.get_action(s).squeeze(0).numpy()
        if training:
            noise = np.random.normal(0, self.noise_std, size=mu.shape)
            mu = np.clip(mu + noise, self.a_low, self.a_high)
        return mu

    def store(self, s, a, r, ns, d):
        self.buf.push(s, a, r, ns, d)

    def update(self):
        if len(self.buf) < self.batch_size:
            return 0.0
        s, a, r, ns, d = self.buf.sample(self.batch_size)

        Q, V, mu = self.online(s, a)
        with torch.no_grad():
            next_V, _ = self.target(ns)
            targets = r + (1 - d) * self.gamma * next_V

        loss = nn.functional.mse_loss(Q, targets)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)
        return loss.item()


def demo_naf():
    print("=" * 60)
    print("NAF Demo")
    print("=" * 60)

    # --- Architecture demo ---
    print("\n--- NAF Architecture ---")
    sd, ad = 8, 2
    net = NAFNetwork(sd, ad, action_low=-1, action_high=1)
    print(f"  Params: {sum(p.numel() for p in net.parameters()):,}")

    s = torch.randn(4, sd)
    V, mu = net(s)
    print(f"  V(s) shape: {V.shape}, μ(s) shape: {mu.shape}")
    print(f"  μ(s)[0]: {mu[0].detach().numpy().round(3)}")

    a = torch.randn(4, ad).clamp(-1, 1)
    Q, V2, mu2 = net(s, a)
    print(f"  Q(s,a) shape: {Q.shape}")
    print(f"  V = max_a Q: V={V[0].item():.3f}, Q(s,μ)={net(s, mu)[0][0].item():.3f}")

    # --- Quadratic advantage visualization ---
    print("\n--- Quadratic Advantage (1D action) ---")
    net1d = NAFNetwork(4, 1, action_low=-1, action_high=1)
    state = torch.randn(1, 4)
    actions = torch.linspace(-1, 1, 11).unsqueeze(1)
    states = state.expand(11, -1)
    Qs, Vs, mus = net1d(states, actions)
    print(f"  μ(s) = {mus[0].item():.3f}")
    for i in range(0, 11, 2):
        print(f"  a={actions[i].item():>5.2f}: Q={Qs[i].item():>7.3f}, "
              f"A={Qs[i].item() - Vs[i].item():>7.3f}")

    # --- Training on Pendulum ---
    print("\n--- NAF Training on Pendulum-v1 ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    env = gym.make('Pendulum-v1')
    agent = NAFAgent(3, 1, a_low=-2.0, a_high=2.0, lr=1e-3, noise_std=0.3)
    rewards = []

    for ep in range(200):
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
    print("\nNAF demo complete!")


if __name__ == "__main__":
    demo_naf()
