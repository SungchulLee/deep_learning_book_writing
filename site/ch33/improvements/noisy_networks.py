"""
33.2.4 Noisy Networks
======================

NoisyLinear layer and NoisyNet Q-Network for learned exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from typing import List, Tuple
import math


# ---------------------------------------------------------------------------
# Noisy Linear Layer (Factorized Gaussian)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet linear layer.
    
    y = (mu_w + sigma_w * eps_w) @ x + (mu_b + sigma_b * eps_b)
    where eps is factorized: eps_ij = f(eps_i) * f(eps_j), f(x) = sign(x)*sqrt(|x|)
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # Factorized noise buffers (not parameters)
        self.register_buffer('eps_in', torch.zeros(in_features))
        self.register_buffer('eps_out', torch.zeros(out_features))

        # Initialize
        bound = 1 / math.sqrt(in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma_init / math.sqrt(in_features))
        self.sigma_bias.data.fill_(sigma_init / math.sqrt(out_features))

        self.reset_noise()

    @staticmethod
    def _factorized_noise(size: int) -> torch.Tensor:
        """f(x) = sign(x) * sqrt(|x|)"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resample factorized noise."""
        eps_i = self._factorized_noise(self.in_features)
        eps_j = self._factorized_noise(self.out_features)
        self.eps_in.copy_(eps_i)
        self.eps_out.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Factorized noise: eps_w = eps_out^T @ eps_in
            eps_w = self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0)
            eps_b = self.eps_out
            weight = self.mu_weight + self.sigma_weight * eps_w
            bias = self.mu_bias + self.sigma_bias * eps_b
        else:
            # No noise during evaluation
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)

    @property
    def noise_magnitude(self) -> float:
        """Average magnitude of sigma parameters (for monitoring)."""
        return (self.sigma_weight.abs().mean() + self.sigma_bias.abs().mean()).item() / 2


# ---------------------------------------------------------------------------
# Independent Gaussian Noisy Linear (for comparison)
# ---------------------------------------------------------------------------

class IndependentNoisyLinear(nn.Module):
    """Independent Gaussian noise per weight (more expensive)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        self.register_buffer('eps_weight', torch.zeros(out_features, in_features))
        self.register_buffer('eps_bias', torch.zeros(out_features))

        bound = 1 / math.sqrt(in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma_init)
        self.sigma_bias.data.fill_(sigma_init)

        self.reset_noise()

    def reset_noise(self):
        self.eps_weight.normal_()
        self.eps_bias.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.eps_weight
            bias = self.mu_bias + self.sigma_bias * self.eps_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# NoisyNet Q-Network
# ---------------------------------------------------------------------------

class NoisyQNetwork(nn.Module):
    """Q-Network with NoisyLinear layers for exploration."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                 sigma_init: float = 0.5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU())
        # Replace last layers with noisy layers
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        feat = F.relu(self.noisy1(feat))
        return self.noisy2(feat)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    @property
    def noise_magnitude(self) -> float:
        return (self.noisy1.noise_magnitude + self.noisy2.noise_magnitude) / 2


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap, sd):
        self.cap=cap;self.sz=0;self.p=0
        self.s=np.zeros((cap,sd),np.float32);self.a=np.zeros(cap,np.int64)
        self.r=np.zeros(cap,np.float32);self.ns=np.zeros((cap,sd),np.float32)
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
# Demo
# ---------------------------------------------------------------------------

def demo_noisy_networks():
    print("=" * 60)
    print("Noisy Networks Demo")
    print("=" * 60)

    env = gym.make('CartPole-v1')
    sd = env.observation_space.shape[0]
    ad = env.action_space.n

    # --- NoisyLinear layer analysis ---
    print("\n--- NoisyLinear Layer ---")
    noisy = NoisyLinear(sd, 64)
    x = torch.randn(5, sd)

    outputs = []
    for _ in range(10):
        noisy.reset_noise()
        outputs.append(noisy(x).detach())
    outputs = torch.stack(outputs)
    print(f"  Output variance across noise samples: {outputs.var(dim=0).mean():.4f}")
    print(f"  Sigma magnitude: {noisy.noise_magnitude:.4f}")

    # --- Training with NoisyNet (no epsilon-greedy) ---
    print("\n--- NoisyNet DQN Training ---")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    online = NoisyQNetwork(sd, ad)
    target = NoisyQNetwork(sd, ad)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=1e-3)
    buf = ReplayBuffer(50000, sd)

    rewards = []
    noise_mags = []
    step = 0

    for ep in range(250):
        s, _ = env.reset(); total = 0; done = False
        online.reset_noise()  # Reset noise per episode

        while not done:
            step += 1
            # No epsilon-greedy! Noise provides exploration
            with torch.no_grad():
                online.train()
                a = online(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buf.push(s, a, r, ns, float(done))

            if len(buf) >= 500:
                st, at, rt, nst, dt = buf.sample(64)
                online.reset_noise()
                target.reset_noise()
                q = online(st).gather(1, at.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = target(nst).max(1)[0]
                    tgt = rt + (1 - dt) * 0.99 * nq
                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()

                if step % 200 == 0:
                    target.load_state_dict(online.state_dict())

            s = ns; total += r

        rewards.append(total)
        noise_mags.append(online.noise_magnitude)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}: avg50={np.mean(rewards[-50:]):.1f}, "
                  f"σ={online.noise_magnitude:.4f}")

    env.close()

    # --- Noise evolution ---
    print("\n--- Noise Magnitude Evolution ---")
    for i in range(0, len(noise_mags), 50):
        end = min(i + 50, len(noise_mags))
        avg_sigma = np.mean(noise_mags[i:end])
        print(f"  Episodes {i+1}-{end}: avg σ = {avg_sigma:.4f}")

    print(f"\n  Initial σ ≈ {noise_mags[0]:.4f}")
    print(f"  Final σ ≈ {noise_mags[-1]:.4f}")
    print(f"  σ reduction: {(1 - noise_mags[-1]/noise_mags[0])*100:.1f}%")

    print("\nNoisy networks demo complete!")


if __name__ == "__main__":
    demo_noisy_networks()
