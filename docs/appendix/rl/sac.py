#!/usr/bin/env python3
"""
SAC - Soft Actor-Critic (continuous control)
Paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (2018)
Authors: Tuomas Haarnoja et al.
Key idea:
  - Actor maximizes expected return + entropy (exploration)
  - Two Q networks (double Q) to reduce positive bias
  - Target Q networks for stability

File: appendix/rl/sac.py
Note: Educational reference (networks + core target computations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    """
    Stochastic actor for continuous actions:
      a ~ N(mu(s), sigma(s)), then typically tanh-squashed to bounds [-1,1].
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)  # numerical stability
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        """
        Reparameterized sampling:
          a = tanh(mu + std * eps)
        Also returns log_prob(a) adjusted for tanh squashing (omitted for brevity).
        """
        mu, std = self.forward(obs)
        eps = torch.randn_like(std)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        # For a real SAC implementation, compute log_prob with tanh correction.
        log_prob = None
        return a, log_prob


class QNetwork(nn.Module):
    """Critic Q(s,a) for continuous actions."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x).squeeze(1)  # (B,)


def sac_target(q1_tgt, q2_tgt, policy, s2, r, done, gamma=0.99, alpha=0.2):
    """
    Compute SAC target:
      a' ~ pi(s')
      y = r + gamma*(1-done) * ( min(Q1_tgt(s',a'), Q2_tgt(s',a')) - alpha * log pi(a'|s') )

    Here, log pi term is omitted for brevity; include it in a full implementation.
    """
    with torch.no_grad():
        a2, logp2 = policy.sample(s2)
        q1v = q1_tgt(s2, a2)
        q2v = q2_tgt(s2, a2)
        qmin = torch.min(q1v, q2v)

        # If logp2 is None (as in this educational code), ignore entropy term.
        if logp2 is None:
            backup = qmin
        else:
            backup = qmin - alpha * logp2

        y = r + gamma * (1.0 - done) * backup
    return y
