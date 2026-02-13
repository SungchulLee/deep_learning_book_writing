#!/usr/bin/env python3
"""
Dueling DQN - Separate value and advantage streams
Paper: "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
Authors: Ziyu Wang et al.
Key idea:
  - Learn V(s) and A(s,a) separately, then combine to Q(s,a)
  - Helps when many actions have similar value

Combination:
  Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )

File: appendix/rl/dueling_dqn.py
"""

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Dueling network:
      shared trunk -> value head + advantage head
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # Value stream outputs a single scalar V(s)
        self.value = nn.Linear(hidden, 1)

        # Advantage stream outputs A(s,a) for all actions
        self.adv = nn.Linear(hidden, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)

        V = self.value(h)           # (B, 1)
        A = self.adv(h)             # (B, A)

        # Center advantages to keep Q identifiable (otherwise V/A not unique)
        A_centered = A - A.mean(dim=1, keepdim=True)

        Q = V + A_centered          # (B, A)
        return Q


if __name__ == "__main__":
    net = DuelingQNetwork(obs_dim=8, num_actions=4)
    x = torch.randn(3, 8)
    q = net(x)
    print("Q:", q.shape)  # (3, 4)
