#!/usr/bin/env python3
"""
DQN - Deep Q-Network
Paper: "Human-level control through deep reinforcement learning" (2015)
Authors: Volodymyr Mnih et al.
Key idea:
  - Approximate Q(s,a) with a neural network
  - Train with TD target using a *target network*
  - Use experience replay to break correlation

File: appendix/rl/dqn.py
Note: Educational reference: model + replay + TD loss computation (no full env loop).
"""

from dataclasses import dataclass
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Simple MLP Q(s,a) approximator for discrete actions."""
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim)
        return self.net(obs)  # (B, num_actions)


@dataclass
class Transition:
    """One experience tuple stored in replay buffer."""
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """Fixed-size FIFO replay buffer."""
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        # Stack fields into batch tensors
        s = torch.stack([b.s for b in batch], dim=0)
        a = torch.stack([b.a for b in batch], dim=0)
        r = torch.stack([b.r for b in batch], dim=0)
        s2 = torch.stack([b.s2 for b in batch], dim=0)
        done = torch.stack([b.done for b in batch], dim=0)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buf)


def dqn_td_loss(q_net: nn.Module, target_net: nn.Module, batch, gamma: float = 0.99):
    """
    Compute DQN TD loss.

    For each transition (s,a,r,s',done):
      target = r + gamma * (1-done) * max_a' Q_target(s', a')
      loss = MSE( Q(s,a), target )

    Note:
      - a is discrete action index (shape: (B,))
      - done is 1 if terminal else 0
    """
    s, a, r, s2, done = batch

    # Current Q-values for all actions: (B, A)
    q_values = q_net(s)

    # Select Q(s,a) using gather:
    # a must be shape (B,1) for gather on dim=1
    q_sa = q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)  # (B,)

    # Compute target using target network (no grad)
    with torch.no_grad():
        q_next = target_net(s2)                   # (B, A)
        max_q_next = q_next.max(dim=1).values     # (B,)
        target = r + gamma * (1.0 - done) * max_q_next

    loss = F.mse_loss(q_sa, target)
    return loss


if __name__ == "__main__":
    # Toy smoke test (no environment)
    obs_dim, num_actions = 8, 4
    q = QNetwork(obs_dim, num_actions)
    tgt = QNetwork(obs_dim, num_actions)

    # Fake batch
    B = 5
    s = torch.randn(B, obs_dim)
    a = torch.randint(0, num_actions, (B,))
    r = torch.randn(B)
    s2 = torch.randn(B, obs_dim)
    done = torch.randint(0, 2, (B,), dtype=torch.float32)

    loss = dqn_td_loss(q, tgt, (s, a, r, s2, done))
    print("loss:", float(loss))
