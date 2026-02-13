#!/usr/bin/env python3
"""
A2C / A3C - Advantage Actor-Critic
Papers:
  - A3C: "Asynchronous Methods for Deep Reinforcement Learning" (2016)
  - A2C: synchronous variant commonly used in practice
Key idea:
  - Actor outputs policy π(a|s)
  - Critic outputs value V(s)
  - Use advantage: A = R - V(s) (or GAE) to update actor

File: appendix/rl/a2c_a3c.py
Note: Educational implementation of the *losses* (policy loss + value loss + entropy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Shared backbone with two heads:
      - policy logits over discrete actions
      - state value estimate
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.backbone(obs)
        logits = self.policy_head(h)            # (B, A)
        value = self.value_head(h).squeeze(1)   # (B,)
        return logits, value


def a2c_loss(logits, values, actions, returns, entropy_coef=0.01, value_coef=0.5):
    """
    Compute A2C/A3C losses.

    Inputs:
      logits:  (B, A) policy logits
      values:  (B,) critic V(s)
      actions: (B,) actions taken
      returns: (B,) empirical returns (e.g., n-step)

    Advantage:
      adv = returns - values

    Loss:
      policy_loss = - E[ log pi(a|s) * adv ]
      value_loss  = MSE(values, returns)
      entropy_bonus encourages exploration
    """
    # Log-probabilities of the chosen actions
    logp = F.log_softmax(logits, dim=1)  # (B, A)
    logp_a = logp.gather(1, actions.long().unsqueeze(1)).squeeze(1)  # (B,)

    # Advantage (stop gradient through advantage when updating actor)
    adv = (returns - values).detach()

    policy_loss = -(logp_a * adv).mean()

    value_loss = F.mse_loss(values, returns)

    # Entropy: -sum p log p (higher entropy = more exploration)
    p = F.softmax(logits, dim=1)
    entropy = -(p * logp).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}
