#!/usr/bin/env python3
"""
Double DQN - Reducing overestimation bias in Q-learning
Paper: "Deep Reinforcement Learning with Double Q-learning" (2016)
Authors: Hado van Hasselt, Arthur Guez, David Silver
Key idea:
  - Use online network to *select* action
  - Use target network to *evaluate* that action

Target:
  a* = argmax_a Q_online(s', a)
  y  = r + gamma * (1-done) * Q_target(s', a*)

File: appendix/rl/double_dqn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_dqn_td_loss(q_online: nn.Module, q_target: nn.Module, batch, gamma: float = 0.99):
    """
    Compute Double DQN TD loss.

    Difference vs DQN:
      - DQN uses max_a Q_target(s', a)
      - Double DQN uses argmax from online, value from target
    """
    s, a, r, s2, done = batch

    # Current Q(s,a)
    q_values = q_online(s)  # (B, A)
    q_sa = q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Action selection by online network
        a_star = q_online(s2).argmax(dim=1)  # (B,)

        # Action evaluation by target network
        q_next_target = q_target(s2)  # (B, A)
        q_s2_astar = q_next_target.gather(1, a_star.unsqueeze(1)).squeeze(1)

        target = r + gamma * (1.0 - done) * q_s2_astar

    loss = F.mse_loss(q_sa, target)
    return loss
