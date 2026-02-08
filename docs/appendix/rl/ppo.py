#!/usr/bin/env python3
"""
PPO - Proximal Policy Optimization
Paper: "Proximal Policy Optimization Algorithms" (2017)
Authors: John Schulman et al.
Key idea:
  - Policy gradient with a *clipped surrogate objective* to prevent large updates
  - Often uses GAE advantages and minibatch epochs

Clipped objective:
  r_t = pi(a|s) / pi_old(a|s)
  L_clip = E[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]

File: appendix/rl/ppo.py
Note: Educational implementation of PPO loss (discrete actions).
"""

import torch
import torch.nn.functional as F


def ppo_loss(
    logits_new,        # (B, A) current policy logits
    logits_old,        # (B, A) behavior/old policy logits (frozen)
    actions,           # (B,)
    advantages,        # (B,)
    returns,           # (B,)
    values,            # (B,) critic values from current network
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """
    Compute PPO losses:
      - clipped policy loss
      - value loss
      - entropy bonus
    """
    # Compute log probs under new and old policies
    logp_new = F.log_softmax(logits_new, dim=1)
    logp_old = F.log_softmax(logits_old, dim=1)

    logp_new_a = logp_new.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    logp_old_a = logp_old.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    # Importance ratio r_t = exp(log pi_new - log pi_old)
    ratio = torch.exp(logp_new_a - logp_old_a)

    # Advantages are typically standardized; detach so actor doesn't backprop into advantage
    A = advantages.detach()

    # Clipped surrogate objective
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Value function loss
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (encourage exploration)
    p_new = torch.softmax(logits_new, dim=1)
    entropy = -(p_new * logp_new).sum(dim=1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}
