#!/usr/bin/env python3
"""
Loss Functions - Common deep learning losses
Includes:
  - Cross-Entropy (classification)
  - MSE (regression)
  - BCEWithLogits (binary)
  - Focal Loss (dense detection)
  - KL divergence helper (VAE-like)

File: appendix/utils/losses.py
Note: Educational implementations for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Focal loss for binary classification (often extended to multi-class).

    logits:  (B,) or (B,1) raw scores
    targets: (B,) in {0,1}

    FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is model probability of the true class.
    """
    targets = targets.float()

    # Compute probability with sigmoid
    p = torch.sigmoid(logits)

    # p_t = p if y=1 else (1-p)
    p_t = p * targets + (1 - p) * (1 - targets)

    # alpha_t = alpha if y=1 else (1-alpha)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    # Standard BCE loss term: -log(p_t)
    ce = -torch.log(p_t.clamp(min=1e-8))

    loss = alpha_t * ((1 - p_t) ** gamma) * ce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def kl_normal(mu, logvar):
    """
    KL divergence between N(mu, sigma^2) and N(0,1) per sample.

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
