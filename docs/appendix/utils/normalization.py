#!/usr/bin/env python3
"""
Normalization Layers - Common variants used in deep learning
Includes:
  - BatchNorm1d/2d (PyTorch builtin wrapper)
  - LayerNorm (PyTorch builtin wrapper)
  - GroupNorm (PyTorch builtin wrapper)
  - RMSNorm (commonly used in modern LLMs)

File: appendix/utils/normalization.py
Note: Educational, with comments explaining when to use each.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)

    Unlike LayerNorm, RMSNorm:
      - does NOT subtract mean
      - only divides by RMS (sqrt(mean(x^2)))

    Often used in LLaMA-like models.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# Note:
# - BatchNorm, LayerNorm, GroupNorm are in PyTorch.
# - You typically import and use:
#     nn.BatchNorm2d(C)
#     nn.LayerNorm(D)
#     nn.GroupNorm(num_groups, C)
#
# This file adds RMSNorm plus short usage notes.
