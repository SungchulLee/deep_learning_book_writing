#!/usr/bin/env python3
"""
RealNVP - Real-valued Non-Volume Preserving Flows
Paper: "Density Estimation using Real NVP" (2017)
Key idea:
  - Invertible transformations
  - Exact likelihood via change-of-variables

File: appendix/generative/realnvp.py
"""

import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """Affine coupling layer."""
    def __init__(self, dim=784):
        super().__init__()
        self.scale = nn.Linear(dim // 2, dim // 2)
        self.shift = nn.Linear(dim // 2, dim // 2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale(x1)
        t = self.shift(x1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=1)
