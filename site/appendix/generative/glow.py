#!/usr/bin/env python3
"""
Glow - Flow-based Generative Model with Invertible 1x1 Convolutions
Paper: "Glow: Generative Flow with Invertible 1x1 Convolutions" (2018)
Key idea:
  - Improve RealNVP with learned invertible 1x1 convolutions

File: appendix/generative/glow.py
"""

import torch
import torch.nn as nn


class Invertible1x1Conv(nn.Module):
    """Invertible linear transform."""
    def __init__(self, dim=784):
        super().__init__()
        W = torch.qr(torch.randn(dim, dim))[0]
        self.weight = nn.Parameter(W)

    def forward(self, x):
        return x @ self.weight
