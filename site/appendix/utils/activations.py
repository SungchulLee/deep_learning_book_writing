#!/usr/bin/env python3
"""
Activation Functions - Common nonlinearities
Includes:
  - ReLU
  - LeakyReLU
  - GELU
  - SiLU (Swish)
  - GLU / SwiGLU (gated activations used in transformers/LLMs)

File: appendix/utils/activations.py
Note: Educational reference implementations + usage notes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """
    Gated Linear Unit:
      GLU(x) = (xW) * sigmoid(xV)
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        # Split into value and gate
        v, g = self.fc(x).chunk(2, dim=-1)
        return v * torch.sigmoid(g)


class SwiGLU(nn.Module):
    """
    SwiGLU (used in LLaMA / PaLM-like FFNs):
      SwiGLU(x) = (SiLU(xW1) * (xW3)) W2
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# Note:
# For standard activations, prefer built-ins:
#   F.relu(x), nn.ReLU()
#   F.gelu(x), nn.GELU()
#   F.silu(x), nn.SiLU()
