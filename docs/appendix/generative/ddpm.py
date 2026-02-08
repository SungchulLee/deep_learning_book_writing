#!/usr/bin/env python3
"""
DDPM - Denoising Diffusion Probabilistic Models
Paper: "Denoising Diffusion Probabilistic Models" (2020)
Key idea:
  - Gradually add Gaussian noise
  - Learn to reverse the diffusion process

File: appendix/generative/ddpm.py
"""

import torch
import torch.nn as nn


class Denoiser(nn.Module):
    """Predicts noise ε given x_t."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        return self.net(x)


class DDPM(nn.Module):
    """Simplified DDPM."""
    def __init__(self):
        super().__init__()
        self.denoiser = Denoiser()

    def forward(self, x_t):
        eps_hat = self.denoiser(x_t)
        return eps_hat
