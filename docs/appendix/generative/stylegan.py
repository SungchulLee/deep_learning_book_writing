#!/usr/bin/env python3
"""
StyleGAN - Style-Based Generator Architecture
Paper: "A Style-Based Generator Architecture for GANs" (2019)
Key idea:
  - Separate latent mapping network
  - Style modulation at each layer

File: appendix/generative/stylegan.py
"""

import torch
import torch.nn as nn


class MappingNetwork(nn.Module):
    """Maps latent z → style w."""
    def __init__(self, z_dim=512, w_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim),
        )

    def forward(self, z):
        return self.net(z)


class StyleGenerator(nn.Module):
    """Simplified StyleGAN generator."""
    def __init__(self, w_dim=512):
        super().__init__()
        self.mapping = MappingNetwork()
        self.fc = nn.Linear(w_dim, 784)

    def forward(self, z):
        w = self.mapping(z)
        img = torch.sigmoid(self.fc(w))
        return img
