#!/usr/bin/env python3
"""
Pix2Pix - Paired Image-to-Image Translation
Paper: "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
Key idea:
  - Conditional GAN
  - Requires paired (x, y) training data

File: appendix/generative/pix2pix.py
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Simple conditional generator."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class Discriminator(nn.Module):
    """Conditional discriminator."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
