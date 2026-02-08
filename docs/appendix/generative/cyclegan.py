#!/usr/bin/env python3
"""
CycleGAN - Unpaired Image-to-Image Translation
Paper: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)
Key idea:
  - Two generators (A→B, B→A)
  - Cycle-consistency loss enforces invertibility

File: appendix/generative/cyclegan.py
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator mapping between domains."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class CycleGAN(nn.Module):
    """CycleGAN with two generators."""
    def __init__(self):
        super().__init__()
        self.G_AB = Generator()
        self.G_BA = Generator()

    def forward(self, xA, xB):
        fakeB = self.G_AB(xA)
        recA = self.G_BA(fakeB)
        return fakeB, recA
