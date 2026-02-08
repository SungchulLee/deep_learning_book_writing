#!/usr/bin/env python3
"""
Stable Diffusion (Conceptual)
Paper: "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)
Key idea:
  - Diffusion in *latent space*
  - Text-conditioned via cross-attention

File: appendix/generative/stable_diffusion.py
"""

import torch
import torch.nn as nn


class LatentUNet(nn.Module):
    """Noise predictor in latent space."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, z, text_emb):
        # text_emb would condition via cross-attention in real models
        return self.net(z)


class StableDiffusion(nn.Module):
    """Conceptual Stable Diffusion."""
    def __init__(self):
        super().__init__()
        self.unet = LatentUNet()

    def forward(self, z, text_emb):
        return self.unet(z, text_emb)
