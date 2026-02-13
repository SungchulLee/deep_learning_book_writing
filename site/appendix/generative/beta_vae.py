#!/usr/bin/env python3
"""
Beta-VAE - Variational Autoencoder with Disentanglement Control
Paper: "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)
Key idea:
  - Add a β coefficient to the KL term
  - Larger β → stronger disentanglement, worse reconstruction

File: appendix/generative/beta_vae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encodes input x into latent mean and log-variance."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(784, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc(x))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """Decodes latent variable z back to input space."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256)
        self.out = nn.Linear(256, 784)

    def forward(self, z):
        h = F.relu(self.fc(z))
        return torch.sigmoid(self.out(h))


class BetaVAE(nn.Module):
    """β-VAE model."""
    def __init__(self, latent_dim=16, beta=4.0):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
