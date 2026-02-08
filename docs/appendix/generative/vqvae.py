#!/usr/bin/env python3
"""
VQ-VAE - Vector Quantized Variational Autoencoder
Paper: "Neural Discrete Representation Learning" (2017)
Key idea:
  - Replace continuous latent space with discrete codebook entries
  - Enables powerful autoregressive priors

File: appendix/generative/vqvae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Codebook for discrete latent variables."""
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # z: (B, D)
        distances = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices)
        return z_q, indices


class VQVAE(nn.Module):
    """Minimal VQ-VAE."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Linear(784, latent_dim)
        self.vq = VectorQuantizer()
        self.decoder = nn.Linear(latent_dim, 784)

    def forward(self, x):
        z = self.encoder(x)
        z_q, _ = self.vq(z)
        recon = torch.sigmoid(self.decoder(z_q))
        return recon
