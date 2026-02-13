#!/usr/bin/env python3
"""
================================================================================
VAE - Variational Autoencoder
================================================================================

Paper: "Auto-Encoding Variational Bayes" (ICLR 2014)
Authors: Diederik P. Kingma, Max Welling
Link: https://arxiv.org/abs/1312.6114

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
The Variational Autoencoder introduced a principled probabilistic framework for 
learning latent representations. It bridges neural networks with variational 
inference, enabling both generation and representation learning.

Key Contributions:
- **Reparameterization Trick**: Enables backpropagation through stochastic nodes
- **Variational Inference in Neural Networks**: Makes Bayesian deep learning practical
- **Generative Modeling**: One of the foundational deep generative models
- **Foundation for Later Work**: VAE-GAN, β-VAE, VQ-VAE, DALL-E

================================================================================
THEORETICAL FOUNDATIONS
================================================================================

Goal: Learn a generative model p(x) for data x by introducing latent variables z.

Generative Story:
1. Sample latent code: z ~ p(z) = N(0, I)
2. Generate data: x ~ p(x|z) parameterized by decoder

Problem: Computing p(x) = ∫ p(x|z)p(z) dz is intractable!

Solution: Variational Inference
────────────────────────────────────────────────────────────────────────────────
Instead of computing true posterior p(z|x), approximate it with q(z|x).

Introduce encoder qφ(z|x) to approximate p(z|x):
    qφ(z|x) = N(μφ(x), σ²φ(x))

The encoder outputs μ and σ (or log σ²) for each input x.

================================================================================
EVIDENCE LOWER BOUND (ELBO)
================================================================================

Key Result: For any qφ(z|x), the log evidence satisfies:

    log p(x) = ELBO + KL(qφ(z|x) || p(z|x))
    
Since KL ≥ 0, we have:
    
    log p(x) ≥ ELBO = Eq[log p(x|z)] - KL(qφ(z|x) || p(z))
                      ════════════════   ═══════════════════
                      Reconstruction      Regularization
                         Term               (KL Divergence)

ELBO Interpretation:
────────────────────────────────────────────────────────────────────────────────
1. Reconstruction Term: How well does the decoder reconstruct x from z?
   - Maximize log p(x|z) ≈ Minimize reconstruction error
   - For Gaussian decoder: -||x - x̂||² (MSE)
   - For Bernoulli decoder: Binary cross-entropy

2. KL Term: How close is qφ(z|x) to the prior p(z)?
   - Regularizes the latent space
   - Prevents encoder from ignoring the prior
   - For Gaussian prior and posterior:
     KL(N(μ, σ²) || N(0, I)) = -½ Σ(1 + log(σ²) - μ² - σ²)

================================================================================
THE REPARAMETERIZATION TRICK (Key Innovation!)
================================================================================

Problem: Can't backprop through sampling z ~ qφ(z|x)

Solution: Rewrite sampling as deterministic + noise

Instead of: z ~ N(μ, σ²)
Write:      z = μ + σ ⊙ ε,  where ε ~ N(0, I)

Now gradients can flow through μ and σ!

                 ┌─────────────┐
                 │   Encoder   │
                 │  qφ(z|x)    │
                 └──────┬──────┘
                        │
                   ┌────┴────┐
                   │         │
                   ▼         ▼
                 ┌───┐     ┌───────┐
                 │ μ │     │ log σ²│
                 └─┬─┘     └───┬───┘
                   │           │
                   │    ┌──────┴──────┐
                   │    │  σ = exp(   │
                   │    │  log σ² / 2)│
                   │    └──────┬──────┘
                   │           │
                   ▼           ▼
              ┌─────────────────────────────┐
              │  z = μ + σ ⊙ ε              │ ← ε ~ N(0, I) (no gradient)
              │  (reparameterization trick) │
              └──────────────┬──────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   Decoder   │
                      │  pθ(x|z)    │
                      └─────────────┘

================================================================================
VAE ARCHITECTURE
================================================================================

    ┌─────────────────────────────────────────────────────────────────┐
    │                         ENCODER                                  │
    │  ┌───────────┐    ┌───────────┐    ┌─────────┐    ┌─────────┐  │
    │  │  Input x  │───▶│  FC Layers│───▶│   μ     │───▶│    z    │  │
    │  │  (784)    │    │  (512,256)│    │ (20)    │    │  (20)   │  │
    │  └───────────┘    └───────────┘    └─────────┘    │         │  │
    │                                     ┌─────────┐    │         │  │
    │                                     │ log σ²  │───▶│ reparam │  │
    │                                     │ (20)    │    └─────────┘  │
    │                                     └─────────┘                  │
    └─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                         DECODER                                  │
    │  ┌───────────┐    ┌───────────┐    ┌─────────────┐              │
    │  │    z      │───▶│  FC Layers│───▶│   Output x̂  │              │
    │  │   (20)    │    │ (256,512) │    │    (784)    │              │
    │  └───────────┘    └───────────┘    └─────────────┘              │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
LOSS FUNCTION
================================================================================

VAE Loss = Reconstruction Loss + β × KL Divergence

Reconstruction Loss (for Bernoulli likelihood):
    L_recon = BCE(x, x̂) = -Σ[x log(x̂) + (1-x) log(1-x̂)]

KL Divergence (closed-form for Gaussian):
    L_KL = -½ Σ_j [1 + log(σ²_j) - μ²_j - σ²_j]

Total Loss:
    L = L_recon + β × L_KL

β-VAE (Higgins et al., 2017):
- β > 1: More disentangled but blurrier reconstructions
- β = 1: Standard VAE
- β < 1: Better reconstructions but less regularized latent space

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch09: Autoencoders (VAE architecture)
- Ch09: VAE (ELBO, reparameterization trick)
- Ch13: Variational Inference (ELBO derivation)
- Ch14: Diffusion Models (comparison with VAE)

Related architectures:
- Autoencoder: Non-probabilistic version (autoencoder.py)
- DCGAN: Adversarial training (dcgan.py)
- VQ-VAE: Discrete latents (advanced topic)

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAE(nn.Module):
    """
    Variational Autoencoder for MNIST-like images
    
    Implements the classic VAE with Gaussian encoder and Bernoulli decoder.
    
    Args:
        input_dim: Flattened input dimension (e.g., 784 for 28×28 images)
        hidden_dim: Hidden layer dimension. Default: 512
        latent_dim: Latent space dimension. Default: 20
    
    Example:
        >>> model = VAE(input_dim=784, latent_dim=20)
        >>> x = torch.randn(32, 784)
        >>> x_recon, mu, logvar = model(x)
        >>> loss = model.loss_function(x_recon, x, mu, logvar)
    
    Shape:
        - Input: (N, input_dim)
        - Output: (x_recon, mu, logvar) with shapes (N, input_dim), (N, latent_dim), (N, latent_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        latent_dim: int = 20
    ):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ====================================================================
        # ENCODER: Maps input x to latent parameters (μ, log σ²)
        # ====================================================================
        # Inference network: qφ(z|x) = N(μφ(x), σ²φ(x))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Separate heads for mean and log-variance
        # Using log-variance ensures σ² > 0 (exp is always positive)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # ====================================================================
        # DECODER: Maps latent z to reconstruction x̂
        # ====================================================================
        # Generative network: pθ(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Bernoulli likelihood: outputs in [0, 1]
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input tensor (N, input_dim)
            
        Returns:
            mu: Mean of q(z|x) with shape (N, latent_dim)
            logvar: Log-variance of q(z|x) with shape (N, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick
        
        Sample z = μ + σ ⊙ ε where ε ~ N(0, I)
        
        This allows gradients to flow through the sampling operation!
        
        Args:
            mu: Mean of q(z|x)
            logvar: Log-variance of q(z|x)
            
        Returns:
            z: Sampled latent vector
        """
        # During training, sample from q(z|x)
        # During eval, can optionally just use mu (deterministic)
        if self.training:
            # std = exp(0.5 * logvar) = sqrt(var)
            std = torch.exp(0.5 * logvar)
            
            # Sample ε ~ N(0, I)
            eps = torch.randn_like(std)
            
            # z = μ + σ ⊙ ε
            return mu + eps * std
        else:
            # During evaluation, use mean for deterministic behavior
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction
        
        Args:
            z: Latent vector (N, latent_dim)
            
        Returns:
            x_recon: Reconstructed input (N, input_dim)
        """
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, sample, decode
        
        Args:
            x: Input tensor (N, input_dim) or (N, C, H, W)
            
        Returns:
            x_recon: Reconstructed input
            mu: Mean of q(z|x)
            logvar: Log-variance of q(z|x)
        """
        # Flatten input if necessary
        x_flat = x.view(-1, self.input_dim)
        
        # Encode to latent distribution
        mu, logvar = self.encode(x_flat)
        
        # Sample z using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruction
        x_recon = self.decode(z)
        
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate samples from the prior p(z) = N(0, I)
        
        Args:
            num_samples: Number of samples to generate
            device: Device to put samples on
            
        Returns:
            Generated samples (num_samples, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input (deterministic, using mean)
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstruction
        """
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
        return x_recon
    
    @staticmethod
    def loss_function(
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE Loss = Reconstruction Loss + β × KL Divergence
        
        Args:
            x_recon: Reconstructed input
            x: Original input
            mu: Mean of q(z|x)
            logvar: Log-variance of q(z|x)
            beta: Weight for KL term (β=1 for standard VAE, β>1 for β-VAE)
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
        """
        # Flatten for loss computation
        x_flat = x.view(-1, x_recon.size(-1))
        
        # ====================================================================
        # RECONSTRUCTION LOSS (Binary Cross-Entropy for Bernoulli likelihood)
        # ====================================================================
        # L_recon = -E[log p(x|z)] ≈ BCE(x, x_recon)
        recon_loss = F.binary_cross_entropy(x_recon, x_flat, reduction='sum')
        
        # ====================================================================
        # KL DIVERGENCE (Closed-form for Gaussian)
        # ====================================================================
        # KL(q(z|x) || p(z)) where q = N(μ, σ²), p = N(0, I)
        # = -½ Σ[1 + log(σ²) - μ² - σ²]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # ====================================================================
        # TOTAL LOSS
        # ====================================================================
        # Average over batch
        batch_size = x.size(0)
        total_loss = (recon_loss + beta * kl_loss) / batch_size
        
        return total_loss, recon_loss / batch_size, kl_loss / batch_size


class ConvVAE(nn.Module):
    """
    Convolutional VAE for image data
    
    Uses convolutional encoder and transposed convolutional decoder.
    Better for larger images than the MLP-based VAE.
    
    Args:
        in_channels: Number of input channels. Default: 1
        latent_dim: Latent space dimension. Default: 128
        image_size: Input image size (assumed square). Default: 28
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        image_size: int = 28
    ):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculate feature map size after encoding
        self.feature_size = image_size // 4  # Two stride-2 convolutions
        self.flatten_dim = 64 * self.feature_size * self.feature_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 64, self.feature_size, self.feature_size)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# DEMO AND TESTING
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("VAE Model Summary")
    print("=" * 70)
    
    # Create VAE for MNIST
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=20)
    total_params, trainable_params = count_parameters(model)
    
    print(f"Configuration:")
    print(f"  input_dim: 784 (28×28 MNIST)")
    print(f"  hidden_dim: 512")
    print(f"  latent_dim: 20")
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)
    
    batch_size = 32
    x = torch.rand(batch_size, 784)  # Normalized to [0, 1]
    print(f"Input shape: {x.shape}")
    
    model.train()
    x_recon, mu, logvar = model(x)
    
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"μ shape: {mu.shape}")
    print(f"log σ² shape: {logvar.shape}")
    
    # Compute loss
    loss, recon_loss, kl_loss = VAE.loss_function(x_recon, x, mu, logvar)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"  Reconstruction: {recon_loss.item():.4f}")
    print(f"  KL Divergence: {kl_loss.item():.4f}")
    
    # Test sampling
    print("\n" + "=" * 70)
    print("Sampling Test")
    print("=" * 70)
    
    model.eval()
    samples = model.sample(16)
    print(f"Generated samples shape: {samples.shape}")
    print("=" * 70)
