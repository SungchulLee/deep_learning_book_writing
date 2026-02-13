"""
Neural Energy-Based Models: Deep Learning for Energy Functions
============================================================

Modern EBMs use deep neural networks as flexible energy functions.
This module covers training techniques, Langevin dynamics, and image generation.

Duration: 120-150 minutes
Prerequisites: Modules 01-06, Deep learning basics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class ConvEnergyNetwork(nn.Module):
    """Convolutional neural network for image energy."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 14x14
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 7x7
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze()


def langevin_dynamics(energy_net, x_init, n_steps=100, step_size=0.01, noise_scale=0.005):
    """
    Sample using Langevin dynamics: x ← x - ε∇E(x) + √(2ε)ξ
    """
    x = x_init.clone().requires_grad_(True)
    
    for _ in range(n_steps):
        energy = energy_net(x).sum()
        grad = torch.autograd.grad(energy, x)[0]
        
        noise = torch.randn_like(x) * noise_scale
        x = x.data - step_size * grad + noise
        x = x.requires_grad_(True)
    
    return x.detach()


def train_neural_ebm():
    """Train neural EBM on MNIST."""
    print("Training Neural EBM on MNIST...")
    
    # Load data
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Create model
    energy_net = ConvEnergyNetwork()
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.0001)
    
    # Training loop (simplified)
    n_epochs = 2
    for epoch in range(n_epochs):
        for batch_idx, (data, _) in enumerate(loader):
            if batch_idx > 50:  # Quick demo
                break
            
            # Positive samples (data)
            pos_energy = energy_net(data).mean()
            
            # Negative samples (via Langevin)
            neg_samples = torch.rand_like(data)
            neg_samples = langevin_dynamics(energy_net, neg_samples, n_steps=20)
            neg_energy = energy_net(neg_samples).mean()
            
            # Contrastive loss
            loss = pos_energy - neg_energy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print("✓ Neural EBM training complete")
    return energy_net


def main():
    print("="*70)
    print("NEURAL ENERGY-BASED MODELS")
    print("="*70)
    
    train_neural_ebm()
    
    print("\nKey Takeaways:")
    print("  ✓ Deep networks as flexible energy functions")
    print("  ✓ Langevin dynamics for sampling")
    print("  ✓ Modern EBM architectures")


if __name__ == "__main__":
    main()
