"""
Score Matching for Energy-Based Model Training
============================================

Score matching provides an alternative to maximum likelihood for training EBMs,
avoiding the intractable partition function. The score is the gradient of log probability.

Learning Objectives:
-------------------
1. Understand the score function and its properties
2. Implement score matching objective
3. Learn denoising score matching
4. Compare with maximum likelihood
5. Apply to continuous data distributions

Key Concepts:
------------
- Score: ∇ₓ log p(x) = -∇ₓ E(x) / T
- Score Matching: min E_p[(∇ₓ log p(x) - ∇ₓ log q(x))²]
- Denoising Score Matching: Practical approximation
- No partition function computation needed

Duration: 90-120 minutes
Prerequisites: Modules 01-05, Calculus (gradients)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)


class EnergyNetwork(nn.Module):
    """Neural network that outputs energy for score matching."""
    
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()


def score_matching_loss(energy_net, x):
    """
    Compute score matching loss.
    
    L = ½ E[‖∇ₓ E(x)‖² + 2 Tr(∇²ₓ E(x))]
    
    This avoids computing the partition function!
    """
    x = x.requires_grad_(True)
    
    # Compute energy
    energy = energy_net(x)
    
    # Compute score (gradient of energy)
    score = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x,
        create_graph=True
    )[0]
    
    # First term: ‖∇E‖²
    score_norm = (score ** 2).sum(dim=1).mean()
    
    # Second term: Tr(∇²E) - trace of Hessian
    trace = 0
    for i in range(x.shape[1]):
        grad2 = torch.autograd.grad(
            outputs=score[:, i].sum(),
            inputs=x,
            create_graph=True
        )[0]
        trace += grad2[:, i]
    
    trace_term = 2 * trace.mean()
    
    return 0.5 * score_norm + trace_term


def denoising_score_matching_loss(energy_net, x, noise_std=0.1):
    """
    Denoising score matching: Easier to compute than full score matching.
    
    Add noise: x̃ = x + ε, ε ~ N(0, σ²I)
    Loss: E[‖∇ₓ̃ E(x̃) + (x̃ - x)/σ²‖²]
    """
    # Add Gaussian noise
    noise = torch.randn_like(x) * noise_std
    x_noisy = x + noise
    x_noisy = x_noisy.requires_grad_(True)
    
    # Compute energy and its gradient
    energy = energy_net(x_noisy)
    score = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x_noisy,
        create_graph=True
    )[0]
    
    # Target score: -(x̃ - x)/σ²
    target_score = -noise / (noise_std ** 2)
    
    # MSE between predicted and target scores
    loss = ((score - target_score) ** 2).sum(dim=1).mean()
    
    return loss


def train_score_matching_2d():
    """Train EBM on 2D mixture of Gaussians using score matching."""
    print("\n" + "="*70)
    print("SCORE MATCHING TRAINING (2D Example)")
    print("="*70)
    
    # Generate 2D mixture of Gaussians
    n_samples = 5000
    means = [torch.tensor([-2., -2.]), torch.tensor([2., 2.]), 
             torch.tensor([-2., 2.])]
    
    samples = []
    for _ in range(n_samples):
        mean = means[np.random.choice(len(means))]
        sample = mean + torch.randn(2) * 0.5
        samples.append(sample)
    
    data = torch.stack(samples)
    print(f"Generated {n_samples} samples from 3-component mixture")
    
    # Create energy network
    energy_net = EnergyNetwork(input_dim=2, hidden_dim=64)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    # Training
    n_epochs = 1000
    batch_size = 128
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            
            # Use denoising score matching (easier to compute)
            loss = denoising_score_matching_loss(energy_net, batch_x, noise_std=0.5)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(loader))
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {losses[-1]:.4f}")
    
    # Visualize learned energy landscape
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training loss
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Score Matching Loss')
    axes[0].set_title('Training Progress')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Learned energy landscape
    x_range = torch.linspace(-4, 4, 100)
    y_range = torch.linspace(-4, 4, 100)
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        energies = energy_net(grid).numpy().reshape(100, 100)
    
    axes[1].contourf(X.numpy(), Y.numpy(), energies, levels=30, cmap='viridis')
    axes[1].scatter(data[:, 0], data[:, 1], c='red', s=1, alpha=0.3)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Learned Energy Landscape')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/06_score_matching_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Score matching training complete")
    return energy_net


def main():
    print("="*70)
    print("SCORE MATCHING FOR ENERGY-BASED MODELS")
    print("="*70)
    
    train_score_matching_2d()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Score matching avoids partition function")
    print("  ✓ Denoising score matching is practical")
    print("  ✓ Connection to diffusion models")
    print("\nNext: 07_neural_ebms.py")


if __name__ == "__main__":
    main()
