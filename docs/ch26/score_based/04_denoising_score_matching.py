"""
FILE: 04_denoising_score_matching.py
DIFFICULTY: Intermediate
ESTIMATED TIME: 3-4 hours
PREREQUISITES: 01-03, basic neural networks, understanding of noise perturbation

LEARNING OBJECTIVES:
    1. Implement practical denoising score matching
    2. Train score networks on 2D datasets
    3. Understand noise kernel selection
    4. Generate samples from learned models

MATHEMATICAL BACKGROUND:
    Denoising Score Matching adds noise to data and learns to predict the noise direction.
    
    Training: Minimize E_x E_ε[||s_θ(x + ε) + ε/σ²||²]
    where ε ~ N(0, σ²I)
    
    This is equivalent to learning ∇log q(x̃|x) where q is the noise kernel.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import os


class ScoreNetwork(nn.Module):
    """Score network for 2D data."""
    
    def __init__(self, hidden_dims=[128, 128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = 2
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Softplus(),  # Smooth activation
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def dsm_loss(model, x, sigma=0.5):
    """Compute denoising score matching loss."""
    # Add noise
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    
    # Predict score
    pred_score = model(x_noisy)
    
    # Target score: -noise/sigma²
    target_score = -noise / (sigma ** 2)
    
    # MSE loss
    loss = torch.mean(torch.sum((pred_score - target_score) ** 2, dim=1))
    return loss


def train_score_model(data, sigma=0.5, epochs=5000, lr=1e-3):
    """Train score model on 2D data."""
    model = ScoreNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = dsm_loss(model, data, sigma)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")
    
    return model, losses


def sample_langevin(model, n_samples=100, n_steps=1000, step_size=0.01, noise_scale=1.0):
    """Generate samples using Langevin dynamics with learned score."""
    # Initialize from prior
    x = torch.randn(n_samples, 2) * noise_scale
    
    samples = []
    
    with torch.no_grad():
        for step in range(n_steps):
            score = model(x)
            noise = torch.randn_like(x)
            x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
            
            if step % 10 == 0:
                samples.append(x.clone())
    
    return x, samples


if __name__ == "__main__":
    print("Denoising Score Matching on 2D Datasets")
    print("=" * 80)
    
    # Create Swiss roll dataset
    from sklearn.datasets import make_swiss_roll
    data_np, _ = make_swiss_roll(n_samples=2000, noise=0.5)
    data = torch.tensor(data_np[:, [0, 2]] / 10, dtype=torch.float32)  # Use X-Z plane
    
    print(f"\nDataset: Swiss Roll")
    print(f"Number of samples: {len(data)}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Train
    print("\nTraining score model...")
    model, losses = train_score_model(data, sigma=0.5, epochs=3000, lr=1e-3)
    
    # Sample
    print("\nGenerating samples via Langevin dynamics...")
    final_samples, trajectory = sample_langevin(
        model, n_samples=500, n_steps=1000, step_size=0.05
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training data
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), 
                    s=1, alpha=0.5, c='blue')
    axes[0].set_title('Training Data')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Generated samples
    axes[1].scatter(final_samples[:, 0].numpy(), final_samples[:, 1].numpy(),
                    s=1, alpha=0.5, c='red')
    axes[1].set_title('Generated Samples')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    # Training curve
    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('DSM Loss')
    axes[2].set_title('Training Curve')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo_dsm_swiss_roll.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_dsm_swiss_roll.png")
    
    print("\n✓ Successfully trained and sampled from Swiss roll distribution!")
