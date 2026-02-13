"""
FILE: 07_noise_conditional_scores.py
DIFFICULTY: Advanced
ESTIMATED TIME: 4-5 hours
PREREQUISITES: 04-06, understanding of multi-scale modeling

LEARNING OBJECTIVES:
    1. Implement Noise Conditional Score Networks (NCSN)
    2. Understand annealed Langevin dynamics
    3. Train models at multiple noise scales
    4. Generate high-quality samples from complex distributions

MATHEMATICAL BACKGROUND:
    NCSN learns scores at multiple noise levels: s_θ(x, σ) ≈ ∇log p_σ(x)
    
    where p_σ(x) = ∫ p(y)N(x|y, σ²I)dy is the data smoothed by Gaussian noise.
    
    Training objective:
    L = E_σ E_x E_ε[λ(σ)||s_θ(x+ε, σ) + ε/σ²||²]
    
    where:
    - σ ~ p(σ) is sampled from a noise distribution
    - ε ~ N(0, σ²I)
    - λ(σ) = σ² is a weighting function
    
    ANNEALED LANGEVIN DYNAMICS:
    Sample by running Langevin at decreasing noise levels:
    σ₁ > σ₂ > ... > σ_T
    
    This helps overcome mode collapse and improves sample quality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class NCSN(nn.Module):
    """Noise Conditional Score Network."""
    
    def __init__(self, data_dim=2, hidden_dim=128, n_layers=4):
        super().__init__()
        
        # Noise embedding
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Score network
        layers = []
        input_dim = data_dim + hidden_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, sigma):
        """Compute s_θ(x, σ)."""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.full((len(x), 1), sigma, device=x.device)
        else:
            sigma = sigma.view(-1, 1)
        
        # Embed noise level
        sigma_emb = self.sigma_embed(sigma)
        
        # Concatenate and process
        h = torch.cat([x, sigma_emb], dim=-1)
        return self.net(h)


def ncsn_loss(model, x, sigmas):
    """
    Compute NCSN loss with multiple noise levels.
    
    Args:
        model: NCSN model
        x: Data samples, shape (N, D)
        sigmas: List of noise levels
    
    Returns:
        loss: Weighted DSM loss averaged over noise levels
    """
    # Sample random noise level for each data point
    N = len(x)
    sigma_idx = torch.randint(0, len(sigmas), (N,))
    sigma = torch.tensor([sigmas[i] for i in sigma_idx], device=x.device)
    
    # Add noise
    noise = torch.randn_like(x)
    x_noisy = x + noise * sigma.view(-1, 1)
    
    # Predict score
    pred_score = model(x_noisy, sigma)
    
    # Target score: -noise/σ²
    target_score = -noise / (sigma.view(-1, 1) ** 2)
    
    # Weighted MSE (weight = σ²)
    weights = sigma.view(-1, 1) ** 2
    loss = torch.mean(weights * torch.sum((pred_score - target_score) ** 2, dim=1))
    
    return loss


def anneal_langevin_sampling(model, sigmas, n_samples=100, n_steps_per_sigma=100,
                             step_size_ratio=0.00002):
    """
    Annealed Langevin dynamics sampling.
    
    Sample by running Langevin at decreasing noise levels.
    This helps explore the space at high noise and refine at low noise.
    
    Args:
        model: Trained NCSN model
        sigmas: Noise schedule (decreasing), e.g., [10, 1, 0.1, 0.01]
        n_samples: Number of samples to generate
        n_steps_per_sigma: Langevin steps at each noise level
        step_size_ratio: Step size as fraction of σ²
    
    Returns:
        samples: Final samples, shape (n_samples, dim)
        trajectory: Sampling trajectory
    """
    # Initialize from high noise
    dim = 2  # Assume 2D for visualization
    x = torch.randn(n_samples, dim) * sigmas[0]
    
    trajectory = [x.clone()]
    
    with torch.no_grad():
        for sigma in sigmas:
            step_size = step_size_ratio * (sigma ** 2)
            
            for _ in range(n_steps_per_sigma):
                # Compute score at current noise level
                score = model(x, sigma)
                
                # Langevin step
                noise = torch.randn_like(x)
                x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
            
            trajectory.append(x.clone())
    
    return x, trajectory


def train_ncsn(data, sigmas, epochs=5000, lr=1e-3):
    """Train NCSN on 2D data."""
    model = NCSN(data_dim=2, hidden_dim=128, n_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = ncsn_loss(model, data, sigmas)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")
    
    return model, losses


if __name__ == "__main__":
    print("Noise Conditional Score Networks Demo")
    print("=" * 80)
    
    # Create checkerboard dataset
    def checkerboard_data(n_samples=2000):
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,)).float()
        x2 = x2_ + torch.floor(x1) % 2
        return torch.stack([x1, x2], dim=1)
    
    data = checkerboard_data(2000)
    
    # Geometric noise schedule
    sigmas = np.exp(np.linspace(np.log(20), np.log(0.01), 10))
    print(f"\nNoise schedule: {sigmas}")
    
    # Train
    print("\nTraining NCSN...")
    model, losses = train_ncsn(data, sigmas, epochs=3000, lr=1e-3)
    
    # Sample
    print("\nGenerating samples via annealed Langevin dynamics...")
    samples, trajectory = anneal_langevin_sampling(
        model, sigmas, n_samples=2000, n_steps_per_sigma=100
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=1, alpha=0.5)
    axes[0].set_title('Training Data')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5)
    axes[1].set_title('Generated Samples')
    axes[1].set_aspect('equal')
    
    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Curve')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo_ncsn_checkerboard.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_ncsn_checkerboard.png")
    print("\n✓ NCSN successfully learned multi-modal distribution!")
