"""
MODULE 07: Noise Conditional Score Networks (NCSN)
=================================================

DIFFICULTY: Advanced
TIME: 3-4 hours
PREREQUISITES: Modules 01-06

LEARNING OBJECTIVES:
- Understand multi-scale score modeling
- Implement annealed Langevin dynamics
- Connect to diffusion forward process

Key idea: Learn scores at multiple noise levels
s_θ(x, σ_i) for σ_1 > σ_2 > ... > σ_L

Author: Sungchul @ Yonsei University
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("MODULE 07: Noise Conditional Score Networks")
print("="*80)

print("""
WHY MULTIPLE NOISE LEVELS?
-------------------------
Problem with single σ:
- Small σ: Accurate near data, but hard to sample (score vanishes far from data)
- Large σ: Easy to sample everywhere, but imprecise

Solution: USE BOTH!
- Start with large σ (easy sampling, covers whole space)
- Gradually decrease σ (refine to data distribution)
- This is ANNEALED LANGEVIN DYNAMICS

CONNECTION TO DIFFUSION:
----------------------
Forward process: x_0 → x_1 → ... → x_T (add noise)
Reverse process: x_T ← ... ← x_1 ← x_0 (denoise)

Each step uses score at appropriate noise level!
This is exactly the diffusion model framework!
""")

class NCSN(nn.Module):
    """Noise Conditional Score Network"""
    def __init__(self, data_dim=2, noise_levels=10):
        super().__init__()
        self.noise_levels = noise_levels
        
        # Shared network with noise conditioning
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, 128),  # +1 for noise level
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, data_dim)
        )
    
    def forward(self, x, sigma_idx):
        """
        Args:
            x: Data [B, D]
            sigma_idx: Noise level index [B], values in [0, noise_levels-1]
        """
        # Normalize sigma_idx to [0, 1]
        sigma_embed = (sigma_idx.float() / self.noise_levels).unsqueeze(-1)
        x_with_sigma = torch.cat([x, sigma_embed], dim=-1)
        return self.net(x_with_sigma)

def train_ncsn(data, n_epochs=2000):
    """Train NCSN on data"""
    # Geometric noise schedule
    sigma_min, sigma_max = 0.01, 1.0
    n_sigmas = 10
    sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n_sigmas))
    
    model = NCSN(data_dim=data.shape[1], noise_levels=n_sigmas)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    data_tensor = torch.FloatTensor(data)
    
    print(f"Training NCSN with {n_sigmas} noise levels...")
    print(f"σ range: [{sigma_min:.3f}, {sigma_max:.3f}]")
    
    for epoch in range(n_epochs):
        # Random noise level
        sigma_idx = torch.randint(0, n_sigmas, (len(data),))
        sigma_vals = torch.FloatTensor([sigmas[i] for i in sigma_idx])
        
        # Add noise
        noise = torch.randn_like(data_tensor)
        noisy_data = data_tensor + sigma_vals.unsqueeze(-1) * noise
        
        # Predict score
        pred_score = model(noisy_data, sigma_idx)
        target_score = -noise / sigma_vals.unsqueeze(-1)
        
        # Weighted loss (weight by 1/σ² for balance)
        weights = 1.0 / (sigma_vals ** 2)
        loss = torch.mean(weights.unsqueeze(-1) * (pred_score - target_score) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    return model, sigmas

def annealed_langevin_sampling(model, sigmas, n_samples=500, n_steps_per_sigma=100):
    """
    Annealed Langevin Dynamics sampling
    
    Start from high noise, gradually decrease
    """
    x = torch.randn(n_samples, 2) * sigmas[0]  # Initialize from prior
    
    trajectory = []
    
    for sigma_idx, sigma in enumerate(sigmas):
        # Langevin steps at this noise level
        epsilon = 2 * (sigma ** 2) / (sigmas[-1] ** 2) * 0.01  # Adaptive step size
        
        for step in range(n_steps_per_sigma):
            with torch.no_grad():
                sigma_idx_tensor = torch.ones(n_samples, dtype=torch.long) * sigma_idx
                score = model(x, sigma_idx_tensor)
            
            # Langevin update
            x = x + epsilon * score + np.sqrt(2 * epsilon) * torch.randn_like(x)
        
        trajectory.append(x.clone().detach().numpy())
        print(f"  Annealing step {sigma_idx+1}/{len(sigmas)}: σ = {sigma:.4f}")
    
    return x.detach().numpy(), trajectory

# Train on moons dataset
from sklearn.datasets import make_moons
data, _ = make_moons(n_samples=2000, noise=0.05)

model, sigmas = train_ncsn(data, n_epochs=2000)

print("\nGenerating samples via annealed Langevin dynamics...")
samples, trajectory = annealed_langevin_sampling(model, sigmas, n_samples=500, n_steps_per_sigma=50)

# Visualize
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, (ax, samples_at_sigma) in enumerate(zip(axes, trajectory)):
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.2, c='blue', label='Data')
    ax.scatter(samples_at_sigma[:, 0], samples_at_sigma[:, 1], s=1, alpha=0.5, c='red', label='Samples')
    ax.set_title(f'σ = {sigmas[idx]:.3f}', fontweight='bold')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    ax.set_aspect('equal')
    if idx == 0:
        ax.legend()

plt.suptitle('Annealed Langevin Dynamics: Gradual Denoising', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('07_annealed_langevin.png', dpi=150)
plt.close()
print("✓ Saved: 07_annealed_langevin.png")

print("""
KEY INSIGHTS:
------------
1. Start with high noise (σ_1): Samples cover whole space
2. Gradually reduce noise: Samples converge to data manifold
3. Each noise level refines the previous level
4. This is exactly the reverse diffusion process!

NOISE SCHEDULE DESIGN:
---------------------
- Geometric progression: σ_i = σ_max * (σ_min/σ_max)^(i/L)
- More levels = smoother transition, slower sampling
- Adaptive step sizes: ε_i ∝ σ_i²

CONNECTION TO DDPM:
-----------------
DDPM forward: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
→ Equivalent to adding noise with schedule

DDPM reverse: Learn p(x_{t-1}|x_t) via score
→ Equivalent to annealed Langevin!

We've now built the complete score-based framework!
Next: Continuous-time formulation (SDEs)
""")

print("\n✓ Module 07 complete!")
