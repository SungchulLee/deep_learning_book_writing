"""
MODULE 06: Score Network Architectures
=====================================

DIFFICULTY: Intermediate
TIME: 2-3 hours
PREREQUISITES: Modules 01-05

LEARNING OBJECTIVES:
- Design effective score networks for different data types
- Understand noise/time conditioning
- Implement modern architectures

Author: Sungchul @ Yonsei University
"""

import torch
import torch.nn as nn
import numpy as np

print("MODULE 06: Score Network Architectures")
print("="*80)

class TimeConditionalScoreNetwork(nn.Module):
    """
    Score network with time/noise conditioning
    
    For diffusion models, score depends on noise level t:
    s_θ(x, t) = ∇_x log p_t(x)
    """
    def __init__(self, data_dim=2, hidden_dim=128, time_embed_dim=32):
        super().__init__()
        
        # Time embedding (sinusoidal like Transformer)
        self.time_embed_dim = time_embed_dim
        
        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def get_timestep_embedding(self, timesteps, max_period=10000):
        """Sinusoidal time embedding (from Transformer)"""
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, x, t):
        """
        Args:
            x: Data [batch_size, data_dim]
            t: Time/noise level [batch_size]
        
        Returns:
            score: ∇_x log p_t(x) [batch_size, data_dim]
        """
        # Embed time
        t_embed = self.get_timestep_embedding(t)
        t_embed = self.time_mlp(t_embed)
        
        # Concatenate with data
        x_with_time = torch.cat([x, t_embed], dim=-1)
        
        # Predict score
        return self.net(x_with_time)

# Example: Train on toy data with multiple noise levels
from sklearn.datasets import make_swiss_roll
X, _ = make_swiss_roll(n_samples=2000, noise=0.1)
X = X[:, [0, 2]] / 10.0  # Normalize

model = TimeConditionalScoreNetwork(data_dim=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Multiple noise levels
sigmas = [0.01, 0.05, 0.1, 0.5, 1.0]

print("Training time-conditional score network...")
X_tensor = torch.FloatTensor(X)

for epoch in range(1000):
    # Random noise level per batch
    sigma_idx = np.random.randint(len(sigmas))
    sigma = sigmas[sigma_idx]
    t = torch.ones(len(X)) * sigma_idx  # Time index
    
    # DSM loss at this noise level
    noise = torch.randn_like(X_tensor)
    X_noisy = X_tensor + sigma * noise
    
    pred_score = model(X_noisy, t)
    target_score = -noise / sigma
    
    loss = torch.mean((pred_score - target_score) ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, σ = {sigma:.3f}")

print("""
KEY ARCHITECTURE CHOICES:

1. TIME CONDITIONING:
   - Essential for diffusion models
   - Sinusoidal embedding (Transformer-style)
   - Or learnable embedding

2. FOR 2D/TABULAR DATA:
   - MLP with LayerNorm
   - SiLU/GELU activations
   - Residual connections help

3. FOR IMAGES (covered in Module 09):
   - U-Net architecture
   - Attention mechanisms
   - Group normalization

4. FOR SEQUENCES:
   - Transformer blocks
   - Causal masking if needed

DESIGN PRINCIPLES:
✓ Score is a VECTOR FIELD (same dim as input)
✓ Must handle varying noise/time levels
✓ Should be smooth and well-behaved
✓ Capacity scales with data complexity
""")

print("\n✓ Module 06 complete!")
