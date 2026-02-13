"""
FILE: 10_advanced_applications.py
DIFFICULTY: Advanced
ESTIMATED TIME: 3-4 hours
PREREQUISITES: 07-09

LEARNING OBJECTIVES:
    1. Implement conditional generation with score models
    2. Understand image inpainting
    3. Learn inverse problem solving with scores
    4. Explore controllable generation

MATHEMATICAL BACKGROUND:
    CONDITIONAL GENERATION:
    Learn s_θ(x, y) = ∇log p(x|y) where y is a condition (e.g., class label).
    
    CLASSIFIER GUIDANCE:
    s(x|y) = s(x) + ∇log p(y|x)
    
    where the second term guides toward desired class.
    
    INPAINTING:
    Given mask M and observed pixels x_obs, solve:
    argmax_x log p(x) s.t. x_M = x_obs
    
    Can be done by projecting during sampling.
"""

import torch
import torch.nn as nn
import numpy as np


class ConditionalScoreNetwork(nn.Module):
    """Score network conditioned on class labels."""
    
    def __init__(self, data_dim=2, n_classes=10, hidden_dim=128):
        super().__init__()
        
        # Class embedding
        self.class_embed = nn.Embedding(n_classes, hidden_dim)
        
        # Score network
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x, y):
        """
        Compute conditional score s(x|y).
        
        Args:
            x: Data points, shape (N, D)
            y: Class labels, shape (N,)
        """
        y_emb = self.class_embed(y)
        inp = torch.cat([x, y_emb], dim=-1)
        return self.net(inp)


def inpaint_with_scores(model, x_obs, mask, n_steps=1000, step_size=0.01):
    """
    Image inpainting using score-based method.
    
    Strategy:
    1. Run Langevin dynamics to sample from p(x)
    2. After each step, project to match observed pixels
    
    Args:
        model: Score model
        x_obs: Observed pixels, shape (H, W, C)
        mask: Binary mask, 1=observed, 0=unknown, shape (H, W, C)
        n_steps: Number of sampling steps
        step_size: Langevin step size
    
    Returns:
        x: Inpainted image
    """
    x = torch.randn_like(x_obs)
    
    with torch.no_grad():
        for step in range(n_steps):
            # Score-based update
            score = model(x.flatten(), torch.tensor([0.0]))
            score = score.reshape(x.shape)
            
            noise = torch.randn_like(x)
            x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
            
            # Project: replace observed pixels
            x = mask * x_obs + (1 - mask) * x
    
    return x


def demo_conditional_generation():
    """Demonstrate conditional score modeling."""
    print("Conditional Score-Based Generation")
    print("=" * 80)
    
    # Synthetic conditional data: class-dependent Gaussians
    n_classes = 3
    samples_per_class = 300
    
    data = []
    labels = []
    
    for c in range(n_classes):
        # Each class has different mean
        mean = torch.tensor([c * 2.0, 0.0])
        samples = torch.randn(samples_per_class, 2) * 0.5 + mean
        data.append(samples)
        labels.append(torch.full((samples_per_class,), c, dtype=torch.long))
    
    data = torch.cat(data)
    labels = torch.cat(labels)
    
    print(f"\nDataset: {len(data)} samples, {n_classes} classes")
    
    # Train conditional model
    model = ConditionalScoreNetwork(data_dim=2, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining conditional score model...")
    for epoch in range(2000):
        # Simple DSM loss
        noise = torch.randn_like(data) * 0.5
        noisy_data = data + noise
        
        pred_score = model(noisy_data, labels)
        target_score = -noise / (0.5 ** 2)
        
        loss = torch.mean(torch.sum((pred_score - target_score) ** 2, dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    # Generate conditional samples
    print("\nGenerating conditional samples...")
    with torch.no_grad():
        samples_0 = torch.randn(100, 2) * 3
        samples_1 = torch.randn(100, 2) * 3
        samples_2 = torch.randn(100, 2) * 3
        
        y_0 = torch.zeros(100, dtype=torch.long)
        y_1 = torch.ones(100, dtype=torch.long)
        y_2 = torch.full((100,), 2, dtype=torch.long)
        
        # Langevin sampling (simplified)
        for _ in range(100):
            score_0 = model(samples_0, y_0)
            score_1 = model(samples_1, y_1)
            score_2 = model(samples_2, y_2)
            
            samples_0 = samples_0 + 0.01 * score_0 + 0.1 * torch.randn_like(samples_0)
            samples_1 = samples_1 + 0.01 * score_1 + 0.1 * torch.randn_like(samples_1)
            samples_2 = samples_2 + 0.01 * score_2 + 0.1 * torch.randn_like(samples_2)
    
    print("✓ Conditional generation successful!")
    print(f"\nGenerated samples - Class 0: {samples_0.shape}")
    print(f"Generated samples - Class 1: {samples_1.shape}")
    print(f"Generated samples - Class 2: {samples_2.shape}")


if __name__ == "__main__":
    demo_conditional_generation()
