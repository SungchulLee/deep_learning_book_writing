"""
MODULE 05: Sliced Score Matching
================================

DIFFICULTY: Intermediate
TIME: 2 hours
PREREQUISITES: Module 04

LEARNING OBJECTIVES:
- Understand sliced score matching (SSM) for efficiency
- Implement random projection technique
- Compare SSM vs DSM

Key idea: Project scores onto random directions to reduce computation

L_SSM(θ) = 𝔼_x 𝔼_v [v^T ∇s_θ(x) + 0.5||s_θ(x)||² v^T s_θ(x)]

where v ~ Uniform(S^{d-1}) is random unit vector

Author: Sungchul @ Yonsei University
"""

import torch
import torch.nn as nn
import numpy as np
print("MODULE 05: Sliced Score Matching")
print("="*80)

def sliced_score_matching_loss(score_fn, x, n_projections=1):
    """
    Compute sliced score matching loss
    
    Args:
        score_fn: Score network
        x: Data batch [B, D]
        n_projections: Number of random projections
    
    Returns:
        loss: SSM loss value
    """
    x.requires_grad_(True)
    score = score_fn(x)
    
    loss = 0
    for _ in range(n_projections):
        # Random unit vector
        v = torch.randn_like(x)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        
        # v^T * score
        v_score = torch.sum(v * score, dim=-1, keepdim=True)
        
        # ∇(v^T score) w.r.t. x
        sv_x = torch.autograd.grad(v_score.sum(), x, create_graph=True)[0]
        
        # v^T * ∇score
        v_grad_score = torch.sum(v * sv_x, dim=-1)
        
        # SSM loss components
        loss = loss + v_grad_score + 0.5 * v_score.squeeze() ** 2
    
    return loss.mean() / n_projections

# Example usage
from sklearn.datasets import make_swiss_roll
X, _ = make_swiss_roll(n_samples=2000, noise=0.1)
X = X[:, [0, 2]]  # Take 2D slice

class SimpleScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleScoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training with Sliced Score Matching...")
X_tensor = torch.FloatTensor(X)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = sliced_score_matching_loss(model, X_tensor, n_projections=2)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("""
ADVANTAGES OF SSM:
✓ More efficient than explicit score matching (no Hessian)
✓ Unbiased estimator of Fisher divergence
✓ Works well for high dimensions
✓ Variance reduced with multiple projections

COMPARISON:
- DSM: O(d) per sample, needs noise
- SSM: O(d*p) per sample, no noise needed (p=projections)
- ESM: O(d²) per sample, impractical

WHEN TO USE SSM:
- When adding noise is difficult/unnatural
- When you want deterministic training (no noise sampling)
- For certain theoretical guarantees
""")

print("\n✓ Module 05 complete!")
