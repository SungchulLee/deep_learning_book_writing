"""
FILE: 05_sliced_score_matching.py
DIFFICULTY: Intermediate
ESTIMATED TIME: 2-3 hours
PREREQUISITES: 02_score_matching_theory.py, 04_denoising_score_matching.py

LEARNING OBJECTIVES:
    1. Understand Sliced Score Matching (SSM)
    2. Implement random projection technique
    3. Compare SSM with DSM efficiency
    4. Analyze trade-offs between methods

MATHEMATICAL BACKGROUND:
    Sliced Score Matching uses random projections to avoid computing the full Jacobian.
    
    L_SSM = E_x E_v[v^T∇s_θ(x)v + 1/2||v^Ts_θ(x)||²]
    
    where v ~ N(0, I) is a random projection direction.
    
    Key advantage: Only requires Jacobian-vector products (JVPs) which are cheap!
"""

import torch
import torch.nn as nn
import numpy as np


def sliced_score_matching_loss(model, x, n_projections=1):
    """
    Compute Sliced Score Matching loss using random projections.
    
    The key insight is that we only need directional derivatives:
    v^T∇s_θ(x)v can be computed efficiently using JVPs.
    
    Args:
        model: Score network
        x: Data samples, shape (N, D)
        n_projections: Number of random projections per sample
    
    Returns:
        loss: SSM loss value
    """
    x = x.requires_grad_(True)
    N, D = x.shape
    
    loss = 0.0
    
    for _ in range(n_projections):
        # Sample random projection direction
        v = torch.randn_like(x)  # v ~ N(0, I)
        
        # Compute score
        score = model(x)
        
        # Compute v^T s_θ(x) (inner product)
        score_v = torch.sum(score * v, dim=1, keepdim=True)
        
        # Compute ∇(v^T s_θ(x)) · v using autograd
        # This is equivalent to v^T ∇s_θ(x) v
        grad_score_v = torch.autograd.grad(
            outputs=score_v,
            inputs=x,
            grad_outputs=torch.ones_like(score_v),
            create_graph=True
        )[0]
        
        trace_term = torch.sum(grad_score_v * v, dim=1)
        norm_term = 0.5 * score_v.squeeze() ** 2
        
        loss = loss + torch.mean(trace_term + norm_term)
    
    return loss / n_projections


def demo_ssm():
    """Demonstrate SSM on simple 2D Gaussian."""
    print("Sliced Score Matching Demo")
    print("=" * 80)
    
    # Generate data
    data = torch.randn(500, 2)
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.Softplus(),
        nn.Linear(64, 64),
        nn.Softplus(),
        nn.Linear(64, 2)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining with SSM...")
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = sliced_score_matching_loss(model, data, n_projections=1)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    print("\n✓ SSM training complete!")
    print("\nKey observation: SSM avoids full Jacobian computation")
    print("  - Only needs JVPs (cheap!)")
    print("  - Scales well to high dimensions")
    print("  - Unbiased estimator of ESM objective")


if __name__ == "__main__":
    demo_ssm()
