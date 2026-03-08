"""
MODULE 02: Score Matching Theory
================================

DIFFICULTY: Beginner-Intermediate
TIME: 3-4 hours
PREREQUISITES: Module 01 (Score Functions Basics)

LEARNING OBJECTIVES:
-------------------
1. Understand why we can't compute scores directly from data
2. Learn Explicit Score Matching (ESM) objective
3. Understand Denoising Score Matching (DSM) - the key idea!
4. Connect DSM to Bayesian denoising
5. Implement basic score matching for toy data

MATHEMATICAL FOUNDATION:
-----------------------
THE PROBLEM:
Given dataset {x_i}_{i=1}^N ~ p_data(x), learn s(x) = ∇_x log p_data(x)

Challenge: We don't know p_data(x)! Only have samples!

NAIVE APPROACH (doesn't work):
Fit p_θ(x) to data, then compute s_θ(x) = ∇_x log p_θ(x)
Problem: Requires normalizing p_θ(x), which is intractable!

SCORE MATCHING SOLUTION:
Match the score s_θ(x) directly to true score, without needing p_data(x)!

Key insight: We can match scores without knowing either distribution!

Author: Sungchul @ Yonsei University  
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from scipy.stats import multivariate_normal

plt.style.use('seaborn-v0_8-darkgrid')
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("MODULE 02: SCORE MATCHING THEORY")
print("=" * 80)

# ============================================================================
# SECTION 1: THE FUNDAMENTAL PROBLEM
# ============================================================================
print("\n" + "="  * 80)
print("SECTION 1: Why Direct Score Computation Fails")
print("=" * 80)

print("""
SCENARIO:
--------
Given: Dataset {x₁, x₂, ..., x_N} sampled from unknown p_data(x)
Goal: Learn score function s(x) = ∇_x log p_data(x)

WHY WE CAN'T COMPUTE SCORE DIRECTLY:
-----------------------------------
1. Don't have formula for p_data(x)
2. Could fit model p_θ(x) to data
3. But normalizing p_θ(x) requires computing:
   Z_θ = ∫ p̃_θ(x) dx  (intractable!)
   
4. So can't compute: log p_θ(x) = log p̃_θ(x) - log Z_θ
5. Therefore can't compute: ∇_x log p_θ(x)

EXAMPLE:
-------
Say we parameterize: p̃_θ(x) = exp(E_θ(x))
where E_θ is a neural network (energy function).

Then: p_θ(x) = exp(E_θ(x)) / Z_θ
where: Z_θ = ∫ exp(E_θ(x)) dx  ← INTRACTABLE!

But we want: s_θ(x) = ∇_x log p_θ(x) = ∇_x E_θ(x)

Good news: Score doesn't need Z_θ!
Bad news: Still can't train without knowing p_data(x)!

SOLUTION: Score matching!
""")

# ============================================================================
# SECTION 2: EXPLICIT SCORE MATCHING (ESM)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Explicit Score Matching (ESM)")
print("=" * 80)

print("""
FISHER DIVERGENCE:
-----------------
Define distance between score functions:

D_Fisher(p||q) = (1/2) 𝔼_{x~p} ||∇_x log p(x) - ∇_x log q(x)||²

Minimize w.r.t. θ:
J_ESM(θ) = (1/2) 𝔼_{x~p_data} ||∇_x log p_data(x) - s_θ(x)||²

Problem: Still need ∇_x log p_data(x) which we don't have!

HYVÄRINEN'S TRICK (2005):
------------------------
Using integration by parts (Stein's identity):

J_ESM(θ) = 𝔼_{x~p_data} [ tr(∇_x s_θ(x)) + (1/2)||s_θ(x)||² ] + const

KEY INSIGHT:
- Removed ∇_x log p_data(x) from objective!
- Only need: samples from p_data and derivatives of s_θ
- tr(∇_x s_θ(x)) = sum of diagonal elements of Jacobian

DRAWBACK:
- Computing tr(∇_x s_θ(x)) requires computing Hessian  
- Very expensive for high dimensions!
- Not practical for images, etc.

Need better approach → Denoising Score Matching!
""")

# Demonstrate ESM on toy data
print("\nDemonstration: ESM on 2D Gaussian")
print("-" * 80)

# True distribution: 2D Gaussian
mu_true = np.array([0, 0])
Sigma_true = np.array([[1.0, 0.5], [0.5, 1.0]])

# Generate samples
n_samples = 1000
samples = np.random.multivariate_normal(mu_true, Sigma_true, n_samples)

# True score function (for comparison)
Sigma_inv = np.linalg.inv(Sigma_true)
def true_score(x):
    return -Sigma_inv @ (x - mu_true)

# Simple linear score model: s_θ(x) = A(x - b)
# For Gaussian, optimal is A = -Σ^(-1), b = μ
class LinearScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(2, 2) * 0.1)
        self.b = nn.Parameter(torch.zeros(2))
    
    def forward(self, x):
        # s_θ(x) = A(x - b)
        return (self.A @ (x - self.b).T).T

# ESM objective: 𝔼[tr(∇s_θ) + 0.5||s_θ||²]
def esm_loss(model, x_batch):
    """Explicit Score Matching loss"""
    x_batch.requires_grad_(True)
    
    # Compute score
    score = model(x_batch)
    
    # Term 1: 0.5 * ||s_θ(x)||²
    score_norm = 0.5 * torch.sum(score ** 2, dim=-1).mean()
    
    # Term 2: tr(∇_x s_θ(x)) - trace of Jacobian
    # For each component of score, take derivative w.r.t. x
    trace_term = 0
    for i in range(2):
        grad_outputs = torch.zeros_like(score)
        grad_outputs[:, i] = 1
        grads = torch.autograd.grad(score, x_batch, grad_outputs, 
                                     create_graph=True)[0]
        trace_term += grads[:, i].mean()
    
    return score_norm + trace_term

# Train with ESM
model_esm = LinearScoreModel()
optimizer = torch.optim.Adam(model_esm.parameters(), lr=0.01)

x_tensor = torch.FloatTensor(samples)
print("Training with Explicit Score Matching...")
for epoch in range(500):
    optimizer.zero_grad()
    loss = esm_loss(model_esm, x_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

print("\n✓ ESM training complete!")
print(f"  Learned A ≈ -Σ^(-1):")
print(f"    {model_esm.A.detach().numpy()}")
print(f"  True -Σ^(-1):")
print(f"    {-Sigma_inv}")

# ============================================================================
# SECTION 3: DENOISING SCORE MATCHING (DSM)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Denoising Score Matching (DSM) - The Key Idea!")
print("=" * 80)

print("""
THE PROBLEM WITH ESM:
-------------------
Computing tr(∇_x s_θ(x)) is expensive (requires Hessian).
Not scalable to high dimensions (images, etc.).

DENOISING SCORE MATCHING (Vincent 2011):
---------------------------------------
Key insight: Learn to denoise instead!

PROCEDURE:
1. Take clean data x ~ p_data(x)
2. Add Gaussian noise: x̃ = x + σ * ε, where ε ~ N(0, I)
3. Learn to predict the NOISE: s_θ(x̃) ≈ -(x̃ - x)/σ²

OBJECTIVE:
J_DSM(θ) = 𝔼_{x~p_data} 𝔼_{ε~N(0,I)} ||s_θ(x + σε) + ε/σ||²

Why this works:
--------------
The score of the noisy distribution q_σ(x̃|x) = N(x̃; x, σ²I) is:

∇_{x̃} log q_σ(x̃|x) = -(x̃ - x)/σ²

And the marginal noisy distribution q_σ(x̃) = ∫ q_σ(x̃|x)p_data(x)dx
has score that can be approximated by this!

CONNECTION TO BAYESIAN INFERENCE:
--------------------------------
Denoising is Bayesian posterior inference!

Given noisy observation x̃ = x + noise,
infer clean x via posterior: p(x|x̃)

The score ∇_x log p(x|x̃) tells us how to denoise!

This is exactly what diffusion models do:
- Forward: Add noise (known)
- Reverse: Denoise using learned score (learned)

ADVANTAGES OF DSM:
-----------------
✓ No Hessian computation needed
✓ Simple gradient computation
✓ Scales to high dimensions
✓ Connection to denoising autoencoders
✓ Foundation for diffusion models
""")

# Demonstrate DSM
print("\nDemonstration: DSM on 2D Swiss Roll")
print("-" * 80)

# Generate Swiss roll data
def generate_swiss_roll(n_samples=1000, noise=0.1):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x1 = t * np.cos(t)
    x2 = t * np.sin(t)
    data = np.stack([x1, x2], axis=1)
    data += noise * np.random.randn(n_samples, 2)
    return data

swiss_roll = generate_swiss_roll(1000, noise=0.1)

# MLP score network
class MLPScoreNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.net(x)

# DSM loss
def dsm_loss(model, x_batch, sigma=0.1):
    """Denoising Score Matching loss"""
    # Add noise
    noise = torch.randn_like(x_batch)
    x_noisy = x_batch + sigma * noise
    
    # Predict score
    predicted_score = model(x_noisy)
    
    # Target score: -(x_noisy - x)/sigma^2 = -noise/sigma
    target_score = -noise / sigma
    
    # MSE loss
    loss = torch.mean((predicted_score - target_score) ** 2)
    return loss

# Train with DSM
model_dsm = MLPScoreNet()
optimizer_dsm = torch.optim.Adam(model_dsm.parameters(), lr=1e-3)

x_swiss = torch.FloatTensor(swiss_roll)
sigma_noise = 0.1

print("Training with Denoising Score Matching...")
for epoch in range(2000):
    optimizer_dsm.zero_grad()
    loss = dsm_loss(model_dsm, x_swiss, sigma=sigma_noise)
    loss.backward()
    optimizer_dsm.step()
    
    if epoch % 400 == 0:
        print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

print("\n✓ DSM training complete!")

# Visualize learned score field
x1_grid = np.linspace(-15, 15, 20)
x2_grid = np.linspace(-15, 15, 20)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
grid_points = np.stack([X1.ravel(), X2.ravel()], axis=1)

with torch.no_grad():
    scores = model_dsm(torch.FloatTensor(grid_points)).numpy()

scores = scores.reshape(X1.shape + (2,))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Data
axes[0].scatter(swiss_roll[:, 0], swiss_roll[:, 1], s=1, alpha=0.5, c='blue')
axes[0].set_title('Swiss Roll Data', fontsize=14, fontweight='bold')
axes[0].set_xlabel(r'$x_1$')
axes[0].set_ylabel(r'$x_2$')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Plot 2: Learned score field
axes[1].scatter(swiss_roll[:, 0], swiss_roll[:, 1], s=1, alpha=0.3, c='blue')
axes[1].quiver(X1, X2, scores[:, :, 0], scores[:, :, 1], 
               color='red', alpha=0.6, scale=50, width=0.003)
axes[1].set_title('Learned Score Field via DSM\n(Points toward data manifold)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel(r'$x_1$')
axes[1].set_ylabel(r'$x_2$')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_dsm_swiss_roll.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_dsm_swiss_roll.png")
plt.close()

# ============================================================================
# SECTION 4: COMPARING ESM AND DSM
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: ESM vs DSM Comparison")
print("=" * 80)

comparison_table = """
|  Aspect                | Explicit Score Matching (ESM)     | Denoising Score Matching (DSM)  |
|-----------------------|-----------------------------------|----------------------------------|
| Objective             | 𝔼[tr(∇s_θ) + 0.5||s_θ||²]        | 𝔼||s_θ(x̃) + ε/σ||²             |
| Derivatives needed    | Second-order (Hessian)            | First-order only                 |
| Computational cost    | O(d²) per sample                  | O(d) per sample                  |
| Scalability           | Poor (high dimensions)            | Excellent                        |
| Connection            | Fisher divergence                 | Bayesian denoising               |
| Practical use         | Rare (too expensive)              | Standard (all modern models)     |
| Noise parameter       | Not needed                        | Requires σ choice                |
"""

print(comparison_table)

print("""
KEY TAKEAWAYS:
-------------
1. ESM is theoretically elegant but computationally expensive
2. DSM is practical and scalable - used in all modern models
3. DSM connects to Bayesian denoising - fundamental for diffusion
4. Both avoid computing normalization constants
5. DSM enables learning scores from data alone

NEXT STEP: Multi-scale DSM
--------------------------
Problem: Single noise level σ may not work well everywhere
Solution: Learn scores at MULTIPLE noise levels
This leads to Noise Conditional Score Networks (NCSN)!
""")

# ============================================================================
# SUMMARY AND EXERCISES
# ============================================================================
print("\n" + "=" * 80)
print("MODULE SUMMARY")
print("=" * 80)

print("""
WHAT WE LEARNED:
---------------
1. Direct score computation from data is intractable
2. Explicit Score Matching (ESM) avoids normalization but needs Hessian
3. Denoising Score Matching (DSM) is practical and scalable
4. DSM = learning to denoise = Bayesian inference!
5. DSM is foundation for diffusion models

KEY FORMULAS:
------------
1. ESM objective:
   J_ESM = 𝔼[tr(∇_x s_θ(x)) + 0.5||s_θ(x)||²]

2. DSM objective:
   J_DSM = 𝔼_{x,ε} ||s_θ(x + σε) + ε/σ||²

3. Denoising score:
   ∇_x log q_σ(x̃|x) = -(x̃ - x)/σ²

FILES GENERATED:
---------------
1. 02_dsm_swiss_roll.png - DSM on Swiss roll data

""")

print("=" * 80)
print("EXERCISES")
print("=" * 80)

print("""
EXERCISE 1: Derive ESM from Fisher Divergence
-------------------------------------------
Starting from:
D_Fisher(p||q) = (1/2)𝔼_{x~p}||∇log p(x) - ∇log q(x)||²

Expand and use integration by parts to derive:
J_ESM = 𝔼[tr(∇s_θ) + 0.5||s_θ||²] + const

EXERCISE 2: Implement ESM
------------------------
a) Implement ESM for mixture of Gaussians
b) Compare computational cost vs DSM
c) Verify they give similar results

EXERCISE 3: DSM Derivation
-------------------------
Show that for q_σ(x̃|x) = N(x̃; x, σ²I):
∇_{x̃} log q_σ(x̃|x) = -(x̃ - x)/σ²

Interpret: denoising = computing posterior score!

EXERCISE 4: Noise Level Analysis
-------------------------------
Train DSM with different σ values:
a) Very small σ (σ=0.01)
b) Medium σ (σ=0.1)
c) Large σ (σ=1.0)

How does choice of σ affect:
- Training stability?
- Quality of learned scores?
- Sampling behavior?

EXERCISE 5: Connection to Autoencoders
-------------------------------------
A denoising autoencoder learns: f(x̃) ≈ x
Show that:
a) Optimal denoising function relates to score
b) Connection: ∇_{x̃} log p(x̃) ∝ f(x̃) - x̃
c) Implement both and compare

CHALLENGE EXERCISE: Multi-Modal Distribution
------------------------------------------
Create 2D "checkerboard" with 9 Gaussian modes.
a) Train DSM with single σ
b) Does it capture all modes?
c) Try different σ values
d) Motivate multi-scale approach (next module!)
""")

print("\n" + "=" * 80)
print("NEXT MODULE: 03_langevin_dynamics.py")
print("=" * 80)
print("""
Now that we can learn scores from data,
how do we USE them to generate samples?

Answer: Langevin dynamics - MCMC sampling using scores!
This connects back to Bayesian computation from Module 01!
""")

print("\n✓ Module 02 complete!")
print("=" * 80)


if __name__ == "__main__":
    pass
