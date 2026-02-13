"""
MODULE 03: Langevin Dynamics for Sampling
==========================================

DIFFICULTY: Intermediate
TIME: 2-3 hours
PREREQUISITES: Modules 01-02, Basic MCMC understanding

LEARNING OBJECTIVES:
-------------------
1. Understand Langevin MCMC as gradient-based sampling
2. Connect to Metropolis-Hastings from Bayesian inference
3. Implement Langevin dynamics for score-based sampling
4. Analyze convergence properties
5. Understand connection to diffusion reverse process

MATHEMATICAL FOUNDATION:
-----------------------
Langevin Dynamics (Langevin 1908, extended by many):

dx_t = ∇_x log p(x_t) dt + √(2)dW_t

Discrete-time version (Langevin MCMC):

x_{t+1} = x_t + ε * ∇_x log p(x_t) + √(2ε) * z_t

where z_t ~ N(0, I)

KEY INSIGHT:
- Drift term: ε * ∇_x log p(x_t) moves toward high probability
- Diffusion term: √(2ε) * z_t adds exploration noise
- Balances exploitation (gradient) with exploration (noise)
- Converges to target distribution p(x)!

CONNECTION TO SCORE MATCHING:
Since s(x) = ∇_x log p(x), we can sample using learned scores!

x_{t+1} = x_t + ε * s_θ(x_t) + √(2ε) * z_t

This is how diffusion models generate samples!

Author: Sungchul @ Yonsei University
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("MODULE 03: LANGEVIN DYNAMICS")
print("=" * 80)

# ============================================================================
# SECTION 1: FROM METROPOLIS-HASTINGS TO LANGEVIN
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Evolution from Metropolis-Hastings to Langevin")
print("=" * 80)

print("""
RECALL: METROPOLIS-HASTINGS (from Bayesian Inference)
----------------------------------------------------
1. Propose: x' ~ q(x'|x_t)  (e.g., x' = x_t + N(0, σ²))
2. Accept with probability: α = min(1, p(x')/p(x_t))
3. If accepted: x_{t+1} = x', else x_{t+1} = x_t

PROBLEMS:
- Acceptance rate can be low (many rejected proposals)
- Requires computing p(x), including normalization
- Random walk proposal inefficient

LANGEVIN MCMC: GRADIENT-BASED IMPROVEMENT
----------------------------------------
Key insight: Use gradient information!

Instead of random walk:
x' = x_t + N(0, σ²)

Use informed proposal:
x' = x_t + ε * ∇log p(x_t) + N(0, 2ε)

Benefits:
- Proposals move toward high probability (gradient drift)
- Still explores (Gaussian noise)
- In limit ε→0, proposals always accepted!
- Only needs SCORE, not full probability!

This is UNADJUSTED Langevin Algorithm (ULA).
""")

# Demonstrate difference
def metropolis_hastings_1d(target_logpdf, n_samples=5000, sigma=0.5):
    """Standard Metropolis-Hastings with random walk"""
    samples = [0.0]  # Start at origin
    x = 0.0
    n_accepted = 0
    
    for _ in range(n_samples):
        # Propose
        x_prop = x + np.random.randn() * sigma
        
        # Accept/reject
        log_alpha = target_logpdf(x_prop) - target_logpdf(x)
        if np.log(np.random.rand()) < log_alpha:
            x = x_prop
            n_accepted += 1
        
        samples.append(x)
    
    acceptance_rate = n_accepted / n_samples
    return np.array(samples), acceptance_rate

def langevin_mcmc_1d(target_logpdf, score_fn, n_samples=5000, epsilon=0.01):
    """Langevin MCMC using gradient information"""
    samples = [0.0]
    x = 0.0
    
    for _ in range(n_samples):
        # Langevin update
        x = x + epsilon * score_fn(x) + np.sqrt(2 * epsilon) * np.random.randn()
        samples.append(x)
    
    return np.array(samples)

# Target: Mixture of Gaussians
def target_logpdf(x):
    """Log PDF of mixture: 0.3*N(-2,0.5²) + 0.7*N(2,0.5²)"""
    from scipy.stats import norm
    p1 = 0.3 * norm.pdf(x, -2, 0.5)
    p2 = 0.7 * norm.pdf(x, 2, 0.5)
    return np.log(p1 + p2 + 1e-10)

def score_fn(x):
    """Score of mixture"""
    from scipy.stats import norm
    p1 = 0.3 * norm.pdf(x, -2, 0.5)
    p2 = 0.7 * norm.pdf(x, 2, 0.5)
    
    s1 = -(x + 2) / 0.25  # Score of N(-2, 0.5²)
    s2 = -(x - 2) / 0.25  # Score of N(2, 0.5²)
    
    total_p = p1 + p2 + 1e-10
    return (p1 * s1 + p2 * s2) / total_p

print("\nComparison: Metropolis-Hastings vs Langevin MCMC")
print("-" * 80)

# Run both
mh_samples, acc_rate = metropolis_hastings_1d(target_logpdf, n_samples=5000)
langevin_samples = langevin_mcmc_1d(target_logpdf, score_fn, n_samples=5000)

print(f"Metropolis-Hastings acceptance rate: {acc_rate:.2%}")
print(f"Langevin MCMC: No rejection (always accepts in continuous limit)")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# True distribution
x_plot = np.linspace(-4, 4, 1000)
true_pdf = np.exp([target_logpdf(x) for x in x_plot])
true_pdf /= np.trapezoid(true_pdf, x_plot)

# Plot 1: MH trajectory
axes[0, 0].plot(mh_samples[:500], alpha=0.7, linewidth=0.5)
axes[0, 0].set_title('Metropolis-Hastings Trajectory', fontweight='bold')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel(r'$x$')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Langevin trajectory  
axes[0, 1].plot(langevin_samples[:500], alpha=0.7, linewidth=0.5)
axes[0, 1].set_title('Langevin MCMC Trajectory', fontweight='bold')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel(r'$x$')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: MH histogram
axes[1, 0].hist(mh_samples[1000:], bins=50, density=True, alpha=0.6, label='MH samples')
axes[1, 0].plot(x_plot, true_pdf, 'r-', linewidth=2, label='True distribution')
axes[1, 0].set_title(f'Metropolis-Hastings\n(Acc. rate: {acc_rate:.1%})', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Langevin histogram
axes[1, 1].hist(langevin_samples[1000:], bins=50, density=True, alpha=0.6, label='Langevin samples')
axes[1, 1].plot(x_plot, true_pdf, 'r-', linewidth=2, label='True distribution')
axes[1, 1].set_title('Langevin MCMC\n(Gradient-guided)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_mh_vs_langevin.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 03_mh_vs_langevin.png")
plt.close()

print("""
KEY OBSERVATIONS:
- Langevin explores more efficiently using gradient information
- Both converge to target distribution
- Langevin can handle complex, multimodal distributions better
- Acceptance rate trade-off in M-H doesn't apply to Langevin
""")

# ============================================================================
# SECTION 2: LANGEVIN DYNAMICS THEORY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Theory of Langevin Dynamics")
print("=" * 80)

print("""
CONTINUOUS-TIME LANGEVIN DYNAMICS:
---------------------------------
Stochastic Differential Equation (SDE):

dx_t = ∇_x log p(x_t) dt + √2 dW_t

where W_t is standard Brownian motion.

INTUITION:
- Deterministic drift: ∇log p(x) pulls toward high probability
- Stochastic diffusion: √2 dW_t explores the space
- Balance ensures convergence to p(x)

FOKKER-PLANCK EQUATION:
----------------------
The distribution p_t(x) of x_t evolves as:

∂p_t/∂t = -∇·(p_t ∇log p) + Δp_t
        = ∇·(p_t ∇log(p/p_t))

At equilibrium (∂p_t/∂t = 0), we have p_t = p!

DISCRETE-TIME VERSION:
--------------------
Euler-Maruyama discretization:

x_{t+1} = x_t + ε ∇_x log p(x_t) + √(2ε) z_t

where z_t ~ N(0, I) and ε is step size.

CONVERGENCE:
- As ε→0, discrete process → continuous Langevin SDE
- For small enough ε, converges to target distribution
- Convergence rate depends on properties of p(x)

PRACTICAL CONSIDERATIONS:
- Step size ε: Too large = instability, too small = slow convergence
- Number of steps: More steps = better samples, but slower
- Initialization: Can start from any distribution (e.g., N(0,I))

CONNECTION TO SCORE MATCHING:
----------------------------
Since we learned s_θ(x) ≈ ∇log p(x) from data,
we can sample via:

x_{t+1} = x_t + ε s_θ(x_t) + √(2ε) z_t

This is SCORE-BASED SAMPLING!
No need for normalized probability!
""")

# ============================================================================
# SECTION 3: IMPLEMENTING LANGEVIN SAMPLING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Langevin Sampling with Learned Scores")
print("=" * 80)

print("\nExample: 2D Swiss Roll with Learned Score Network")
print("-" * 80)

# Generate Swiss roll
def make_swiss_roll(n_samples=2000, noise=0.1):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    X = np.stack([x, y], axis=1)
    X += noise * np.random.randn(n_samples, 2)
    return X

swiss_data = make_swiss_roll(2000)

# Train score network (simplified from Module 02)
class ScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.net(x)

def train_score_dsm(data, sigma=0.5, n_epochs=1000):
    """Train score network via DSM"""
    model = ScoreNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_tensor = torch.FloatTensor(data)
    
    for epoch in range(n_epochs):
        # DSM loss
        noise = torch.randn_like(data_tensor)
        noisy_data = data_tensor + sigma * noise
        predicted_score = model(noisy_data)
        target_score = -noise / sigma
        
        loss = torch.mean((predicted_score - target_score) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
    
    return model

print("Training score network...")
score_net = train_score_dsm(swiss_data, sigma=0.5, n_epochs=1000)
print("✓ Training complete!")

# Langevin sampling function
def langevin_sampling(score_model, n_samples=500, n_steps=1000, epsilon=0.01, 
                     init_x=None):
    """
    Sample using Langevin dynamics with learned score.
    
    Args:
        score_model: Trained score network
        n_samples: Number of samples to generate
        n_steps: Number of Langevin steps
        epsilon: Step size
        init_x: Initial samples (if None, use N(0, I))
    
    Returns:
        samples: Generated samples [n_samples, dim]
        trajectory: Full sampling trajectory [n_steps, n_samples, dim]
    """
    if init_x is None:
        x = torch.randn(n_samples, 2) * 3  # Initialize from wider distribution
    else:
        x = init_x.clone()
    
    trajectory = [x.clone().detach().numpy()]
    
    for step in range(n_steps):
        with torch.no_grad():
            score = score_model(x)
        
        # Langevin update
        x = x + epsilon * score + np.sqrt(2 * epsilon) * torch.randn_like(x)
        
        if step % 100 == 0:
            trajectory.append(x.clone().detach().numpy())
    
    samples = x.detach().numpy()
    return samples, trajectory

print("\nGenerating samples via Langevin dynamics...")
samples, trajectory = langevin_sampling(score_net, n_samples=500, n_steps=1000, epsilon=0.01)
print("✓ Sampling complete!")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: True data
axes[0].scatter(swiss_data[:, 0], swiss_data[:, 1], s=1, alpha=0.5, c='blue')
axes[0].set_title('True Data Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel(r'$x_1$')
axes[0].set_ylabel(r'$x_2$')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Plot 2: Generated samples
axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='red')
axes[1].set_title('Generated via Langevin Sampling', fontsize=14, fontweight='bold')
axes[1].set_xlabel(r'$x_1$')
axes[1].set_ylabel(r'$x_2$')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# Plot 3: Comparison
axes[2].scatter(swiss_data[:, 0], swiss_data[:, 1], s=1, alpha=0.3, c='blue', label='True')
axes[2].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c='red', label='Generated')
axes[2].set_title('Overlay Comparison', fontsize=14, fontweight='bold')
axes[2].set_xlabel(r'$x_1$')
axes[2].set_ylabel(r'$x_2$')
axes[2].legend()
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_langevin_sampling_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 03_langevin_sampling_results.png")
plt.close()

# Visualize trajectory evolution
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (ax, traj_snapshot) in enumerate(zip(axes, trajectory)):
    step = idx * 100
    ax.scatter(traj_snapshot[:, 0], traj_snapshot[:, 1], s=1, alpha=0.5, c='purple')
    ax.scatter(swiss_data[:, 0], swiss_data[:, 1], s=0.5, alpha=0.1, c='blue')
    ax.set_title(f'Step {step}', fontsize=12, fontweight='bold')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Langevin Sampling Evolution Over Time', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_langevin_trajectory.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 03_langevin_trajectory.png")
plt.close()

print("""
✓ Langevin sampling successfully generated samples from Swiss roll!

Key observations:
- Started from random Gaussian noise
- Gradually moved toward data distribution via score guidance
- Final samples match true data distribution well
- This is exactly how diffusion models generate!
""")

# ============================================================================
# SECTION 4: CONNECTION TO DIFFUSION MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Connection to Diffusion Models")
print("=" * 80)

print("""
DIFFUSION MODELS = MULTI-SCALE LANGEVIN DYNAMICS
-----------------------------------------------

Key insight: Use Langevin sampling at MULTIPLE noise levels!

FORWARD PROCESS (Diffusion):
x_0 → x_1 → x_2 → ... → x_T
Add noise gradually until x_T ~ N(0, I)

REVERSE PROCESS (Generation):
x_T ← x_{T-1} ← ... ← x_1 ← x_0
Denoise gradually using learned scores!

REVERSE STEP:
x_{t-1} = x_t + ε * s_θ(x_t, t) + √(2ε) * z_t

where s_θ(x_t, t) is the score at noise level t.

THIS IS EXACTLY LANGEVIN DYNAMICS!
Just with TIME-DEPENDENT scores!

ANNEALED LANGEVIN DYNAMICS:
--------------------------
Use sequence of noise levels: σ_1 > σ_2 > ... > σ_L

For each level i:
1. Learn score s_θ(x, σ_i)
2. Run Langevin: x ← x + ε * s_θ(x, σ_i) + √(2ε) * z

Start with high noise (easy to sample, imprecise)
Gradually reduce noise (harder to sample, more precise)

This is the foundation of:
- Score-Based Generative Models (Song & Ermon, 2019)
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- All modern diffusion models!

WHAT WE'VE LEARNED SO FAR:
------------------------
✓ Module 01: What are score functions?
✓ Module 02: How to learn them (score matching)?
✓ Module 03: How to use them (Langevin sampling)?

NEXT: Multi-scale framework for real applications!
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
1. Langevin dynamics = gradient-based MCMC sampling
2. Uses score functions to guide sampling
3. More efficient than random-walk Metropolis-Hastings
4. Works with learned scores from score matching
5. Foundation for diffusion model sampling!

KEY FORMULAS:
------------
1. Continuous Langevin SDE:
   dx_t = ∇log p(x_t) dt + √2 dW_t

2. Discrete Langevin MCMC:
   x_{t+1} = x_t + ε * ∇log p(x_t) + √(2ε) * z_t

3. Score-based sampling:
   x_{t+1} = x_t + ε * s_θ(x_t) + √(2ε) * z_t

4. Annealed Langevin (preview):
   Use scores at multiple noise levels σ_1 > ... > σ_L

FILES GENERATED:
---------------
1. 03_mh_vs_langevin.png - Comparison with Metropolis-Hastings
2. 03_langevin_sampling_results.png - Generated samples
3. 03_langevin_trajectory.png - Evolution over time
""")

print("\n" + "=" * 80)
print("EXERCISES")
print("=" * 80)

print("""
EXERCISE 1: Convergence Analysis
-------------------------------
For 1D Gaussian N(0,1):
a) Run Langevin with different step sizes ε
b) Measure convergence rate (KL divergence to target)
c) Plot convergence vs ε
d) Find optimal step size

EXERCISE 2: Multimodal Distributions
-----------------------------------
Create 2D mixture with 4 Gaussians (corners of square):
a) Train score network via DSM
b) Run Langevin sampling
c) Does it visit all modes?
d) Try different initializations and step sizes

EXERCISE 3: Metropolis-Adjusted Langevin
---------------------------------------
Implement MALA (adds Metropolis acceptance):
a) Propose via Langevin: x' = x + ε*∇log p(x) + √(2ε)*z
b) Accept with M-H probability
c) Compare with unadjusted Langevin
d) When does MALA help?

EXERCISE 4: Score Errors
-----------------------
If score network has errors s_θ(x) ≠ ∇log p(x):
a) How does this affect samples?
b) Implement with intentional score bias
c) Measure distribution mismatch
d) Implications for diffusion models?

EXERCISE 5: Annealing Schedule
-----------------------------
Implement annealed Langevin:
a) Define noise schedule σ_1 > σ_2 > ... > σ_L
b) Train scores for each level (see Module 02)
c) Sample starting from high noise
d) Compare with single-level Langevin

CHALLENGE EXERCISE: 3D Sampling
------------------------------
Extend to 3D:
a) Generate 3D Swiss roll or helix data
b) Train 3D score network
c) Implement Langevin sampling
d) Visualize 3D trajectories
e) How does dimensionality affect convergence?
""")

print("\n" + "=" * 80)
print("NEXT: Multi-Scale Score Modeling")
print("=" * 80)
print("""
We now have all the building blocks:
✓ Score functions (Module 01)
✓ Score matching to learn them (Module 02)
✓ Langevin dynamics to sample (Module 03)

Next: How to make this work for REAL data (images, etc.)?

Answer: Learn scores at MULTIPLE noise levels!
This leads to the full diffusion framework!
""")

print("\n✓ Module 03 complete!")
print("=" * 80)
