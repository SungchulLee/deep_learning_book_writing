"""
MODULE 01: Score Functions Basics
==================================

DIFFICULTY: Beginner
TIME: 2-3 hours
PREREQUISITES: 01_Bayesian_Inference (especially conjugate priors and MAP estimation)

LEARNING OBJECTIVES:
-------------------
1. Understand what a score function is and why it matters
2. Connect score functions to Bayesian posterior inference
3. Compute scores analytically for simple distributions
4. Visualize score fields in 2D
5. Understand why scores avoid normalization constants

MATHEMATICAL FOUNDATION:
-----------------------
Definition:
    The score function s(x) is the gradient of the log-probability:
    
    s(x) = ∇_x log p(x)
    
Connection to Bayesian Inference:
    In Bayesian inference, we learned:
    p(θ|D) = p(D|θ)p(θ) / p(D)
    
    The denominator p(D) = ∫ p(D|θ)p(θ) dθ is intractable!
    
    But the score of the posterior is:
    ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)] - ∇_θ log p(D)
                    = ∇_θ log[p(D|θ)p(θ)]    (constant w.r.t. θ!)
    
    The score DOESN'T need the normalization constant!

KEY INSIGHT:
    Scores let us work with unnormalized distributions,
    which is exactly what we need for Bayesian inference!

Author: Sungchul @ Yonsei University
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, multivariate_normal
from matplotlib import cm

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

print("=" * 80)
print("MODULE 01: SCORE FUNCTIONS BASICS")
print("=" * 80)

# ============================================================================
# SECTION 1: DEFINITION AND INTUITION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: What is a Score Function?")
print("=" * 80)

print("""
DEFINITION:
-----------
For a probability distribution p(x), the score function is:

    s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

INTUITION:
---------
- The score points toward regions of HIGHER probability
- At a mode (peak) of p(x), the score is ZERO  
- The magnitude ||s(x)|| indicates how steeply probability changes
- NO NORMALIZATION CONSTANT needed in the score!

CONNECTION TO BAYESIAN INFERENCE:
--------------------------------
Remember from Module 01_Bayesian_Inference:
- Posterior: p(θ|D) ∝ p(D|θ)p(θ)
- We couldn't compute p(D) = ∫ p(D|θ)p(θ) dθ

With scores:
- ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]
- NO NEED for ∫! The score works with unnormalized distributions!
""")

# ============================================================================
# SECTION 2: SIMPLE EXAMPLE - 1D GAUSSIAN
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Score of 1D Gaussian Distribution")
print("=" * 80)

print("""
Example: Gaussian N(μ, σ²)
--------------------------
PDF: p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

Log-PDF: log p(x) = -log(√(2πσ²)) - (x-μ)²/(2σ²)

Score: s(x) = d/dx log p(x) = -(x-μ)/σ²

Key observations:
1. Constant term -log(√(2πσ²)) disappears! (derivative of constant = 0)
2. Score is LINEAR in x
3. Score is ZERO at x=μ (the mode)
4. Score points TOWARD the mean μ
""")

# Demonstrate with computation
mu, sigma = 0.0, 1.0
x_vals = np.linspace(-4, 4, 1000)

# PDF
pdf_vals = norm.pdf(x_vals, mu, sigma)

# Log-PDF
log_pdf_vals = norm.logpdf(x_vals, mu, sigma)

# Score (analytical)
score_vals = -(x_vals - mu) / (sigma ** 2)

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: PDF
axes[0].plot(x_vals, pdf_vals, 'b-', linewidth=2, label='p(x)')
axes[0].fill_between(x_vals, pdf_vals, alpha=0.3)
axes[0].axvline(mu, color='r', linestyle='--', label=f'μ = {mu}')
axes[0].set_ylabel('p(x)', fontsize=12)
axes[0].set_title('Probability Density Function (PDF)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Log-PDF
axes[1].plot(x_vals, log_pdf_vals, 'g-', linewidth=2, label='log p(x)')
axes[1].axvline(mu, color='r', linestyle='--', label=f'μ = {mu}')
axes[1].set_ylabel('log p(x)', fontsize=12)
axes[1].set_title('Log Probability', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Score function
axes[2].plot(x_vals, score_vals, 'purple', linewidth=2, label=r's(x) = \nabla\log p(x)')
axes[2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[2].axvline(mu, color='r', linestyle='--', label=fr'$\mu$ = {mu} (score=0)')
axes[2].fill_between(x_vals, score_vals, alpha=0.3, color='purple')

# Add arrows showing direction
for x_point in [-2, -1, 1, 2]:
    score_at_x = -(x_point - mu) / (sigma ** 2)
    axes[2].arrow(x_point, score_at_x, 0.3 * np.sign(score_at_x), 0,
                  head_width=0.3, head_length=0.15, fc='darkred', ec='darkred', linewidth=2)

axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('s(x)', fontsize=12)
axes[2].set_title(r'Score Function $s(x) = \nabla\log p(x)$\n(Points toward mean $\mu$)', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), '01_score_1d_gaussian.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_score_1d_gaussian.png")
plt.close()

print("\nKEY OBSERVATIONS:")
print("  1. Score is ZERO at the mode (x = μ)")
print("  2. Score POINTS TOWARD the mode:")
print("     - For x < μ: score is POSITIVE (points right)")
print("     - For x > μ: score is NEGATIVE (points left)")
print("  3. Magnitude increases with distance from mode")
print("  4. NO normalization constant √(2πσ²) in the score!")

# ============================================================================
# SECTION 3: 2D GAUSSIAN - VISUALIZING SCORE FIELDS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Score Field for 2D Gaussian")
print("=" * 80)

print("""
For multivariate Gaussian N(μ, Σ):
---------------------------------
Score: s(x) = ∇_x log p(x) = -Σ^(-1)(x - μ)

This is a LINEAR function pointing toward the mean μ!

Interpretation:
- Score vectors point toward high probability regions
- At the mean μ, score is zero
- Covariance Σ shapes the direction and magnitude
""")

# 2D Gaussian parameters
mu_2d = np.array([0, 0])
Sigma_2d = np.array([[1.0, 0.5], [0.5, 1.0]])  # Correlated
Sigma_inv = np.linalg.inv(Sigma_2d)

# Create grid
x1 = np.linspace(-3, 3, 30)
x2 = np.linspace(-3, 3, 30)
X1, X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))

# Compute PDF
rv = multivariate_normal(mu_2d, Sigma_2d)
pdf_2d = rv.pdf(pos)

# Compute score at each point
# s(x) = -Σ^(-1)(x - μ)
score_field = np.zeros((len(x2), len(x1), 2))
for i in range(len(x2)):
    for j in range(len(x1)):
        x = np.array([X1[i, j], X2[i, j]])
        score_field[i, j] = -Sigma_inv @ (x - mu_2d)

# Create visualization
fig = plt.figure(figsize=(16, 6))

# Plot 1: PDF with contours
ax1 = fig.add_subplot(121)
contour = ax1.contourf(X1, X2, pdf_2d, levels=20, cmap='viridis', alpha=0.8)
ax1.contour(X1, X2, pdf_2d, levels=10, colors='white', alpha=0.4, linewidths=0.5)
ax1.plot(mu_2d[0], mu_2d[1], 'r*', markersize=20, label=r'Mean $\mu$')
plt.colorbar(contour, ax=ax1, label=r'$p(x)$')
ax1.set_xlabel(r'$x_1$', fontsize=12)
ax1.set_ylabel(r'$x_2$', fontsize=12)
ax1.set_title('2D Gaussian PDF', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Score field
ax2 = fig.add_subplot(122)
# Background: PDF contours
ax2.contourf(X1, X2, pdf_2d, levels=20, cmap='viridis', alpha=0.3)
ax2.contour(X1, X2, pdf_2d, levels=10, colors='gray', alpha=0.3, linewidths=0.5)

# Score vectors
skip = 2  # Subsample for clarity
ax2.quiver(X1[::skip, ::skip], X2[::skip, ::skip],
           score_field[::skip, ::skip, 0], score_field[::skip, ::skip, 1],
           color='red', alpha=0.8, scale=20, width=0.004)
ax2.plot(mu_2d[0], mu_2d[1], 'r*', markersize=20, label=r'Mean $\mu$ (score=0)')
ax2.set_xlabel(r'$x_1$', fontsize=12)
ax2.set_ylabel(r'$x_2$', fontsize=12)
ax2.set_title(r'Score Function $s(x)=\nabla \log p(x)$' r"\n(Points toward mean $\mu$)", 
                  fontsize=14, fontweight='bold')

ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), '01_score_2d_gaussian_field.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_score_2d_gaussian_field.png")
plt.close()

print("\nKEY OBSERVATIONS:")
print("  1. Score vectors point toward the mean (high probability)")
print("  2. Score is zero at the mean")
print("  3. The covariance structure affects score directions")
print("  4. Longer arrows = steeper probability gradient")

# ============================================================================
# SECTION 4: WHY SCORES MATTER FOR SAMPLING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Why Score Functions Enable Sampling")
print("=" * 80)

print("""
THE FUNDAMENTAL PROBLEM (from Bayesian Inference):
-------------------------------------------------
Given: Posterior p(θ|D) ∝ p(D|θ)p(θ)
Want: Samples from p(θ|D)
Problem: Can't compute normalizing constant p(D)!

THE SCORE SOLUTION:
------------------
Key insight: ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]

The score DOESN'T need p(D)!

LANGEVIN DYNAMICS (Preview of Module 03):
----------------------------------------
If we know the score s(x) = ∇_x log p(x), we can sample via:

    x_{t+1} = x_t + (ε/2) * s(x_t) + √ε * z_t
    
where z_t ~ N(0, I) is Gaussian noise.

Intuition:
- Drift term (ε/2)*s(x_t) moves toward high probability
- Diffusion term √ε*z_t adds randomness
- Together, they explore the distribution!

This is how diffusion models work!
""")

# Demonstrate simple Langevin sampling for 1D Gaussian
print("\nDemonstration: Langevin Sampling from 1D Gaussian")
print("-" * 80)

# Langevin dynamics parameters
n_steps = 1000
epsilon = 0.1
x_current = -3.0  # Start far from mean

# Storage
trajectory = [x_current]

# Run Langevin dynamics
for step in range(n_steps):
    # Score at current position
    score = -(x_current - mu) / (sigma ** 2)
    
    # Langevin update
    x_current = x_current + (epsilon / 2) * score + np.sqrt(epsilon) * np.random.randn()
    trajectory.append(x_current)

trajectory = np.array(trajectory)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Trajectory
axes[0].plot(trajectory, 'b-', alpha=0.6, linewidth=1)
axes[0].axhline(mu, color='r', linestyle='--', linewidth=2, label=f'Target mean μ={mu}')
axes[0].set_xlabel('Step', fontsize=12)
axes[0].set_ylabel('x', fontsize=12)
axes[0].set_title('Langevin Dynamics Trajectory\n(Using score to sample from N(0,1))', 
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Histogram vs true distribution
axes[1].hist(trajectory[100:], bins=50, density=True, alpha=0.6, color='blue', 
             label='Langevin samples')
axes[1].plot(x_vals, pdf_vals, 'r-', linewidth=3, label='True N(0,1)')
axes[1].set_xlabel(r'$x$', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Samples vs True Distribution\n(Langevin correctly samples from target!)', 
                  fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), '01_langevin_preview.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_langevin_preview.png")
plt.close()

print("\n✓ Langevin dynamics successfully sampled from target distribution!")
print("  This preview shows why scores are powerful:")
print("  - Only need score, not normalization")
print("  - Works for any distribution")
print("  - Foundation for diffusion models!")

# ============================================================================
# SECTION 5: CONNECTION TO DIFFUSION MODELS (PREVIEW)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Preview - Connection to Diffusion Models")
print("=" * 80)

print("""
THE BIG PICTURE:
---------------

1. BAYESIAN INFERENCE (Module 01_Bayesian_Inference):
   - p(θ|D) ∝ p(D|θ)p(θ)
   - Can't compute normalizing constant
   - Need sampling methods

2. SCORE FUNCTIONS (This module):
   - s(x) = ∇_x log p(x)
   - Works without normalization
   - Enables Langevin sampling

3. SCORE MATCHING (Next module):
   - Learn score from data
   - Denoising connection

4. MULTI-SCALE SCORES (Later modules):
   - Learn scores at different noise levels
   - σ₁ > σ₂ > ... > σ_L

5. DIFFUSION MODELS (Final modules):
   - Forward: Add noise gradually
   - Reverse: Denoise using learned scores
   - Generation via sampling!

KEY INSIGHT FOR DIFFUSION:
-------------------------
Denoising noisy data = Bayesian posterior inference!

Given: x_noisy = x_clean + σ * noise
Find: p(x_clean | x_noisy)

The score of this posterior is exactly what diffusion models learn!

  ∇_x log p(x|x_noisy) ← This is learned by neural networks!
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
1. Score function: s(x) = ∇_x log p(x)
2. Scores point toward high probability regions
3. Scores don't need normalization constants
4. Connection to Bayesian posterior inference
5. Scores enable sampling via Langevin dynamics
6. Foundation for diffusion models

KEY FORMULAS:
------------
1. Score definition:
   s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

2. Gaussian score:
   For N(μ, σ²): s(x) = -(x-μ)/σ²
   For N(μ, Σ): s(x) = -Σ^(-1)(x-μ)

3. Posterior score (Bayesian):
   ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]  (no p(D) needed!)

4. Langevin dynamics (preview):
   x_{t+1} = x_t + (ε/2)*s(x_t) + √ε*z_t

FILES GENERATED:
---------------
1. 01_score_1d_gaussian.png - Score for 1D Gaussian
2. 01_score_2d_gaussian_field.png - Score field in 2D
3. 01_langevin_preview.png - Sampling using scores
""")

print("\n" + "=" * 80)
print("EXERCISES")
print("=" * 80)

print("""
EXERCISE 1: Analytical Score Computation
---------------------------------------
Compute the score function for:
a) Exponential distribution: p(x) = λ exp(-λx) for x ≥ 0
b) Laplace distribution: p(x) = (1/2b) exp(-|x-μ|/b)
c) Mixture of two Gaussians

EXERCISE 2: Score Properties
---------------------------
Prove that:
a) The score at a mode is zero
b) ∫ p(x) s(x) dx = 0 (mean of score is zero)
c) For Gaussian, score is linear

EXERCISE 3: Implementation
-------------------------
Implement score computation for:
a) 2D mixture of 3 Gaussians
b) Visualize the score field
c) Run Langevin dynamics to sample

EXERCISE 4: Connection to Bayesian Inference
-------------------------------------------
For Beta-Binomial conjugate pair (from 01_Bayesian_Inference):
a) Derive the score of the posterior
b) Show it doesn't need the normalizing constant
c) Compare with direct posterior computation

EXERCISE 5: Score Matching Preview
---------------------------------
If we only have samples from p(x), not the formula:
a) Can we compute s(x) = ∇_x log p(x) directly? Why not?
b) Propose an alternative approach using denoising
c) This motivates the next module!

CHALLENGE EXERCISE: Multi-Modal Distribution
------------------------------------------
Create a 2D "checkerboard" distribution with multiple modes.
a) Define p(x) as mixture of Gaussians on a grid
b) Compute and visualize the score field
c) Run Langevin dynamics - does it explore all modes?
d) What happens with different step sizes ε?
""")

print("\n" + "=" * 80)
print("NEXT MODULE: 02_score_matching_theory.py")
print("=" * 80)
print("""
In the next module, we'll address the key question:

  "How do we learn score functions from data alone?"

This leads to score matching - the foundation of modern generative models!
""")

print("\n✓ Module 01 complete! Generated 3 visualizations.")
print("  Ready for Module 02: Score Matching Theory")
print("=" * 80)
