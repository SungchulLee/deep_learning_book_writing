#!/usr/bin/env python3
# ==========================================================
# pca_2d_to_1d_demo.py
# ==========================================================
# COMPREHENSIVE PCA TUTORIAL: 2D → 1D DIMENSIONALITY REDUCTION
#
# This script demonstrates Principal Component Analysis (PCA) from scratch
# using only NumPy and basic linear algebra (SVD). No sklearn required!
#
# WHAT THIS SCRIPT DOES:
# - Generates a 2D dataset with correlation structure (elongated cloud)
# - Computes PCA using Singular Value Decomposition (SVD)
# - Reduces dimensionality from 2D to 1D (finds the "best-fit line")
# - Reconstructs the data back to 2D from the 1D representation
# - Visualizes the entire process with clear graphics
#
# KEY CONCEPTS DEMONSTRATED:
# 1. Data centering (required for PCA)
# 2. SVD decomposition and its relationship to PCA
# 3. Principal components as directions of maximum variance
# 4. Orthogonal projection onto principal components
# 5. Dimensionality reduction and reconstruction
#
# Run: python pca_2d_to_1d_demo.py
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# STEP 1: GENERATE SYNTHETIC 2D DATASET
# ==========================================================
# We create a 2D Gaussian dataset with correlation between dimensions.
# This simulates real-world data where features are often correlated.
# The elongated shape along one direction is what PCA will discover!

rng = np.random.default_rng(42)  # Reproducible random number generator
n = 150                          # Number of data points

# Define the "true" distribution parameters
mean_true = np.array([2.0, -1.0])  # Center point of the cloud in 2D space

# Covariance matrix defines the shape and orientation of the data cloud
# [[3.0, 2.2],   <- Variance in x₁ = 3.0, Covariance = 2.2
#  [2.2, 2.0]]   <- Covariance = 2.2, Variance in x₂ = 2.0
# The off-diagonal terms (2.2) create correlation between x₁ and x₂
# This makes the data cloud elongated at an angle (not axis-aligned)
cov_true = np.array([[3.0, 2.2],
                     [2.2, 2.0]])  # Positive-definite, correlated

# Generate n samples from the multivariate Gaussian distribution
# Result: X is a (150, 2) matrix where each row is a 2D point
X = rng.multivariate_normal(mean_true, cov_true, size=n)

print(f"Generated dataset: {X.shape}")  # (150, 2)
print(f"Sample mean: {X.mean(axis=0)}")  # Should be close to [2.0, -1.0]
print(f"Sample covariance:\n{np.cov(X.T)}")  # Should approximate cov_true

# ==========================================================
# STEP 2: CENTER THE DATA (CRITICAL FOR PCA!)
# ==========================================================
# PCA requires zero-centered data. Why?
# - PCA finds directions through the origin that maximize variance
# - If data isn't centered, the "origin" would be arbitrary
# - Centering ensures we measure variance relative to the data's mean
#
# Mathematical note: The covariance matrix is computed as:
#   Cov = (1/n) * X_centered^T @ X_centered

mu = X.mean(axis=0)  # Compute mean of each feature (column)
                     # Result: mu is shape (2,) = [mean_x₁, mean_x₂]

Xc = X - mu          # Subtract mean from each point (broadcasting)
                     # Result: Xc has the same shape (150, 2) but mean = [0, 0]

print(f"\nCentered data mean: {Xc.mean(axis=0)}")  # Should be ~[0, 0]
print(f"Verification: mean is close to zero? {np.allclose(Xc.mean(axis=0), 0)}")

# ==========================================================
# STEP 3: COMPUTE PCA VIA SINGULAR VALUE DECOMPOSITION (SVD)
# ==========================================================
# SVD is the most stable and efficient way to compute PCA.
#
# SVD FACTORIZATION:
# For any matrix X_centered (n × d), SVD decomposes it as:
#   X_centered = U @ Σ @ V^T
#
# Where:
#   U  : (n × min(n,d)) - Left singular vectors (sample space)
#   Σ  : (min(n,d),)    - Singular values (diagonal, stored as 1D array S)
#   V^T: (min(n,d) × d) - Right singular vectors^T (feature space)
#
# CONNECTION TO PCA:
# - Columns of V are the PRINCIPAL COMPONENTS (directions)
# - Singular values S relate to explained variance: variance = (S²)/(n-1)
# - The principal components are ordered by decreasing variance
# - U @ Σ gives the "scores" (data projected onto principal components)
#
# In our case: X_centered is (150 × 2), so:
#   U: (150, 2) - coordinates of points in the principal component space
#   S: (2,)     - two singular values (one per dimension)
#   Vt: (2, 2)  - the two principal directions (as rows)

U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
# full_matrices=False gives "economy" SVD: smaller, more efficient

print(f"\nSVD decomposition shapes:")
print(f"  U:  {U.shape}  - Left singular vectors")
print(f"  S:  {S.shape}  - Singular values: {S}")
print(f"  Vt: {Vt.shape}  - Right singular vectors (transposed)")

# Extract principal components
V = Vt.T  # Transpose so COLUMNS are principal directions
          # V is now (2, 2) where V[:, 0] is PC1, V[:, 1] is PC2

pc1 = V[:, 0]  # First principal component (direction of maximum variance)
               # This is a unit vector: ||pc1|| = 1
               # Shape: (2,)

print(f"\nFirst principal component (PC1): {pc1}")
print(f"PC1 is unit vector? {np.allclose(np.linalg.norm(pc1), 1.0)}")

# Compute explained variance ratios (how much variance each PC captures)
explained_variance = (S ** 2) / (n - 1)  # Variance = (singular value)² / (n-1)
total_variance = explained_variance.sum()
variance_ratios = explained_variance / total_variance

print(f"\nExplained variance:")
print(f"  PC1: {variance_ratios[0]:.2%} of total variance")
print(f"  PC2: {variance_ratios[1]:.2%} of total variance")

# ==========================================================
# STEP 4: DIMENSIONALITY REDUCTION (2D → 1D)
# ==========================================================
# Project the centered data onto the first principal component.
# This is the essence of PCA: reducing to fewer dimensions while
# preserving as much variance (information) as possible.
#
# MATHEMATICAL OPERATION:
# For each point x_i (centered), compute its projection onto pc1:
#   score_i = x_i · pc1  (dot product)
#
# This gives us a single scalar for each point (1D representation).
# These scalars are called "scores" or "principal component scores".
#
# Matrix form for all points at once:
#   scores_1d = X_centered @ pc1
# Shape: (150, 2) @ (2,) = (150,)

scores_1d = Xc @ pc1  # Project all points onto PC1
                      # Shape: (150,) - one scalar per point

print(f"\n1D scores shape: {scores_1d.shape}")
print(f"1D scores range: [{scores_1d.min():.2f}, {scores_1d.max():.2f}]")
print(f"1D scores mean: {scores_1d.mean():.2e} (should be ~0)")

# ==========================================================
# STEP 5: RECONSTRUCTION (1D → 2D)
# ==========================================================
# Reconstruct the 2D points from their 1D representation.
# These reconstructed points lie exactly on the principal axis (PC1).
#
# RECONSTRUCTION FORMULA:
# For each score s_i, reconstruct the 2D point:
#   x_reconstructed_i = s_i * pc1 + mu
#
# Why does this work?
# - s_i * pc1 gives the point's position along PC1 (still centered)
# - Adding mu shifts it back to the original coordinate system
#
# Matrix form for all points:
#   X_recon = outer(scores_1d, pc1) + mu
# The outer product creates an (n × 2) matrix where each row is: score_i * pc1

X_recon = np.outer(scores_1d, pc1) + mu  # Shape: (150, 2)
# np.outer(a, b) with a:(n,) and b:(d,) creates (n, d) matrix

print(f"\nReconstructed data shape: {X_recon.shape}")

# Verify that reconstructed points lie on a line
# (they should all be collinear along PC1)
# We can check by seeing if the variance perpendicular to PC1 is zero
pc2 = V[:, 1]  # Second principal component (orthogonal to PC1)
scores_pc2 = (X_recon - mu) @ pc2  # Project reconstructed points onto PC2
print(f"Variance along PC2 in reconstruction: {np.var(scores_pc2):.2e}")
print(f"(Should be ~0 since we only kept PC1)")

# Compute reconstruction error (how much information we lost)
reconstruction_error = np.mean((X - X_recon) ** 2)  # Mean squared error
print(f"\nReconstruction MSE: {reconstruction_error:.6f}")
print(f"This is {(1 - variance_ratios[0])*100:.1f}% of original variance (lost by dropping PC2)")

# ==========================================================
# STEP 6: PREPARE VISUALIZATION OF PRINCIPAL AXIS
# ==========================================================
# To visualize the 1D subspace (principal axis), we draw a line segment
# through the mean point along the PC1 direction.
#
# The line should span the range of the data to look nice.
# We scale by S[0] (the first singular value) to match data spread.

t = np.linspace(-4.0, 4.0, 2)  # Parameter values for line (just 2 endpoints)
                                # Could use any range; this works well visually

# Create points along the principal axis:
# axis_point(t) = mu + (t * scale) * pc1
# where scale = S[0] / sqrt(n) gives appropriate visual span
axis_pts = mu + np.outer(t * S[0] / np.sqrt(n), pc1)
# Shape: (2, 2) - two endpoints defining the principal axis line

print(f"\nPrincipal axis line endpoints:")
print(f"  Start: {axis_pts[0]}")
print(f"  End:   {axis_pts[1]}")

# ==========================================================
# STEP 7: VISUALIZATION
# ==========================================================
# Create a comprehensive plot showing:
# 1. Original 2D data points
# 2. Reconstructed points (projections on PC1)
# 3. The principal axis (best-fit 1D subspace)
# 4. Orthogonal projection lines from original to reconstructed points
# 5. The mean point

fig, ax = plt.subplots(figsize=(8, 6))

# Plot original data points
ax.scatter(X[:, 0], X[:, 1], 
           s=25,           # Point size
           alpha=0.6,      # Transparency
           color='C0',     # Default blue
           label="Original points",
           zorder=2)       # Draw order (higher = on top)

# Plot reconstructed (projected) points on the principal axis
# These show where each point lands when projected onto PC1
ax.scatter(X_recon[:, 0], X_recon[:, 1], 
           s=18,           # Slightly smaller
           alpha=0.9,      # More opaque
           marker="x",     # X marker for distinction
           color='C1',     # Default orange
           label="Projection (1D→2D)",
           zorder=3)

# Draw orthogonal projection lines (original → reconstructed)
# Show only a subset to avoid clutter
step = max(1, n // 40)  # Draw approximately 40 segments
for i in range(0, n, step):
    ax.plot([X[i, 0], X_recon[i, 0]],      # x coordinates
            [X[i, 1], X_recon[i, 1]],      # y coordinates
            color='gray',
            linewidth=0.8, 
            alpha=0.6,
            zorder=1)      # Draw behind points

# Draw the principal axis (the 1D subspace)
# This is the line that best fits the data in the least-squares sense
ax.plot(axis_pts[:, 0], axis_pts[:, 1], 
        color='C2',        # Default green
        linewidth=2.0, 
        label="Principal axis (PC1)",
        zorder=4)

# Mark the mean point
# This is the "center" of the data and the point through which PC1 passes
ax.scatter([mu[0]], [mu[1]], 
           s=70,           # Larger size
           edgecolor="k",  # Black edge
           facecolor="none",  # Hollow center
           linewidth=2,
           label="Mean",
           zorder=5)

# Add labels and formatting
ax.set_title(f"PCA: 2D → 1D Projection and Reconstruction\n"
             f"(PC1 explains {variance_ratios[0]:.1%} of variance)", 
             fontsize=12, fontweight='bold')
ax.set_xlabel("x₁", fontsize=11)
ax.set_ylabel("x₂", fontsize=11)
ax.axis("equal")  # Equal aspect ratio (squares look square)
ax.legend(loc="best", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig('pca_2d_to_1d_demo.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: pca_2d_to_1d_demo.png")

plt.show()

# ==========================================================
# SUMMARY AND KEY TAKEAWAYS
# ==========================================================
print("\n" + "="*60)
print("PCA SUMMARY")
print("="*60)
print(f"Original dimensionality:  2D")
print(f"Reduced dimensionality:   1D")
print(f"Variance preserved:       {variance_ratios[0]:.2%}")
print(f"Reconstruction error:     {reconstruction_error:.6f}")
print(f"\nWhat we learned:")
print(f"  • PCA finds directions of maximum variance")
print(f"  • PC1 is the best 1D summary of 2D data")
print(f"  • Reconstruction shows information loss")
print(f"  • Projection is orthogonal (perpendicular)")
print("="*60)
