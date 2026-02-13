#!/usr/bin/env python3
# ==========================================================
# pca_2d_to_1d_sklearn_demo.py
# ==========================================================
# COMPREHENSIVE PCA TUTORIAL: USING SCIKIT-LEARN
#
# This script demonstrates Principal Component Analysis (PCA) using
# scikit-learn's high-level PCA class. This is the "production-ready"
# way to use PCA, with robust implementations and convenient methods.
#
# WHAT THIS SCRIPT DOES:
# - Generates the same 2D dataset as the NumPy version
# - Computes PCA using sklearn.decomposition.PCA
# - Reduces dimensionality from 2D to 1D
# - Reconstructs data using inverse_transform()
# - Visualizes the complete pipeline
#
# ADVANTAGES OF SKLEARN PCA:
# 1. Clean, high-level API (fit/transform/inverse_transform)
# 2. Handles edge cases automatically
# 3. Provides useful attributes (explained_variance_ratio_, etc.)
# 4. Well-tested and optimized implementation
# 5. Consistent with other sklearn transformers
#
# COMPARISON WITH MANUAL SVD:
# - sklearn PCA does the same SVD under the hood
# - But provides convenient methods and error handling
# - Recommended for production code
#
# Run: python pca_2d_to_1d_sklearn_demo.py
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================================
# STEP 1: GENERATE SYNTHETIC 2D DATASET
# ==========================================================
# Create the same dataset as in the NumPy version for direct comparison.
# This is a 2D Gaussian cloud with correlation between dimensions.

rng = np.random.default_rng(42)  # Reproducible random seed
n = 150                          # Number of data points

# Distribution parameters
mean_true = np.array([2.0, -1.0])  # Center of the data cloud

# Covariance matrix: defines the shape and correlation structure
# [[3.0, 2.2],   <- var(x₁) = 3.0, cov(x₁,x₂) = 2.2
#  [2.2, 2.0]]   <- cov(x₁,x₂) = 2.2, var(x₂) = 2.0
# The positive off-diagonal (2.2) means x₁ and x₂ are positively correlated
cov_true = np.array([[3.0, 2.2],
                     [2.2, 2.0]])

# Generate samples from multivariate Gaussian
X = rng.multivariate_normal(mean_true, cov_true, size=n)  # Shape: (150, 2)

print(f"Generated dataset: {X.shape}")
print(f"Sample mean: {X.mean(axis=0)}")
print(f"Sample covariance:\n{np.cov(X.T)}")

# ==========================================================
# STEP 2: FIT PCA WITH SCIKIT-LEARN
# ==========================================================
# sklearn's PCA class provides a clean interface for PCA operations.
#
# KEY METHODS:
# - fit(X):              Compute principal components from X
# - transform(X):        Project X onto principal components
# - fit_transform(X):    Fit and transform in one step
# - inverse_transform(X_reduced): Reconstruct from reduced representation
#
# KEY ATTRIBUTES (available after fitting):
# - components_:               Principal directions (rows are components)
# - explained_variance_:       Variance explained by each component
# - explained_variance_ratio_: Fraction of total variance explained
# - mean_:                     Mean of the training data (used for centering)
# - n_components_:             Number of components retained
# - singular_values_:          Singular values from SVD

# Create PCA object with 1 component (2D → 1D reduction)
pca = PCA(n_components=1)
# Other useful parameters:
#   - svd_solver: 'auto', 'full', 'arpack', 'randomized'
#   - whiten: If True, scale components to unit variance
#   - random_state: For reproducibility with randomized solver

# Fit PCA and transform data in one step
# This is equivalent to:
#   pca.fit(X)
#   scores_1d = pca.transform(X)
scores_1d = pca.fit_transform(X)  # Shape: (150, 1)
# Note: scores_1d is 2D with shape (n, 1), not 1D array like NumPy version

print(f"\nPCA fitted!")
print(f"Number of components kept: {pca.n_components_}")
print(f"1D scores shape: {scores_1d.shape}")  # (150, 1)

# ==========================================================
# STEP 3: RECONSTRUCT DATA FROM 1D REPRESENTATION
# ==========================================================
# sklearn provides inverse_transform() to reconstruct the original
# (or approximated) data from the reduced representation.
#
# MATHEMATICAL OPERATION:
# For scores in reduced space, reconstruction is:
#   X_reconstructed = scores @ components + mean
#
# This is exactly what inverse_transform() does internally.

X_recon = pca.inverse_transform(scores_1d)  # Shape: (150, 2)
# Reconstructed points lie on the principal axis in the original 2D space

print(f"Reconstructed data shape: {X_recon.shape}")

# Compute reconstruction error
reconstruction_error = np.mean((X - X_recon) ** 2)
print(f"Mean reconstruction error (MSE): {reconstruction_error:.6f}")

# ==========================================================
# STEP 4: EXTRACT LEARNED PCA COMPONENTS AND STATISTICS
# ==========================================================
# After fitting, PCA object contains useful information about the
# principal components and how much variance they explain.

# Data mean (computed during fit, used for centering)
mu = pca.mean_  # Shape: (2,) - mean of each original feature
print(f"\nLearned mean: {mu}")

# Principal components (directions of maximum variance)
# Note: sklearn stores components as ROWS (unlike our NumPy code with columns)
pc1 = pca.components_[0]  # First (and only) principal component
                          # Shape: (2,) - unit vector in 2D space
print(f"First principal component: {pc1}")
print(f"PC1 is unit vector? {np.allclose(np.linalg.norm(pc1), 1.0)}")

# Explained variance ratio (fraction of total variance)
var_ratio = pca.explained_variance_ratio_[0]
print(f"\nVariance explained by PC1: {var_ratio:.2%}")
print(f"Variance lost (PC2): {(1 - var_ratio):.2%}")

# Singular values (from SVD)
singular_val = pca.singular_values_[0]
print(f"First singular value: {singular_val:.4f}")

# Total explained variance
explained_var = pca.explained_variance_[0]
print(f"Explained variance (absolute): {explained_var:.4f}")

# ==========================================================
# STEP 5: PREPARE PRINCIPAL AXIS LINE FOR VISUALIZATION
# ==========================================================
# Draw a line segment representing the 1D principal subspace
# (the principal axis) through the mean point.
#
# We scale the line by the standard deviation of the 1D scores
# to make it visually span the data nicely.

score_std = scores_1d.std()  # Standard deviation of projected data
print(f"\n1D scores std: {score_std:.4f}")

# Create line endpoints: mean ± (4 * std) * pc1
t = np.linspace(-4.0 * score_std, 4.0 * score_std, 2)  # Two endpoints
# Each endpoint is: mu + t_i * pc1
axis_pts = mu + np.outer(t, pc1)  # Shape: (2, 2)

print(f"Principal axis endpoints:")
print(f"  Start: {axis_pts[0]}")
print(f"  End:   {axis_pts[1]}")

# ==========================================================
# STEP 6: VISUALIZATION
# ==========================================================
# Create a comprehensive visualization showing:
# 1. Original 2D data points
# 2. Reconstructed points on the principal axis
# 3. Orthogonal projection lines (original → reconstructed)
# 4. The principal axis (PC1)
# 5. The mean point
#
# This visualization is identical to the NumPy version,
# demonstrating that sklearn gives the same results!

fig, ax = plt.subplots(figsize=(8, 6))

# Original data points
ax.scatter(X[:, 0], X[:, 1], 
           s=25,           # Size
           alpha=0.6,      # Transparency
           color='C0',     # Blue
           label="Original points",
           zorder=2)       # Drawing order

# Reconstructed (projected) points
ax.scatter(X_recon[:, 0], X_recon[:, 1], 
           s=18,           # Slightly smaller
           alpha=0.9,      # More opaque  
           marker="x",     # X marker
           color='C1',     # Orange
           label="Projection (1D→2D)",
           zorder=3)

# Orthogonal projection lines
# Draw a subset to avoid visual clutter
step = max(1, n // 40)  # Approximately 40 segments
for i in range(0, n, step):
    ax.plot([X[i, 0], X_recon[i, 0]],    # x-coordinates
            [X[i, 1], X_recon[i, 1]],    # y-coordinates
            color='gray',
            linewidth=0.8, 
            alpha=0.6,
            zorder=1)      # Draw behind points

# Principal axis line (PC1)
# This is the 1D subspace that best approximates the 2D data
ax.plot(axis_pts[:, 0], axis_pts[:, 1], 
        color='C2',        # Green
        linewidth=2.0, 
        label="Principal axis (PC1)",
        zorder=4)

# Mean point
# The center of the data, through which the principal axis passes
ax.scatter([mu[0]], [mu[1]], 
           s=70,           # Larger
           edgecolor="k",  # Black edge
           facecolor="none",  # Hollow
           linewidth=2,
           label="Mean",
           zorder=5)

# Labels, title, and formatting
ax.set_title(f"PCA (sklearn): 2D → 1D (Explained Var: {var_ratio:.2%})", 
             fontsize=12, fontweight='bold')
ax.set_xlabel("x₁", fontsize=11)
ax.set_ylabel("x₂", fontsize=11)
ax.axis("equal")  # Equal aspect ratio
ax.legend(loc="best", framealpha=0.9, fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig('pca_2d_to_1d_sklearn.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: pca_2d_to_1d_sklearn.png")

plt.show()

# ==========================================================
# COMPARISON: SKLEARN VS MANUAL SVD
# ==========================================================
# Let's verify that sklearn PCA gives identical results to manual SVD
print("\n" + "="*60)
print("VERIFICATION: SKLEARN PCA = MANUAL SVD")
print("="*60)

# Manual SVD (as in the NumPy version)
Xc = X - X.mean(axis=0)  # Center the data
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
V = Vt.T
pc1_manual = V[:, 0]  # First principal component from SVD
explained_var_manual = (S[0] ** 2) / (n - 1)
var_ratio_manual = explained_var_manual / ((S ** 2) / (n - 1)).sum()

# Compare with sklearn results
print(f"Principal component (sklearn): {pc1}")
print(f"Principal component (manual):  {pc1_manual}")
print(f"Components match? {np.allclose(np.abs(pc1), np.abs(pc1_manual))}")
print(f"  (Note: signs may differ, but direction is the same)")
print(f"\nExplained variance ratio (sklearn): {var_ratio:.6f}")
print(f"Explained variance ratio (manual):  {var_ratio_manual:.6f}")
print(f"Variance ratios match? {np.allclose(var_ratio, var_ratio_manual)}")

# ==========================================================
# KEY ADVANTAGES OF SKLEARN PCA
# ==========================================================
print("\n" + "="*60)
print("WHY USE SKLEARN PCA?")
print("="*60)
print("1. Clean API:")
print("   • fit_transform() instead of manual SVD")
print("   • inverse_transform() for reconstruction")
print("   • Consistent with other sklearn transformers")
print("\n2. Convenience:")
print("   • Automatic centering (no need to subtract mean)")
print("   • explained_variance_ratio_ computed automatically")
print("   • Can specify n_components as int or variance ratio")
print("\n3. Production ready:")
print("   • Well-tested and optimized")
print("   • Handles edge cases (e.g., singular matrices)")
print("   • Integration with sklearn pipelines")
print("\n4. Additional features:")
print("   • Whitening (decorrelate and normalize)")
print("   • Different SVD solvers for efficiency")
print("   • Sparse matrix support")
print("="*60)

# ==========================================================
# SUMMARY
# ==========================================================
print("\n" + "="*60)
print("PCA SUMMARY")
print("="*60)
print(f"Original dimensionality:  2D")
print(f"Reduced dimensionality:   1D") 
print(f"Variance preserved:       {var_ratio:.2%}")
print(f"Reconstruction MSE:       {reconstruction_error:.6f}")
print(f"\nMethod: sklearn.decomposition.PCA")
print(f"  • Easy to use, production-ready")
print(f"  • Same results as manual SVD")
print(f"  • Recommended for most applications")
print("="*60)
