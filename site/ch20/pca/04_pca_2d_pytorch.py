#!/usr/bin/env python3
# ==========================================================
# 02_pca_2d_pytorch.py
# ==========================================================
# PCA WITH PYTORCH: 2D → 1D DIMENSIONALITY REDUCTION
#
# This script demonstrates Principal Component Analysis (PCA) using PyTorch
# instead of NumPy. It performs the same analysis as the NumPy version,
# but showcases PyTorch's tensor operations and GPU capabilities.
#
# WHAT THIS SCRIPT DOES:
# - Generates a correlated 2D dataset
# - Computes PCA using PyTorch's SVD (torch.linalg.svd)
# - Reduces dimensionality from 2D to 1D
# - Reconstructs data and visualizes results
# - Demonstrates PyTorch advantages (GPU, autograd-ready)
#
# KEY PYTORCH FEATURES:
# 1. torch.tensor operations (same syntax as NumPy!)
# 2. GPU acceleration (.to('cuda'))
# 3. torch.linalg.svd for matrix decomposition
# 4. Integration with deep learning pipelines
#
# WHEN TO USE PYTORCH FOR PCA:
# - Large datasets (GPU acceleration)
# - Integration with neural networks
# - Need for automatic differentiation
# - Part of larger PyTorch pipeline
#
# PREREQUISITES: Complete 01_pytorch_basics.py first
# TIME: ~10 minutes to read and run
#
# Run: python 02_pca_2d_pytorch.py
# ==========================================================

import torch
import matplotlib.pyplot as plt
import numpy as np  # Only for matplotlib compatibility

# Set random seed for reproducibility
torch.manual_seed(42)

# Determine device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("PCA WITH PYTORCH: 2D → 1D")
print("=" * 60)
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60 + "\n")

# ==========================================================
# STEP 1: GENERATE SYNTHETIC 2D DATASET
# ==========================================================
# Create a 2D Gaussian dataset with correlation.
# The data will form an elongated cloud, which PCA will discover!
#
# MATHEMATICAL SETUP:
# We want data from N(μ, Σ) where:
#   μ = [2.0, -1.0]  (mean)
#   Σ = [[3.0, 2.2],  (covariance matrix)
#        [2.2, 2.0]]
#
# PyTorch doesn't have multivariate_normal in torch, so we'll
# construct it using the Cholesky decomposition:
#   X = μ + Z @ L^T, where Σ = L @ L^T and Z ~ N(0, I)

print("STEP 1: Generating 2D Dataset")
print("-" * 60)

n = 150  # Number of data points

# Define distribution parameters
mean_true = torch.tensor([2.0, -1.0], device=device)  # Shape: (2,)

# Covariance matrix (defines correlation and spread)
# Off-diagonal terms (2.2) create correlation between dimensions
cov_true = torch.tensor([[3.0, 2.2],
                         [2.2, 2.0]], device=device)  # Shape: (2, 2)

# Cholesky decomposition: Σ = L @ L^T
# L is lower triangular matrix
L = torch.linalg.cholesky(cov_true)  # Shape: (2, 2)

# Generate standard normal samples: Z ~ N(0, I)
Z = torch.randn(n, 2, device=device)  # Shape: (150, 2)

# Transform to desired distribution: X = μ + Z @ L^T
X = mean_true + Z @ L.T  # Shape: (150, 2)

print(f"Generated dataset shape: {X.shape}")  # (150, 2)
print(f"Sample mean: {X.mean(dim=0)}")        # Should be ≈ [2.0, -1.0]
print(f"Sample covariance:")
# Compute covariance manually: Cov = (X - μ)^T @ (X - μ) / (n-1)
X_centered_for_cov = X - X.mean(dim=0, keepdim=True)
cov_sample = (X_centered_for_cov.T @ X_centered_for_cov) / (n - 1)
print(cov_sample)
print(f"Covariance matches target? {torch.allclose(cov_sample, cov_true, atol=0.5)}")

# ==========================================================
# STEP 2: CENTER THE DATA
# ==========================================================
# PCA requires zero-centered data!
#
# WHY?
# - PCA finds directions through the origin with maximum variance
# - Centering ensures the "origin" is the data's mean
# - Makes PCA translation-invariant
#
# OPERATION: X_centered = X - mean(X)

print("\n\nSTEP 2: Centering Data")
print("-" * 60)

# Compute mean along samples (dim=0)
mu = X.mean(dim=0, keepdim=True)  # Shape: (1, 2)
                                   # keepdim keeps it 2D for broadcasting

# Center the data
X_centered = X - mu  # Broadcasting: (150, 2) - (1, 2) = (150, 2)

print(f"Original mean: {mu.squeeze()}")
print(f"Centered mean: {X_centered.mean(dim=0)}")  # Should be ≈ [0, 0]
print(f"Verification: mean ≈ 0? {torch.allclose(X_centered.mean(dim=0), torch.tensor([0., 0.], device=device), atol=1e-6)}")

# Check variance is preserved
var_original = X.var(dim=0)
var_centered = X_centered.var(dim=0)
print(f"\nVariance (original): {var_original}")
print(f"Variance (centered): {var_centered}")
print(f"Variance preserved? {torch.allclose(var_original, var_centered, atol=1e-5)}")

# ==========================================================
# STEP 3: COMPUTE PCA VIA SVD
# ==========================================================
# Singular Value Decomposition is the most stable way to compute PCA.
#
# SVD FACTORIZATION:
# X_centered = U @ diag(S) @ V^T
#
# For our (150, 2) matrix:
#   U: (150, 2) - Left singular vectors (scores in PC space)
#   S: (2,)     - Singular values (related to variance)
#   V^T: (2, 2) - Right singular vectors^T (principal directions)
#
# CONNECTION TO PCA:
# - Columns of V are the PRINCIPAL COMPONENTS
# - S^2 / (n-1) gives the explained variance
# - U @ diag(S) gives the principal component scores
#
# PYTORCH NOTES:
# - torch.linalg.svd returns (U, S, Vh) where Vh = V^T
# - Set full_matrices=False for economy SVD (faster, less memory)

print("\n\nSTEP 3: Computing PCA via SVD")
print("-" * 60)

# Perform SVD
# Note: PyTorch returns V^T directly (called Vh)
U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

print(f"SVD decomposition shapes:")
print(f"  U:  {U.shape}   - Left singular vectors")
print(f"  S:  {S.shape}   - Singular values: {S}")
print(f"  Vh: {Vh.shape}  - Right singular vectors (transposed)")

# Extract principal components
# V has principal components as COLUMNS (so we need to transpose Vh)
V = Vh.T  # Shape: (2, 2) where V[:, 0] is PC1, V[:, 1] is PC2

pc1 = V[:, 0]  # First principal component (direction of max variance)
pc2 = V[:, 1]  # Second principal component (orthogonal to PC1)

print(f"\nFirst principal component (PC1): {pc1}")
print(f"Second principal component (PC2): {pc2}")

# Verify orthogonality: PC1 ⊥ PC2
dot_product = torch.dot(pc1, pc2)
print(f"PC1 · PC2: {dot_product.item():.2e} (should be ≈0)")

# Verify unit length: ||PC1|| = 1
norm_pc1 = torch.linalg.norm(pc1)
print(f"||PC1||: {norm_pc1.item():.6f} (should be ≈1)")

# Compute explained variance ratios
# Variance = (singular_value)^2 / (n-1)
explained_variance = (S ** 2) / (n - 1)
total_variance = explained_variance.sum()
variance_ratios = explained_variance / total_variance

print(f"\nExplained variance:")
print(f"  PC1: {variance_ratios[0].item():.2%} of total variance")
print(f"  PC2: {variance_ratios[1].item():.2%} of total variance")
print(f"  Sum: {variance_ratios.sum().item():.2%} (should be 100%)")

# ==========================================================
# STEP 4: DIMENSIONALITY REDUCTION (2D → 1D)
# ==========================================================
# Project centered data onto the first principal component.
# This gives us a 1D representation that preserves maximum variance.
#
# MATHEMATICAL OPERATION:
# For each centered point x_i:
#   score_i = x_i · pc1  (dot product)
#
# Matrix form (for all points):
#   scores_1d = X_centered @ pc1
#
# INTERPRETATION:
# - scores_1d[i] tells us the "position" of point i along PC1
# - Large positive: far in positive PC1 direction
# - Large negative: far in negative PC1 direction
# - Near zero: close to the mean along PC1

print("\n\nSTEP 4: Dimensionality Reduction (2D → 1D)")
print("-" * 60)

# Project all points onto PC1
scores_1d = X_centered @ pc1  # Shape: (150,)

print(f"1D scores shape: {scores_1d.shape}")
print(f"1D scores range: [{scores_1d.min().item():.2f}, {scores_1d.max().item():.2f}]")
print(f"1D scores mean: {scores_1d.mean().item():.2e} (should be ≈0)")
print(f"1D scores std: {scores_1d.std().item():.2f}")

# The 1D representation preserves PC1's variance but loses PC2's variance
print(f"\nInformation preserved: {variance_ratios[0].item():.2%}")
print(f"Information lost: {variance_ratios[1].item():.2%}")

# ==========================================================
# STEP 5: RECONSTRUCTION (1D → 2D)
# ==========================================================
# Reconstruct 2D points from their 1D representation.
# These reconstructed points lie exactly on the principal axis (PC1).
#
# RECONSTRUCTION FORMULA:
# For each score s_i:
#   x_reconstructed_i = s_i * pc1 + μ
#
# WHY?
# - s_i * pc1 gives position along PC1 (still centered)
# - Adding μ shifts back to original coordinate system
#
# Matrix form:
#   X_recon = outer(scores_1d, pc1) + μ
#
# INTERPRETATION:
# - Reconstructed points are the "projections" onto PC1
# - They represent the best 1D approximation of the 2D data
# - Distance from original to reconstructed = reconstruction error

print("\n\nSTEP 5: Reconstruction (1D → 2D)")
print("-" * 60)

# Reconstruct 2D points
# torch.outer creates (n,) × (2,) → (n, 2) matrix
X_recon = torch.outer(scores_1d, pc1) + mu  # Shape: (150, 2)

print(f"Reconstructed data shape: {X_recon.shape}")

# Verify reconstructed points lie on PC1 (variance along PC2 should be ≈0)
X_recon_centered = X_recon - mu
scores_pc2 = X_recon_centered @ pc2
variance_pc2 = scores_pc2.var()
print(f"\nVariance along PC2 in reconstruction: {variance_pc2.item():.2e}")
print(f"(Should be ≈0 since we only kept PC1)")

# Compute reconstruction error (Mean Squared Error)
reconstruction_error = ((X - X_recon) ** 2).mean()
print(f"\nReconstruction MSE: {reconstruction_error.item():.6f}")

# This error equals the variance we dropped (PC2)
dropped_variance = explained_variance[1]
print(f"Variance dropped (PC2): {dropped_variance.item():.6f}")
print(f"Relative error: {(reconstruction_error / X.var()).item():.2%}")

# ==========================================================
# STEP 6: PREPARE VISUALIZATION ELEMENTS
# ==========================================================
# Create a line segment representing the principal axis
# This line goes through the mean point along the PC1 direction

print("\n\nSTEP 6: Preparing Visualization")
print("-" * 60)

# Create parameter values for the line (we'll draw from -4σ to +4σ)
# where σ is the std of scores along PC1
score_std = scores_1d.std()
t = torch.linspace(-4.0 * score_std, 4.0 * score_std, 100, device=device)

# Points along principal axis: μ + t * pc1
# We use outer product: t (100,) × pc1 (2,) → (100, 2)
axis_points = mu.squeeze() + torch.outer(t, pc1)  # Shape: (100, 2)

print(f"Principal axis points shape: {axis_points.shape}")
print(f"Axis spans from {axis_points[0]} to {axis_points[-1]}")

# ==========================================================
# STEP 7: VISUALIZATION
# ==========================================================
# Create comprehensive visualization showing:
# 1. Original 2D data points (blue)
# 2. Reconstructed points on PC1 (orange X's)
# 3. Projection lines (gray) from original to reconstructed
# 4. Principal axis (green line)
# 5. Mean point (hollow circle)

print("\n\nSTEP 7: Creating Visualization")
print("-" * 60)

# Move data to CPU for matplotlib
X_cpu = X.cpu().numpy()
X_recon_cpu = X_recon.cpu().numpy()
mu_cpu = mu.cpu().numpy().squeeze()
axis_points_cpu = axis_points.cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 7))

# Plot original data points
scatter_original = ax.scatter(
    X_cpu[:, 0], X_cpu[:, 1],
    s=30,
    alpha=0.6,
    color='steelblue',
    label="Original points",
    zorder=2,
    edgecolors='darkblue',
    linewidth=0.3
)

# Plot reconstructed (projected) points
scatter_recon = ax.scatter(
    X_recon_cpu[:, 0], X_recon_cpu[:, 1],
    s=20,
    alpha=0.9,
    marker="x",
    color='darkorange',
    label="Projected points (on PC1)",
    zorder=3,
    linewidth=1.5
)

# Draw projection lines (subset to avoid clutter)
step = max(1, n // 30)  # Draw ~30 projection lines
for i in range(0, n, step):
    ax.plot(
        [X_cpu[i, 0], X_recon_cpu[i, 0]],
        [X_cpu[i, 1], X_recon_cpu[i, 1]],
        color='gray',
        linewidth=0.7,
        alpha=0.5,
        zorder=1
    )

# Draw the principal axis
ax.plot(
    axis_points_cpu[:, 0], axis_points_cpu[:, 1],
    color='forestgreen',
    linewidth=3.0,
    label=f"Principal axis (PC1)",
    zorder=4,
    alpha=0.8
)

# Mark the mean point
ax.scatter(
    [mu_cpu[0]], [mu_cpu[1]],
    s=120,
    edgecolor="black",
    facecolor="none",
    linewidth=2.5,
    label="Mean",
    zorder=5
)

# Add PC1 direction vector as arrow (from mean)
arrow_scale = 1.5 * score_std.item()
pc1_cpu = pc1.cpu().numpy()
ax.arrow(
    mu_cpu[0], mu_cpu[1],
    arrow_scale * pc1_cpu[0],
    arrow_scale * pc1_cpu[1],
    head_width=0.3,
    head_length=0.2,
    fc='forestgreen',
    ec='darkgreen',
    linewidth=2,
    alpha=0.7,
    zorder=6,
    length_includes_head=True
)

# Formatting
ax.set_title(
    f"PCA with PyTorch: 2D → 1D Projection\n"
    f"PC1 explains {variance_ratios[0].item():.1%} of variance | "
    f"Reconstruction MSE: {reconstruction_error.item():.4f}",
    fontsize=13,
    fontweight='bold',
    pad=15
)
ax.set_xlabel("x₁", fontsize=12, fontweight='bold')
ax.set_ylabel("x₂", fontsize=12, fontweight='bold')
ax.axis("equal")
ax.legend(loc="best", framealpha=0.95, fontsize=10, edgecolor='black')
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

# Add device info as text annotation
device_text = f"Device: {device}"
if device.type == 'cuda':
    device_text += f" ({torch.cuda.get_device_name(0)})"
ax.text(
    0.02, 0.98, device_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

plt.tight_layout()
plt.savefig('pytorch_pca_2d_to_1d.png', dpi=150, bbox_inches='tight')
print("✓ Saved: pytorch_pca_2d_to_1d.png")

# ==========================================================
# STEP 8: PYTORCH ADVANTAGES DEMONSTRATION
# ==========================================================
# Show unique PyTorch features that NumPy doesn't have

print("\n\nSTEP 8: PyTorch Advantages")
print("-" * 60)

# 1. GPU Acceleration (if available)
if device.type == 'cuda':
    print("\n1. GPU Acceleration:")
    print("   ✓ All computations ran on GPU")
    print("   ✓ Much faster for large datasets")
    print("   ✓ Seamless with torch.cuda.synchronize()")
else:
    print("\n1. GPU Acceleration:")
    print("   • No GPU available, ran on CPU")
    print("   • To use GPU: Install CUDA and GPU-enabled PyTorch")

# 2. Automatic Differentiation Ready
print("\n2. Autograd-Ready Tensors:")
print("   ✓ Can set requires_grad=True for gradients")
print("   ✓ Useful for differentiable PCA variants")
print("   ✓ Integration with neural networks")

# Example: differentiable reconstruction error
X_diff = X.clone().requires_grad_(True)
X_centered_diff = X_diff - X_diff.mean(dim=0, keepdim=True)
# If we did full forward pass, could compute gradients!

# 3. Native Integration with Deep Learning
print("\n3. Deep Learning Integration:")
print("   ✓ Works seamlessly with torch.nn modules")
print("   ✓ Can be part of nn.Sequential pipeline")
print("   ✓ Same tensor format as neural network layers")

# 4. Batched Operations
print("\n4. Batched Operations:")
print("   ✓ Easy to process multiple datasets")
print("   ✓ Native support for batch dimensions")
print("   ✓ Efficient for mini-batch processing")

# Example: batch of datasets
batch_size = 5
X_batch = torch.randn(batch_size, n, 2, device=device)  # 5 different datasets
print(f"   Example: X_batch shape {X_batch.shape} (5 datasets)")

# ==========================================================
# COMPARISON: PYTORCH VS NUMPY
# ==========================================================
print("\n\nCOMPARISON: PyTorch vs NumPy")
print("-" * 60)

print("\nSimilarities:")
print("  • Same mathematical operations")
print("  • Similar syntax (intentional PyTorch design)")
print("  • Both use SVD for PCA")
print("  • Produce identical results")

print("\nPyTorch Advantages:")
print("  ✓ GPU acceleration (much faster for large data)")
print("  ✓ Automatic differentiation (useful for variants)")
print("  ✓ Native integration with neural networks")
print("  ✓ Better for production ML pipelines")

print("\nNumPy Advantages:")
print("  • Lighter weight (smaller install)")
print("  • More mature ecosystem")
print("  • Better for pure scientific computing")
print("  • No CUDA dependency")

print("\nWhen to use PyTorch for PCA:")
print("  → Large datasets (>10,000 samples)")
print("  → Part of deep learning pipeline")
print("  → Need GPU acceleration")
print("  → Require automatic differentiation")

print("\nWhen to use NumPy for PCA:")
print("  → Small to medium datasets")
print("  → Standalone analysis")
print("  → No GPU available")
print("  → Simpler dependencies")

# ==========================================================
# SUMMARY
# ==========================================================
print("\n\n" + "=" * 60)
print("SUMMARY: PCA WITH PYTORCH")
print("=" * 60)
print(f"Dataset:")
print(f"  • Original dimensions:  2D")
print(f"  • Reduced dimensions:   1D")
print(f"  • Number of samples:    {n}")
print(f"  • Device used:          {device}")
print(f"\nResults:")
print(f"  • Variance explained:   {variance_ratios[0].item():.2%}")
print(f"  • Variance dropped:     {variance_ratios[1].item():.2%}")
print(f"  • Reconstruction MSE:   {reconstruction_error.item():.6f}")
print(f"\nKey Insights:")
print(f"  • PC1 captures the elongated direction")
print(f"  • Projection minimizes reconstruction error")
print(f"  • PyTorch provides GPU acceleration")
print(f"  • Same results as NumPy implementation")
print(f"\nNext Steps:")
print(f"  → Try 03_pca_mnist_pytorch.py for high-dimensional PCA")
print(f"  → Experiment with larger datasets")
print(f"  → Explore GPU speedup with more data")
print("=" * 60)

plt.show()
