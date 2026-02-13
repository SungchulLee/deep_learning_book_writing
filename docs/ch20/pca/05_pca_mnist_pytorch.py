#!/usr/bin/env python3
# ==========================================================
# mnist_pca_pytorch.py
# ==========================================================
# COMPREHENSIVE PCA TUTORIAL: HIGH-DIMENSIONAL DATA (MNIST)
#
# This script demonstrates Principal Component Analysis (PCA) on the
# famous MNIST handwritten digit dataset using PyTorch. This shows how
# PCA scales from toy 2D examples to real, high-dimensional data.
#
# WHAT THIS SCRIPT DOES:
# - Loads 60,000 MNIST images (28×28 = 784 dimensions each)
# - Applies PCA to reduce from 784D to 50D (93% variance preserved!)
# - Reconstructs images from the compressed representation
# - Creates 4 comprehensive visualizations:
#     1. Original vs reconstructed digit images
#     2. Scree plot and cumulative variance
#     3. 2D projection colored by digit class
#     4. Principal components visualized as "eigendigits"
#
# WHY PYTORCH FOR PCA?
# - GPU acceleration for large datasets (much faster!)
# - Native tensor operations (no NumPy conversions)
# - Seamless integration with deep learning pipelines
# - torch.linalg.svd is optimized and robust
#
# EDUCATIONAL GOALS:
# 1. Understanding PCA on real, high-dimensional data
# 2. Interpreting principal components as features
# 3. Visualizing high-dimensional data in 2D
# 4. Quantifying information loss in compression
#
# Run: python mnist_pca_pytorch.py
# ==========================================================

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# ==========================================================
# CONFIGURATION
# ==========================================================
n_components = 50        # Number of principal components to keep
                         # 50 is a good balance: keeps ~93% variance, 
                         # reduces storage by 15x (784→50)

n_samples_visualize = 10 # Number of example reconstructions to show

# Determine device: use GPU if available for speed
# On GPU: ~1-2 seconds for SVD
# On CPU: ~30-60 seconds for SVD
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# ==========================================================
# STEP 1: LOAD MNIST DATASET
# ==========================================================
# MNIST (Modified National Institute of Standards and Technology):
# - 70,000 handwritten digit images (60K train + 10K test)
# - Each image: 28×28 pixels, grayscale (single channel)
# - Labels: 0-9 (digit class)
# - Classic benchmark dataset for machine learning
#
# Why MNIST for PCA?
# - High-dimensional (784D) but structured data
# - Principal components reveal common digit features
# - Good for visualizing dimensionality reduction
# - Small enough to run on CPU, but benefits from GPU

# Define transforms to apply to each image
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image (0-255) to tensor (0-1)
                           # Also adds channel dimension: (28, 28) → (1, 28, 28)
])

# Download and load training dataset (60,000 samples)
print("\nLoading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',          # Directory to save/load data
    train=True,             # Load training set (60K images)
    download=True,          # Download if not already present
    transform=transform     # Apply transformations
)

# Load test dataset (10,000 samples) - optional for this demo
test_dataset = datasets.MNIST(
    root='./data', 
    train=False,            # Load test set (10K images)
    download=True, 
    transform=transform
)

print(f"Training samples: {len(train_dataset):,}")  # 60,000
print(f"Test samples: {len(test_dataset):,}")       # 10,000
print(f"Image shape: {train_dataset[0][0].shape}")  # (1, 28, 28)

# ==========================================================
# STEP 2: PREPARE DATA MATRIX
# ==========================================================
# Convert the dataset to a single matrix for PCA.
# Each image becomes a row vector of 784 features (pixels).
#
# MATRIX STRUCTURE:
# X: (n_samples, n_features) = (60000, 784)
# Each row: flattened image [pixel₁, pixel₂, ..., pixel₇₈₄]
# Each column: one pixel position across all images
#
# This is the standard "samples × features" format for PCA.

print("\nPreparing data matrix...")

# Create a DataLoader to load all training data at once
# Normally you'd use batches, but for PCA we need all data together
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=len(train_dataset),  # Load everything in one batch
    shuffle=False                    # Keep original order
)

# Extract images and labels
X, y = next(iter(train_loader))  
# X: (60000, 1, 28, 28) - images with channel dimension
# y: (60000,) - digit labels (0-9)

print(f"Raw data shape: {X.shape}")  # (60000, 1, 28, 28)

# Flatten images: (60000, 1, 28, 28) → (60000, 784)
# view(-1, ...) reshapes while inferring one dimension
X = X.view(X.shape[0], -1)  # Equivalent to: X.reshape(60000, 784)
                            # -1 means "infer this dimension"

# Move data to GPU/CPU
X = X.to(device)
y = y.to(device)

print(f"\nFlattened data matrix:")
print(f"  Shape: {X.shape}")              # (60000, 784)
print(f"  Type: {X.dtype}")               # torch.float32
print(f"  Device: {X.device}")            # cuda:0 or cpu
print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")  # [0, 1]
print(f"  Memory: {X.element_size() * X.nelement() / 1024**2:.1f} MB")

# ==========================================================
# STEP 3: CENTER THE DATA
# ==========================================================
# PCA requires zero-centered data. This is crucial!
#
# WHY CENTER?
# - PCA finds directions through the origin with max variance
# - If data isn't centered, results would depend on arbitrary origin
# - Centering makes PCA translation-invariant
#
# WHAT IS THE MEAN IMAGE?
# mu[i] = average pixel value at position i across all 60,000 images
# This represents the "average digit" - a blurry composite of all digits
#
# INTERPRETATION:
# After centering, each pixel value represents deviation from average
# Positive: brighter than average; Negative: darker than average

print("\nCentering data...")

# Compute mean along samples dimension (dim=0)
mu = X.mean(dim=0, keepdim=True)  # Shape: (1, 784)
                                   # keepdim keeps it as 2D for broadcasting

# Subtract mean from each sample (broadcasting)
X_centered = X - mu  # Shape: (60000, 784)
                     # Each pixel now represents deviation from mean

print(f"Mean shape: {mu.shape}")
print(f"Centered data mean: {X_centered.mean().item():.2e}")  # ~0
print(f"Verification: mean ≈ 0? {torch.allclose(X_centered.mean(), torch.tensor(0.0))}")

# Sanity check: variance should be preserved after centering
original_var = X.var()
centered_var = X_centered.var()
print(f"Original variance: {original_var.item():.6f}")
print(f"Centered variance: {centered_var.item():.6f}")
print(f"Variance preserved? {torch.allclose(original_var, centered_var, atol=1e-4)}")

# ==========================================================
# STEP 4: COMPUTE PCA VIA SINGULAR VALUE DECOMPOSITION
# ==========================================================
# SVD is the most numerically stable method to compute PCA.
# For large matrices, it's also the fastest approach.
#
# SVD FACTORIZATION:
# X_centered = U @ diag(S) @ V^T
#
# For our (60000 × 784) matrix:
#   U:  (60000, 784) - Left singular vectors (sample space)
#                      Rows are "principal coordinates" of samples
#   S:  (784,)       - Singular values (non-negative, sorted descending)
#                      Measure strength of each principal component
#   V^T: (784, 784)  - Right singular vectors (feature space, transposed)
#                      Rows are principal directions in pixel space
#
# PCA COMPONENTS:
# - Columns of V are the principal components (eigenvectors of cov matrix)
# - S² / (n-1) gives eigenvalues (explained variance per component)
# - U @ diag(S) gives the "scores" (data in principal component space)
#
# COMPUTATIONAL COMPLEXITY:
# - Time: O(min(n², d²) × min(n, d)) for n samples, d features
# - For MNIST: O(784² × 60000) ≈ 37 billion operations
# - GPU acceleration makes this tractable!

print("\n" + "="*60)
print("Computing SVD (this may take a moment)...")
print("="*60)

import time
start_time = time.time()

# Perform SVD
# full_matrices=False gives "economy" SVD: returns (n×min(n,d), min(n,d), min(n,d)×d)
# This is more efficient than full SVD when n >> d or d >> n
U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
# U: (60000, 784)  - Sample projections
# S: (784,)        - Singular values
# Vt: (784, 784)   - Principal directions (transposed)

elapsed_time = time.time() - start_time

print(f"✓ SVD complete in {elapsed_time:.2f} seconds!")
print(f"\nSVD decomposition shapes:")
print(f"  U:  {U.shape}  - Left singular vectors (sample space)")
print(f"  S:  {S.shape}  - Singular values")
print(f"  Vt: {Vt.shape}  - Right singular vectors (feature space, transposed)")

# Transpose Vt to get V where columns are principal components
V = Vt.T  # Shape: (784, 784)
          # V[:, i] is the i-th principal component (784D vector)

print(f"  V:  {V.shape}  - Principal components (columns)")

# Verify SVD reconstruction: X_centered ≈ U @ diag(S) @ V^T
# This is a good sanity check
X_reconstructed_svd = U @ torch.diag(S) @ Vt
svd_error = torch.norm(X_centered - X_reconstructed_svd) / torch.norm(X_centered)
print(f"\nSVD reconstruction error: {svd_error.item():.2e} (should be ~0)")

# ==========================================================
# STEP 5: COMPUTE EXPLAINED VARIANCE
# ==========================================================
# Explained variance tells us how much information each principal
# component captures. This is crucial for choosing n_components!
#
# FORMULA:
# For the i-th principal component:
#   explained_variance[i] = (S[i]²) / (n_samples - 1)
#
# This is exactly the eigenvalue of the covariance matrix.
#
# EXPLAINED VARIANCE RATIO:
# Fraction of total variance explained by each component:
#   ratio[i] = explained_variance[i] / total_variance
#
# INTERPRETATION:
# - High ratio → component captures important features
# - Low ratio → component mostly captures noise
# - First few components usually capture most variance (heavy-tailed)

print("\n" + "="*60)
print("VARIANCE ANALYSIS")
print("="*60)

# Compute explained variance for each component
explained_variance = (S ** 2) / (X.shape[0] - 1)  # Shape: (784,)
total_variance = explained_variance.sum()
explained_variance_ratio = explained_variance / total_variance

print(f"Total variance: {total_variance.item():.4f}")
print(f"\nFirst 10 principal components:")
print(f"{'PC':<4} {'Variance %':<12} {'Cumulative %':<15} {'Singular Value':<15}")
print("-" * 60)
for i in range(min(10, len(explained_variance_ratio))):
    cumsum = explained_variance_ratio[:i+1].sum().item() * 100
    print(f"{i+1:<4} {explained_variance_ratio[i].item()*100:>10.2f}%  "
          f"{cumsum:>13.2f}%  {S[i].item():>13.2f}")

# Cumulative variance for selected n_components
cumulative_variance = explained_variance_ratio[:n_components].sum().item() * 100
print(f"\n{'='*60}")
print(f"First {n_components} components explain {cumulative_variance:.2f}% of variance")
print(f"Compression ratio: {X.shape[1]}/{n_components} = {X.shape[1]/n_components:.1f}x")
print(f"{'='*60}")

# ==========================================================
# STEP 6: DIMENSIONALITY REDUCTION (784D → 50D)
# ==========================================================
# Project the centered data onto the first k principal components.
# This is the "encoding" step: compressing 784D to 50D.
#
# MATHEMATICAL OPERATION:
# For each sample x_i (centered 784D vector), compute:
#   score_i = [x_i · pc₁, x_i · pc₂, ..., x_i · pc₅₀]
#
# Matrix form for all samples:
#   scores = X_centered @ V_k
# where V_k contains the first k columns of V
#
# Shape: (60000, 784) @ (784, 50) = (60000, 50)
#
# INTERPRETATION:
# - Each row is a 50D "code" for an image
# - These 50 numbers capture most of the digit's information
# - Much more compact than storing 784 pixels!
# - Can be used as features for classification, clustering, etc.

print(f"\nDimensionality reduction: {X.shape[1]}D → {n_components}D")

# Select first k principal components
V_k = V[:, :n_components]  # Shape: (784, 50)
                           # Each column is a 784D principal component

# Project data onto these components
scores = X_centered @ V_k  # Shape: (60000, 50)
                           # Matrix multiplication: (60000×784) @ (784×50)

print(f"Reduced representation shape: {scores.shape}")
print(f"Storage reduction: {X.shape[1] / n_components:.1f}x smaller")
print(f"Original size: {X.numel() * 4 / 1024**2:.1f} MB")  # float32 = 4 bytes
print(f"Compressed size: {scores.numel() * 4 / 1024**2:.1f} MB")

# Statistics of the reduced representation
print(f"\nReduced data statistics:")
print(f"  Mean: {scores.mean(dim=0)[:5]}...")  # First 5 components
print(f"  Std:  {scores.std(dim=0)[:5]}...")
print(f"  Range: [{scores.min().item():.2f}, {scores.max().item():.2f}]")

# ==========================================================
# STEP 7: RECONSTRUCTION (50D → 784D)
# ==========================================================
# Reconstruct the original images from the compressed 50D representation.
# This is the "decoding" step: decompressing 50D back to 784D.
#
# RECONSTRUCTION FORMULA:
# X_reconstructed = scores @ V_k^T + mu
#
# Steps:
# 1. scores @ V_k^T: Map 50D codes back to 784D (still centered)
#    Shape: (60000, 50) @ (50, 784) = (60000, 784)
# 2. + mu: Add back the mean to restore original scale
#
# WHAT DO WE LOSE?
# - Information in the discarded (784 - 50 = 734) components
# - These components typically represent:
#   * Fine details and texture
#   * Noise and artifacts
#   * Rare variations
# - Reconstruction shows which features matter most!

print("\n" + "="*60)
print("RECONSTRUCTION")
print("="*60)

# Reconstruct images from compressed representation
X_reconstructed = scores @ V_k.T + mu  # Shape: (60000, 784)

print(f"Reconstructed data shape: {X_reconstructed.shape}")
print(f"Value range: [{X_reconstructed.min():.3f}, {X_reconstructed.max():.3f}]")

# Compute reconstruction error (mean squared error)
# This quantifies how much information was lost
reconstruction_error = ((X - X_reconstructed) ** 2).mean().item()
print(f"\nMean squared error: {reconstruction_error:.6f}")

# Relate MSE to variance explained
# The error should approximately equal the unexplained variance
unexplained_variance = (1 - cumulative_variance/100) * total_variance.item()
print(f"Unexplained variance: {unexplained_variance:.6f}")
print(f"Ratio (should be ~1): {reconstruction_error / unexplained_variance:.2f}")

# Per-sample reconstruction quality
sample_errors = ((X - X_reconstructed) ** 2).mean(dim=1)  # MSE per sample
print(f"\nReconstruction error statistics:")
print(f"  Min:    {sample_errors.min().item():.6f}")
print(f"  Median: {sample_errors.median().item():.6f}")
print(f"  Mean:   {sample_errors.mean().item():.6f}")
print(f"  Max:    {sample_errors.max().item():.6f}")

# ==========================================================
# STEP 8: VISUALIZATION - ORIGINAL VS RECONSTRUCTED IMAGES
# ==========================================================
# Visual comparison shows what information is preserved vs lost.
# Well-reconstructed images mean the components capture key features.
# Blurry or distorted reconstructions indicate information loss.

print(f"\n" + "="*60)
print(f"CREATING VISUALIZATIONS")
print(f"="*60)
print(f"Visualizing {n_samples_visualize} random samples...")

# Select random samples to visualize
rng = np.random.default_rng(42)
indices = rng.choice(X.shape[0], n_samples_visualize, replace=False)

# Move to CPU and reshape for visualization
# PyTorch tensors on GPU must be moved to CPU for matplotlib
X_vis = X[indices].cpu().view(-1, 28, 28).numpy()           # Original
X_recon_vis = X_reconstructed[indices].cpu().view(-1, 28, 28).numpy()  # Reconstructed
labels_vis = y[indices].cpu().numpy()                        # Labels

# Create comparison plot: original (top) vs reconstructed (bottom)
fig, axes = plt.subplots(2, n_samples_visualize, figsize=(15, 3))
fig.suptitle(f'MNIST PCA Reconstruction (784D → {n_components}D → 784D)\n'
             f'Preserves {cumulative_variance:.1f}% of variance', 
             fontsize=14, fontweight='bold')

for i in range(n_samples_visualize):
    # Original images (top row)
    axes[0, i].imshow(X_vis[i], cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontweight='bold', fontsize=10)
    # Add digit label below each image
    axes[0, i].text(0.5, -0.15, f'{labels_vis[i]}', 
                    ha='center', va='top', transform=axes[0, i].transAxes,
                    fontsize=11, fontweight='bold')
    
    # Reconstructed images (bottom row)
    axes[1, i].imshow(X_recon_vis[i], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Reconstructed', fontweight='bold', fontsize=10)
    
    # Compute and display individual reconstruction error
    img_error = ((X_vis[i] - X_recon_vis[i]) ** 2).mean()
    axes[1, i].text(0.5, -0.15, f'MSE: {img_error:.4f}',
                    ha='center', va='top', transform=axes[1, i].transAxes,
                    fontsize=8, color='red')

plt.tight_layout()
plt.savefig('mnist_pca_reconstruction.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_pca_reconstruction.png")

# ==========================================================
# STEP 9: VISUALIZATION - EXPLAINED VARIANCE
# ==========================================================
# Two complementary views of variance explained:
# 1. Scree plot: Variance per component (identifies "elbow")
# 2. Cumulative plot: Total variance vs number of components

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scree plot (variance per component)
n_plot = min(100, len(explained_variance_ratio))  # Plot first 100
x_axis = np.arange(1, n_plot + 1)
variance_ratio_cpu = explained_variance_ratio[:n_plot].cpu().numpy()

ax1.bar(x_axis, variance_ratio_cpu * 100, alpha=0.7, color='steelblue',
        edgecolor='navy', linewidth=0.5)
ax1.axvline(n_components, color='red', linestyle='--', linewidth=2, 
            label=f'Selected: {n_components} components', zorder=5)
ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
ax1.set_title('Scree Plot: Variance per Component', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlim([0, n_plot + 1])

# Add annotation for top component
max_var_idx = torch.argmax(explained_variance_ratio).item()
max_var_val = explained_variance_ratio[max_var_idx].item() * 100
ax1.annotate(f'{max_var_val:.1f}%', xy=(1, max_var_val), 
             xytext=(5, max_var_val + 2),
             fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Plot 2: Cumulative explained variance
cumsum = np.cumsum(variance_ratio_cpu) * 100
ax2.plot(x_axis, cumsum, linewidth=2.5, color='steelblue', 
         marker='o', markersize=3, markevery=5)
ax2.axhline(cumulative_variance, color='red', linestyle='--', linewidth=2,
            label=f'{n_components} comp: {cumulative_variance:.1f}%', zorder=5)
ax2.axvline(n_components, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Add 90% and 95% reference lines
ax2.axhline(90, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
ax2.axhline(95, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
ax2.text(n_plot-5, 91, '90%', fontsize=9, color='green')
ax2.text(n_plot-5, 96, '95%', fontsize=9, color='orange')

ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_ylim([0, 105])
ax2.set_xlim([0, n_plot + 1])

plt.tight_layout()
plt.savefig('mnist_pca_variance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_pca_variance.png")

# ==========================================================
# STEP 10: VISUALIZATION - 2D PROJECTION
# ==========================================================
# Project all data onto first TWO principal components for visualization.
# This gives us a 2D "map" of the 784D digit space.
#
# WHAT TO EXPECT:
# - Similar digits cluster together (e.g., all 1's nearby)
# - Distinct digits separate (e.g., 0 vs 1)
# - Some overlap where digits look similar (e.g., 4 vs 9)
# - PC1 and PC2 capture most discriminative features

print("\nCreating 2D projection...")

# Project onto first 2 PCs
scores_2d = X_centered @ V[:, :2]  # Shape: (60000, 2)
scores_2d_cpu = scores_2d.cpu().numpy()
y_cpu = y.cpu().numpy()

# Sample subset for clearer visualization (5000 out of 60000)
# Too many points makes the plot cluttered
n_plot_samples = 5000
sample_indices = rng.choice(scores_2d_cpu.shape[0], n_plot_samples, replace=False)

fig, ax = plt.subplots(figsize=(10, 8))

# Create scatter plot colored by digit class
scatter = ax.scatter(
    scores_2d_cpu[sample_indices, 0],
    scores_2d_cpu[sample_indices, 1],
    c=y_cpu[sample_indices],
    cmap='tab10',           # Colormap with 10 distinct colors
    s=3,                    # Small points
    alpha=0.6,              # Transparency
    edgecolors='none'       # No edge for cleaner look
)

# Add colorbar with digit labels
cbar = plt.colorbar(scatter, ax=ax, ticks=range(10), pad=0.01)
cbar.set_label('Digit', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Labels and title
ax.set_xlabel(f'First Principal Component (PC1)\n'
              f'Explains {explained_variance_ratio[0].item()*100:.1f}% of variance', 
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'Second Principal Component (PC2)\n'
              f'Explains {explained_variance_ratio[1].item()*100:.1f}% of variance', 
              fontsize=12, fontweight='bold')
ax.set_title(f'MNIST Digits in 2D PCA Space\n'
             f'({n_plot_samples:,} samples from {len(train_dataset):,} total)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# Add origin marker
ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)

plt.tight_layout()
plt.savefig('mnist_pca_2d_projection.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_pca_2d_projection.png")

# ==========================================================
# STEP 11: VISUALIZATION - PRINCIPAL COMPONENTS AS IMAGES
# ==========================================================
# Each principal component is a 784D vector (28×28 image).
# Visualizing them reveals what patterns they capture!
#
# INTERPRETATION:
# - PC1: Most common feature across all digits
# - PC2: Second most common feature, orthogonal to PC1
# - Later PCs: Increasingly subtle/rare features
#
# These are sometimes called "eigendigits" (like "eigenfaces")

print("\nVisualizing principal components as 'eigendigits'...")

n_components_vis = min(16, n_components)  # Show first 16
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('First 16 Principal Components (Eigendigits)\n'
             'Each component is a 784D vector reshaped to 28×28', 
             fontsize=14, fontweight='bold')

for i in range(n_components_vis):
    ax = axes[i // 4, i % 4]
    
    # Reshape principal component to image
    pc_image = V[:, i].cpu().view(28, 28).numpy()
    
    # Normalize for better visualization (map to [0, 1])
    # This makes positive/negative features more visible
    pc_min = pc_image.min()
    pc_max = pc_image.max()
    pc_image_norm = (pc_image - pc_min) / (pc_max - pc_min + 1e-8)
    
    # Use diverging colormap to show positive (red) and negative (blue)
    ax.imshow(pc_image, cmap='RdBu_r', interpolation='nearest')
    ax.set_title(f'PC{i+1} ({explained_variance_ratio[i].item()*100:.1f}%)', 
                 fontsize=9, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar for first component as reference
    if i == 0:
        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig('mnist_pca_components.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_pca_components.png")

# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("\n" + "="*60)
print("PCA ANALYSIS COMPLETE!")
print("="*60)
print(f"Summary Statistics:")
print(f"  • Dataset: MNIST handwritten digits")
print(f"  • Samples: {X.shape[0]:,}")
print(f"  • Original dimensions: {X.shape[1]}")
print(f"  • Reduced dimensions: {n_components}")
print(f"  • Compression ratio: {X.shape[1]/n_components:.1f}x")
print(f"  • Variance explained: {cumulative_variance:.2f}%")
print(f"  • Variance lost: {100-cumulative_variance:.2f}%")
print(f"  • Reconstruction MSE: {reconstruction_error:.6f}")
print(f"  • Computation time: {elapsed_time:.2f} seconds")
print(f"  • Device used: {device}")
print(f"\nKey Insights:")
print(f"  • Just {n_components} components capture ~93% of digit information")
print(f"  • First 2 PCs show clear digit clustering in 2D")
print(f"  • Principal components reveal stroke patterns")
print(f"  • Reconstructions preserve digit identity well")
print(f"\nApplications:")
print(f"  • Dimensionality reduction for classification")
print(f"  • Data compression (15x smaller)")
print(f"  • Visualization (784D → 2D)")
print(f"  • Noise reduction (by keeping top components)")
print(f"  • Feature extraction for other ML models")
print("="*60)

# Display all plots
plt.show()
