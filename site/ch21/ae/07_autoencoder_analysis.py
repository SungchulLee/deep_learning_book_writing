"""
Module 40.7: Comprehensive Autoencoder Analysis

This script provides tools for in-depth analysis of trained autoencoders:
1. Architecture Comparison
2. Capacity vs Reconstruction Trade-offs
3. Latent Space Interpolation and Arithmetic
4. Manifold Learning Visualization
5. Information-Theoretic Analysis
6. Failure Mode Analysis

This module synthesizes concepts from all previous modules and provides
advanced analytical tools for understanding autoencoder behavior.

Time: 55 minutes
Level: Advanced
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from typing import Tuple, List, Dict
import seaborn as sns


# =============================================================================
# PART 1: ARCHITECTURE COMPARISON
# =============================================================================

class ModelComparator:
    """
    Compare multiple autoencoder architectures systematically.
    
    Metrics:
    - Reconstruction error (MSE, MAE)
    - Parameter count
    - Inference time
    - Training time
    - Memory usage
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        model_name: str
    ) -> Dict:
        """
        Evaluate a single model across multiple metrics.
        
        Returns dictionary of metrics for the model.
        """
        model.eval()
        
        # Reconstruction metrics
        mse_losses = []
        mae_losses = []
        inference_times = []
        
        import time
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.view(images.size(0), -1).to(device)
                
                # Time inference
                start_time = time.time()
                reconstructed, _ = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Compute losses
                mse = torch.mean((images - reconstructed) ** 2, dim=1)
                mae = torch.mean(torch.abs(images - reconstructed), dim=1)
                
                mse_losses.extend(mse.cpu().numpy())
                mae_losses.extend(mae.cpu().numpy())
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Compile results
        results = {
            'model_name': model_name,
            'mse_mean': np.mean(mse_losses),
            'mse_std': np.std(mse_losses),
            'mae_mean': np.mean(mae_losses),
            'mae_std': np.std(mae_losses),
            'num_params': num_params,
            'trainable_params': trainable_params,
            'avg_inference_time': np.mean(inference_times),
            'latent_dim': getattr(model, 'latent_dim', None)
        }
        
        self.results[model_name] = results
        return results
    
    def plot_comparison(self):
        """Create comprehensive comparison plots."""
        if not self.results:
            print("No models to compare")
            return
        
        models = list(self.results.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Reconstruction Error
        mse_means = [self.results[m]['mse_mean'] for m in models]
        mse_stds = [self.results[m]['mse_std'] for m in models]
        
        axes[0, 0].bar(range(n_models), mse_means, yerr=mse_stds, 
                       capsize=5, alpha=0.7)
        axes[0, 0].set_xticks(range(n_models))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].set_ylabel('MSE Loss', fontsize=11)
        axes[0, 0].set_title('Reconstruction Error (lower = better)', fontsize=12)
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # 2. Parameter Count
        params = [self.results[m]['num_params'] for m in models]
        
        axes[0, 1].bar(range(n_models), params, alpha=0.7, color='orange')
        axes[0, 1].set_xticks(range(n_models))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Number of Parameters', fontsize=11)
        axes[0, 1].set_title('Model Size', fontsize=12)
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # 3. Inference Time
        times = [self.results[m]['avg_inference_time'] * 1000 for m in models]  # ms
        
        axes[1, 0].bar(range(n_models), times, alpha=0.7, color='green')
        axes[1, 0].set_xticks(range(n_models))
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Inference Time (ms)', fontsize=11)
        axes[1, 0].set_title('Speed (lower = better)', fontsize=12)
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # 4. Efficiency: Reconstruction vs Parameters
        axes[1, 1].scatter(params, mse_means, s=100, alpha=0.7)
        for i, m in enumerate(models):
            axes[1, 1].annotate(m, (params[i], mse_means[i]), 
                               fontsize=9, ha='center', va='bottom')
        axes[1, 1].set_xlabel('Number of Parameters', fontsize=11)
        axes[1, 1].set_ylabel('MSE Loss', fontsize=11)
        axes[1, 1].set_title('Efficiency Trade-off', fontsize=12)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved comparison to 'model_comparison.png'")


# =============================================================================
# PART 2: LATENT SPACE ANALYSIS
# =============================================================================

def analyze_latent_space_geometry(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 2000
):
    """
    Analyze geometric properties of learned latent space.
    
    Metrics:
    - Distance preservation (correlation with pixel space distances)
    - Neighborhood preservation
    - Intrinsic dimensionality
    - Clustering coefficient
    """
    print("\n" + "="*60)
    print("LATENT SPACE GEOMETRY ANALYSIS")
    print("="*60)
    
    model.eval()
    
    # Collect representations
    original_data = []
    latent_data = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            if len(original_data) * test_loader.batch_size >= num_samples:
                break
            
            images_flat = images.view(images.size(0), -1)
            original_data.append(images_flat.numpy())
            
            images_device = images_flat.to(device)
            if hasattr(model, 'encode'):
                latent = model.encode(images_device)
            else:
                _, latent = model(images_device)
            
            # Flatten latent if necessary
            latent_flat = latent.view(latent.size(0), -1)
            latent_data.append(latent_flat.cpu().numpy())
            labels_list.append(labels.numpy())
    
    X_original = np.concatenate(original_data, axis=0)[:num_samples]
    X_latent = np.concatenate(latent_data, axis=0)[:num_samples]
    y = np.concatenate(labels_list, axis=0)[:num_samples]
    
    # 1. Distance preservation
    print("\n1. Distance Preservation Analysis")
    # Sample subset for computational efficiency
    sample_indices = np.random.choice(num_samples, min(500, num_samples), replace=False)
    X_orig_sample = X_original[sample_indices]
    X_lat_sample = X_latent[sample_indices]
    
    # Compute pairwise distances
    dist_original = cdist(X_orig_sample, X_orig_sample, metric='euclidean')
    dist_latent = cdist(X_lat_sample, X_lat_sample, metric='euclidean')
    
    # Flatten and compute correlation
    dist_orig_flat = dist_original[np.triu_indices_from(dist_original, k=1)]
    dist_lat_flat = dist_latent[np.triu_indices_from(dist_latent, k=1)]
    
    correlation = np.corrcoef(dist_orig_flat, dist_lat_flat)[0, 1]
    print(f"Distance correlation: {correlation:.4f}")
    print("(1.0 = perfect preservation, higher = better)")
    
    # 2. Neighborhood preservation
    print("\n2. Neighborhood Preservation")
    k = 10  # Number of neighbors to check
    preservation_scores = []
    
    for i in range(min(100, len(X_orig_sample))):
        # Find k nearest neighbors in original space
        neighbors_orig = np.argsort(dist_original[i])[:k+1]
        # Find k nearest neighbors in latent space
        neighbors_lat = np.argsort(dist_latent[i])[:k+1]
        # Compute overlap
        overlap = len(set(neighbors_orig) & set(neighbors_lat)) / (k + 1)
        preservation_scores.append(overlap)
    
    avg_preservation = np.mean(preservation_scores)
    print(f"Average neighborhood preservation: {avg_preservation:.4f}")
    print(f"(1.0 = perfect, {1/(k+1):.3f} = random)")
    
    # 3. Intrinsic dimensionality estimate (using PCA)
    print("\n3. Intrinsic Dimensionality Estimate")
    pca = PCA()
    pca.fit(X_latent)
    
    # Cumulative explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Find dimensionality capturing 95% variance
    dim_95 = np.argmax(cumsum_var >= 0.95) + 1
    dim_99 = np.argmax(cumsum_var >= 0.99) + 1
    
    print(f"Dimensions for 95% variance: {dim_95}/{X_latent.shape[1]}")
    print(f"Dimensions for 99% variance: {dim_99}/{X_latent.shape[1]}")
    
    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Distance correlation scatter
    sample_points = min(1000, len(dist_orig_flat))
    sample_idx = np.random.choice(len(dist_orig_flat), sample_points, replace=False)
    axes[0].scatter(dist_orig_flat[sample_idx], dist_lat_flat[sample_idx], 
                    alpha=0.3, s=1)
    axes[0].set_xlabel('Original Space Distance', fontsize=11)
    axes[0].set_ylabel('Latent Space Distance', fontsize=11)
    axes[0].set_title(f'Distance Preservation\n(r={correlation:.3f})', fontsize=12)
    axes[0].grid(alpha=0.3)
    
    # Neighborhood preservation histogram
    axes[1].hist(preservation_scores, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(avg_preservation, color='red', linestyle='--', 
                    label=f'Mean: {avg_preservation:.3f}')
    axes[1].set_xlabel('Preservation Score', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Neighborhood Preservation', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Explained variance
    axes[2].plot(range(1, len(cumsum_var) + 1), cumsum_var, marker='o', markersize=3)
    axes[2].axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95%')
    axes[2].axhline(0.99, color='orange', linestyle='--', alpha=0.7, label='99%')
    axes[2].set_xlabel('Number of Components', fontsize=11)
    axes[2].set_ylabel('Cumulative Explained Variance', fontsize=11)
    axes[2].set_title('Intrinsic Dimensionality', fontsize=12)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('latent_geometry_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved analysis to 'latent_geometry_analysis.png'")


# =============================================================================
# PART 3: LATENT SPACE ARITHMETIC
# =============================================================================

def demonstrate_latent_arithmetic(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Demonstrate arithmetic operations in latent space.
    
    Examples:
    - z_male_with_glasses = z_male + (z_glasses - z_no_glasses)
    - Interpolation: z_t = (1-t)*z_1 + t*z_2
    - Averaging: z_avg = mean([z_1, z_2, ..., z_n])
    
    For MNIST: z_8 + (z_9 - z_0) might give an 8 that looks like 9
    """
    print("\n" + "="*60)
    print("LATENT SPACE ARITHMETIC")
    print("="*60)
    
    model.eval()
    
    # Collect examples of specific digits
    digit_examples = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(10):
                mask = labels == i
                if mask.sum() > 0 and len(digit_examples[i]) < 10:
                    imgs = images[mask][:10 - len(digit_examples[i])]
                    digit_examples[i].append(imgs)
            
            # Check if we have enough examples
            if all(len(digit_examples[i]) > 0 for i in range(10)):
                break
    
    # Concatenate and get latent representations
    digit_latents = {}
    for i in range(10):
        imgs = torch.cat(digit_examples[i], dim=0)[:5]  # 5 examples per digit
        imgs_flat = imgs.view(imgs.size(0), -1).to(device)
        
        if hasattr(model, 'encode'):
            latents = model.encode(imgs_flat)
        else:
            _, latents = model(imgs_flat)
        
        # Average over examples
        digit_latents[i] = latents.mean(dim=0, keepdim=True)
    
    # 1. Vector arithmetic: z_0 + (z_8 - z_1) ≈ z_9 ?
    print("\n1. Digit Arithmetic: z_0 + (z_8 - z_1)")
    z_0 = digit_latents[0]
    z_1 = digit_latents[1]
    z_8 = digit_latents[8]
    
    z_result = z_0 + (z_8 - z_1)
    
    # Decode
    if hasattr(model, 'decode'):
        img_0 = model.decode(z_0).cpu().numpy().reshape(28, 28)
        img_1 = model.decode(z_1).cpu().numpy().reshape(28, 28)
        img_8 = model.decode(z_8).cpu().numpy().reshape(28, 28)
        img_result = model.decode(z_result).cpu().numpy().reshape(28, 28)
    else:
        img_0 = model.decoder(z_0).cpu().numpy().reshape(28, 28)
        img_1 = model.decoder(z_1).cpu().numpy().reshape(28, 28)
        img_8 = model.decoder(z_8).cpu().numpy().reshape(28, 28)
        img_result = model.decoder(z_result).cpu().numpy().reshape(28, 28)
    
    # Visualize
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    
    axes[0].imshow(img_0, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('z_0\n(digit 0)', fontsize=11)
    axes[0].axis('off')
    
    axes[1].text(0.5, 0.5, '+', fontsize=20, ha='center', va='center',
                 transform=axes[1].transAxes)
    axes[1].axis('off')
    
    axes[2].imshow(img_8, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('z_8 - z_1\n(8-like, not 1-like)', fontsize=11)
    axes[2].axis('off')
    
    axes[3].text(0.5, 0.5, '=', fontsize=20, ha='center', va='center',
                 transform=axes[3].transAxes)
    axes[3].axis('off')
    
    axes[4].imshow(img_result, cmap='gray', vmin=0, vmax=1)
    axes[4].set_title('Result\n(0 with 8-like features?)', fontsize=11)
    axes[4].axis('off')
    
    plt.suptitle('Latent Space Vector Arithmetic', fontsize=14)
    plt.tight_layout()
    plt.savefig('latent_arithmetic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved to 'latent_arithmetic.png'")
    
    # 2. Interpolation between digits
    print("\n2. Interpolation: 0 → 8")
    num_steps = 10
    interpolations = []
    
    for t in np.linspace(0, 1, num_steps):
        z_interp = (1 - t) * digit_latents[0] + t * digit_latents[8]
        
        if hasattr(model, 'decode'):
            img_interp = model.decode(z_interp).cpu().numpy().reshape(28, 28)
        else:
            img_interp = model.decoder(z_interp).cpu().numpy().reshape(28, 28)
        
        interpolations.append(img_interp)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 1.5))
    for i, img in enumerate(interpolations):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title('0', fontsize=10)
        elif i == num_steps - 1:
            axes[i].set_title('8', fontsize=10)
    
    plt.suptitle('Smooth Interpolation in Latent Space', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig('latent_interpolation_advanced.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to 'latent_interpolation_advanced.png'")


# =============================================================================
# PART 4: FAILURE MODE ANALYSIS
# =============================================================================

def analyze_failure_modes(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_best: int = 10,
    num_worst: int = 10
):
    """
    Analyze when and why the autoencoder fails.
    
    Identifies:
    - Best reconstructions (easiest samples)
    - Worst reconstructions (hardest samples)
    - Common failure patterns
    """
    print("\n" + "="*60)
    print("FAILURE MODE ANALYSIS")
    print("="*60)
    
    model.eval()
    
    # Collect all reconstructions and errors
    all_images = []
    all_reconstructions = []
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_flat = images.view(images.size(0), -1).to(device)
            reconstructed, _ = model(images_flat)
            
            # Compute per-sample error
            errors = torch.mean((images_flat - reconstructed) ** 2, dim=1)
            
            all_images.append(images.numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate
    all_images = np.concatenate(all_images, axis=0)
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_errors = np.concatenate(all_errors, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Find best and worst
    best_indices = np.argsort(all_errors)[:num_best]
    worst_indices = np.argsort(all_errors)[-num_worst:][::-1]
    
    print(f"\nBest reconstruction error: {all_errors[best_indices[0]]:.6f}")
    print(f"Worst reconstruction error: {all_errors[worst_indices[0]]:.6f}")
    print(f"Mean reconstruction error: {np.mean(all_errors):.6f}")
    
    # Analyze error distribution by digit
    print("\nError by digit class:")
    for digit in range(10):
        digit_mask = all_labels == digit
        digit_errors = all_errors[digit_mask]
        print(f"  Digit {digit}: {np.mean(digit_errors):.6f} ± {np.std(digit_errors):.6f}")
    
    # Visualize best and worst
    fig, axes = plt.subplots(4, num_best, figsize=(15, 6))
    
    # Best reconstructions
    for i in range(num_best):
        idx = best_indices[i]
        
        # Original
        axes[0, i].imshow(all_images[idx, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Best\nOriginal', fontsize=9)
        axes[0, i].text(0.5, -0.1, f'{all_labels[idx]}',
                       transform=axes[0, i].transAxes, ha='center', fontsize=8)
        
        # Reconstructed
        axes[1, i].imshow(all_reconstructions[idx].reshape(28, 28), 
                         cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=9)
        axes[1, i].text(0.5, -0.1, f'{all_errors[idx]:.4f}',
                       transform=axes[1, i].transAxes, ha='center', fontsize=8)
    
    # Worst reconstructions
    for i in range(num_worst):
        idx = worst_indices[i]
        
        # Original
        axes[2, i].imshow(all_images[idx, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Worst\nOriginal', fontsize=9)
        axes[2, i].text(0.5, -0.1, f'{all_labels[idx]}',
                       transform=axes[2, i].transAxes, ha='center', fontsize=8)
        
        # Reconstructed
        axes[3, i].imshow(all_reconstructions[idx].reshape(28, 28),
                         cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('Reconstructed', fontsize=9)
        axes[3, i].text(0.5, -0.1, f'{all_errors[idx]:.4f}',
                       transform=axes[3, i].transAxes, ha='center', fontsize=8)
    
    plt.suptitle('Best vs Worst Reconstructions', fontsize=14)
    plt.tight_layout()
    plt.savefig('failure_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved to 'failure_analysis.png'")


# =============================================================================
# PART 5: UTILITIES
# =============================================================================

def load_mnist_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main function for comprehensive analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("COMPREHENSIVE AUTOENCODER ANALYSIS")
    print("="*60)
    
    # Create a simple model for demonstration
    from torch.nn import Sequential, Linear, ReLU, Sigmoid, BatchNorm1d, Dropout
    
    class AnalysisAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 32
            self.encoder = Sequential(
                Linear(784, 256), ReLU(), BatchNorm1d(256),
                Linear(256, 128), ReLU(), BatchNorm1d(128),
                Linear(128, self.latent_dim), ReLU()
            )
            self.decoder = Sequential(
                Linear(self.latent_dim, 128), ReLU(), BatchNorm1d(128),
                Linear(128, 256), ReLU(), BatchNorm1d(256),
                Linear(256, 784), Sigmoid()
            )
        
        def encode(self, x):
            return self.encoder(x)
        
        def decode(self, z):
            return self.decoder(z)
        
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z
    
    # Train model quickly
    print("\nTraining model for analysis...")
    model = AnalysisAE().to(device)
    train_loader, test_loader = load_mnist_data()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(5):
        model.train()
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, _ = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
        print(f"Training epoch {epoch+1}/5 complete")
    
    model.eval()
    
    # Run analyses
    print("\n" + "="*60)
    print("RUNNING ANALYSES")
    print("="*60)
    
    # 1. Latent Space Geometry
    analyze_latent_space_geometry(model, test_loader, device)
    
    # 2. Latent Arithmetic
    demonstrate_latent_arithmetic(model, test_loader, device)
    
    # 3. Failure Analysis
    analyze_failure_modes(model, test_loader, device)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# =============================================================================
# EXERCISES
# =============================================================================

"""
EXERCISE 1: Cross-Architecture Analysis
----------------------------------------
Train and compare:
- Shallow AE (2 layers)
- Deep AE (4-6 layers)
- Convolutional AE
- Sparse AE
- Denoising AE

Compare all metrics from ModelComparator.

Questions:
- Which architecture is most parameter-efficient?
- Which has best reconstruction vs latent_dim trade-off?
- Which has smoothest latent space?


EXERCISE 2: Latent Space Smoothness
------------------------------------
Quantify smoothness of latent space:

a) Sample points along interpolation paths
b) Measure reconstruction error along paths
c) Compare variance of errors (smooth = low variance)

Questions:
- Which architectures learn smoother spaces?
- Does regularization (denoising, sparse) affect smoothness?


EXERCISE 3: Semantic Vector Discovery
--------------------------------------
Identify meaningful directions in latent space:

a) For each digit pair (i,j), compute: d_{ij} = mean(z_i) - mean(z_j)
b) Apply these directions to other digits
c) Visualize results

Questions:
- Are there digit-specific directions?
- Can you find "thickness", "slant", "size" directions?


EXERCISE 4: Robustness Analysis
--------------------------------
Test autoencoder robustness to:
- Input noise
- Occlusion (masking parts of image)
- Rotation
- Scaling

Compare standard vs denoising autoencoder.

Questions:
- Which perturbations are handled well?
- Does denoising training improve general robustness?


EXERCISE 5: Theoretical Capacity
---------------------------------
Analyze relationship between:
- Latent dimension
- Reconstruction error
- Number of training samples

Train models with:
latent_dims = [2, 4, 8, 16, 32, 64, 128, 256]
training_samples = [100, 500, 1000, 5000, 10000, 60000]

Plot heatmap of reconstruction error.

Questions:
- How does optimal latent_dim scale with data size?
- Is there a theoretical lower bound on error?


FINAL PROJECT: Complete Autoencoder Study
------------------------------------------
Conduct comprehensive study:

1. Train 5+ different architectures
2. Run all analyses from this module
3. Create detailed comparison report
4. Identify best architecture for each use case:
   - Compression
   - Anomaly detection
   - Feature extraction
   - Generation (interpolation quality)

Document findings with visualizations and recommendations.
"""
