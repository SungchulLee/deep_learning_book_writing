# Comprehensive Autoencoder Analysis

In-depth analysis tools for understanding trained autoencoder behavior.

---

## Overview

**Analysis Topics:**

1. Architecture Comparison
2. Latent Space Geometry
3. Latent Space Arithmetic
4. Failure Mode Analysis

**Learning Objectives:**

- Systematically compare autoencoder architectures
- Understand geometric properties of latent spaces
- Perform vector arithmetic in latent space
- Identify when and why autoencoders fail

**Time:** ~55 minutes  
**Level:** Advanced

---

## Part 1: Architecture Comparison

### Model Comparator

Compare multiple architectures across multiple metrics:

- Reconstruction error (MSE, MAE)
- Parameter count
- Inference time
- Efficiency trade-offs

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import time

class ModelComparator:
    """Compare multiple autoencoder architectures systematically."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, test_loader, device, model_name):
        """Evaluate a single model across multiple metrics."""
        model.eval()
        
        mse_losses = []
        mae_losses = []
        inference_times = []
        
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
        models = list(self.results.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Reconstruction Error
        mse_means = [self.results[m]['mse_mean'] for m in models]
        mse_stds = [self.results[m]['mse_std'] for m in models]
        axes[0, 0].bar(range(n_models), mse_means, yerr=mse_stds, capsize=5)
        axes[0, 0].set_title('Reconstruction Error (lower = better)')
        
        # 2. Parameter Count
        params = [self.results[m]['num_params'] for m in models]
        axes[0, 1].bar(range(n_models), params, color='orange')
        axes[0, 1].set_title('Model Size')
        
        # 3. Inference Time
        times = [self.results[m]['avg_inference_time'] * 1000 for m in models]
        axes[1, 0].bar(range(n_models), times, color='green')
        axes[1, 0].set_title('Speed (lower = better)')
        
        # 4. Efficiency: MSE vs Parameters
        axes[1, 1].scatter(params, mse_means, s=100)
        for i, m in enumerate(models):
            axes[1, 1].annotate(m, (params[i], mse_means[i]))
        axes[1, 1].set_title('Efficiency Trade-off')
        
        plt.savefig('model_comparison.png', dpi=150)
        plt.show()
```

---

## Part 2: Latent Space Geometry Analysis

### Key Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Distance correlation** | How well latent distances preserve original distances | Close to 1.0 |
| **Neighborhood preservation** | Fraction of neighbors preserved in latent space | Close to 1.0 |
| **Intrinsic dimensionality** | Effective dimensions used in latent space | < latent_dim |

```python
def analyze_latent_space_geometry(model, test_loader, device, num_samples=2000):
    """
    Analyze geometric properties of learned latent space.
    """
    print("LATENT SPACE GEOMETRY ANALYSIS")
    
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
            
            latent_flat = latent.view(latent.size(0), -1)
            latent_data.append(latent_flat.cpu().numpy())
            labels_list.append(labels.numpy())
    
    X_original = np.concatenate(original_data, axis=0)[:num_samples]
    X_latent = np.concatenate(latent_data, axis=0)[:num_samples]
    
    # 1. Distance preservation
    print("\n1. Distance Preservation Analysis")
    sample_indices = np.random.choice(num_samples, min(500, num_samples), replace=False)
    X_orig_sample = X_original[sample_indices]
    X_lat_sample = X_latent[sample_indices]
    
    dist_original = cdist(X_orig_sample, X_orig_sample, metric='euclidean')
    dist_latent = cdist(X_lat_sample, X_lat_sample, metric='euclidean')
    
    dist_orig_flat = dist_original[np.triu_indices_from(dist_original, k=1)]
    dist_lat_flat = dist_latent[np.triu_indices_from(dist_latent, k=1)]
    
    correlation = np.corrcoef(dist_orig_flat, dist_lat_flat)[0, 1]
    print(f"Distance correlation: {correlation:.4f}")
    print("(1.0 = perfect preservation)")
    
    # 2. Neighborhood preservation
    print("\n2. Neighborhood Preservation")
    k = 10  # Number of neighbors
    preservation_scores = []
    
    for i in range(min(100, len(X_orig_sample))):
        neighbors_orig = np.argsort(dist_original[i])[:k+1]
        neighbors_lat = np.argsort(dist_latent[i])[:k+1]
        overlap = len(set(neighbors_orig) & set(neighbors_lat)) / (k + 1)
        preservation_scores.append(overlap)
    
    avg_preservation = np.mean(preservation_scores)
    print(f"Average neighborhood preservation: {avg_preservation:.4f}")
    print(f"(1.0 = perfect, {1/(k+1):.3f} = random)")
    
    # 3. Intrinsic dimensionality
    print("\n3. Intrinsic Dimensionality Estimate")
    pca = PCA()
    pca.fit(X_latent)
    
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    dim_95 = np.argmax(cumsum_var >= 0.95) + 1
    dim_99 = np.argmax(cumsum_var >= 0.99) + 1
    
    print(f"Dimensions for 95% variance: {dim_95}/{X_latent.shape[1]}")
    print(f"Dimensions for 99% variance: {dim_99}/{X_latent.shape[1]}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Distance correlation scatter
    axes[0].scatter(dist_orig_flat[:1000], dist_lat_flat[:1000], alpha=0.3, s=1)
    axes[0].set_xlabel('Original Space Distance')
    axes[0].set_ylabel('Latent Space Distance')
    axes[0].set_title(f'Distance Preservation (r={correlation:.3f})')
    
    # Neighborhood preservation histogram
    axes[1].hist(preservation_scores, bins=20, edgecolor='black')
    axes[1].axvline(avg_preservation, color='red', linestyle='--')
    axes[1].set_title('Neighborhood Preservation')
    
    # Explained variance
    axes[2].plot(range(1, len(cumsum_var) + 1), cumsum_var, marker='o', markersize=3)
    axes[2].axhline(0.95, color='red', linestyle='--', label='95%')
    axes[2].axhline(0.99, color='orange', linestyle='--', label='99%')
    axes[2].set_title('Intrinsic Dimensionality')
    axes[2].legend()
    
    plt.savefig('latent_geometry_analysis.png', dpi=150)
    plt.show()
```

---

## Part 3: Latent Space Arithmetic

### Principle

Latent spaces often support meaningful arithmetic operations:

- **Vector arithmetic:** $z_{\text{result}} = z_A + (z_B - z_C)$
- **Interpolation:** $z_t = (1-t) \cdot z_1 + t \cdot z_2$
- **Averaging:** $z_{\text{avg}} = \frac{1}{n}\sum_{i=1}^{n} z_i$

```python
def demonstrate_latent_arithmetic(model, test_loader, device):
    """
    Demonstrate arithmetic operations in latent space.
    
    For MNIST: z_0 + (z_8 - z_1) might transform 0 to have 8-like features
    """
    print("LATENT SPACE ARITHMETIC")
    
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
            
            if all(len(digit_examples[i]) > 0 for i in range(10)):
                break
    
    # Get average latent representation for each digit
    digit_latents = {}
    for i in range(10):
        imgs = torch.cat(digit_examples[i], dim=0)[:5]
        imgs_flat = imgs.view(imgs.size(0), -1).to(device)
        
        if hasattr(model, 'encode'):
            latents = model.encode(imgs_flat)
        else:
            _, latents = model(imgs_flat)
        
        digit_latents[i] = latents.mean(dim=0, keepdim=True)
    
    # 1. Vector arithmetic: z_0 + (z_8 - z_1)
    print("\n1. Digit Arithmetic: z_0 + (z_8 - z_1)")
    z_0 = digit_latents[0]
    z_1 = digit_latents[1]
    z_8 = digit_latents[8]
    
    z_result = z_0 + (z_8 - z_1)
    
    # Decode results
    decoder = model.decode if hasattr(model, 'decode') else model.decoder
    img_0 = decoder(z_0).cpu().numpy().reshape(28, 28)
    img_result = decoder(z_result).cpu().numpy().reshape(28, 28)
    
    # Visualize
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    
    axes[0].imshow(img_0, cmap='gray')
    axes[0].set_title('z_0 (digit 0)')
    axes[0].axis('off')
    
    axes[1].text(0.5, 0.5, '+', fontsize=20, ha='center', va='center',
                 transform=axes[1].transAxes)
    axes[1].axis('off')
    
    axes[2].imshow(decoder(z_8).cpu().numpy().reshape(28, 28), cmap='gray')
    axes[2].set_title('z_8 - z_1')
    axes[2].axis('off')
    
    axes[3].text(0.5, 0.5, '=', fontsize=20, ha='center', va='center',
                 transform=axes[3].transAxes)
    axes[3].axis('off')
    
    axes[4].imshow(img_result, cmap='gray')
    axes[4].set_title('Result')
    axes[4].axis('off')
    
    plt.suptitle('Latent Space Vector Arithmetic')
    plt.savefig('latent_arithmetic.png', dpi=150)
    plt.show()
    
    # 2. Smooth interpolation
    print("\n2. Interpolation: 0 → 8")
    num_steps = 10
    interpolations = []
    
    for t in np.linspace(0, 1, num_steps):
        z_interp = (1 - t) * digit_latents[0] + t * digit_latents[8]
        img_interp = decoder(z_interp).cpu().numpy().reshape(28, 28)
        interpolations.append(img_interp)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 1.5))
    for i, img in enumerate(interpolations):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Smooth Interpolation in Latent Space')
    plt.savefig('latent_interpolation_advanced.png', dpi=150)
    plt.show()
```

---

## Part 4: Failure Mode Analysis

### Understanding When Autoencoders Fail

Identifies:
- **Best reconstructions** — easiest samples
- **Worst reconstructions** — hardest samples
- **Error patterns by class** — which digits are harder?

```python
def analyze_failure_modes(model, test_loader, device, num_best=10, num_worst=10):
    """Analyze when and why the autoencoder fails."""
    print("FAILURE MODE ANALYSIS")
    
    model.eval()
    
    all_images = []
    all_reconstructions = []
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_flat = images.view(images.size(0), -1).to(device)
            reconstructed, _ = model(images_flat)
            
            errors = torch.mean((images_flat - reconstructed) ** 2, dim=1)
            
            all_images.append(images.numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.numpy())
    
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
    
    # Error by digit class
    print("\nError by digit class:")
    for digit in range(10):
        digit_mask = all_labels == digit
        digit_errors = all_errors[digit_mask]
        print(f"  Digit {digit}: {np.mean(digit_errors):.6f} ± {np.std(digit_errors):.6f}")
    
    # Visualize best and worst
    fig, axes = plt.subplots(4, num_best, figsize=(15, 6))
    
    for i in range(num_best):
        # Best - Original
        idx = best_indices[i]
        axes[0, i].imshow(all_images[idx, 0], cmap='gray')
        axes[0, i].axis('off')
        
        # Best - Reconstructed
        axes[1, i].imshow(all_reconstructions[idx].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        
        # Worst - Original
        idx = worst_indices[i]
        axes[2, i].imshow(all_images[idx, 0], cmap='gray')
        axes[2, i].axis('off')
        
        # Worst - Reconstructed
        axes[3, i].imshow(all_reconstructions[idx].reshape(28, 28), cmap='gray')
        axes[3, i].axis('off')
    
    plt.suptitle('Best vs Worst Reconstructions')
    plt.savefig('failure_analysis.png', dpi=150)
    plt.show()
```

---

## Part 5: Main Execution

```python
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train model
    class AnalysisAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 32
            self.encoder = nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(), nn.BatchNorm1d(256),
                nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Linear(128, self.latent_dim), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256),
                nn.Linear(256, 784), nn.Sigmoid()
            )
        
        def encode(self, x):
            return self.encoder(x)
        
        def decode(self, z):
            return self.decoder(z)
        
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z
    
    model = AnalysisAE().to(device)
    train_loader, test_loader = load_mnist_data()
    
    # Train...
    
    # Run analyses
    analyze_latent_space_geometry(model, test_loader, device)
    demonstrate_latent_arithmetic(model, test_loader, device)
    analyze_failure_modes(model, test_loader, device)

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Cross-Architecture Analysis

Train and compare using ModelComparator:
- Shallow AE (2 layers)
- Deep AE (4-6 layers)
- Convolutional AE
- Sparse AE
- Denoising AE

**Questions:**
- Which is most parameter-efficient?
- Which has smoothest latent space?

### Exercise 2: Latent Space Smoothness

Quantify smoothness:

a) Sample points along interpolation paths  
b) Measure reconstruction error along paths  
c) Compare variance of errors (smooth = low variance)

### Exercise 3: Semantic Vector Discovery

Find meaningful directions in latent space:

a) For each digit pair (i,j), compute: $d_{ij} = \text{mean}(z_i) - \text{mean}(z_j)$  
b) Apply these directions to other digits  
c) Visualize results

**Questions:**
- Can you find "thickness", "slant", "size" directions?

### Exercise 4: Robustness Analysis

Test autoencoder robustness to:
- Input noise
- Occlusion
- Rotation
- Scaling

Compare standard vs denoising autoencoder.

### Exercise 5: Theoretical Capacity

Analyze relationship between:
- Latent dimension
- Reconstruction error
- Number of training samples

```python
latent_dims = [2, 4, 8, 16, 32, 64, 128, 256]
training_samples = [100, 500, 1000, 5000, 10000, 60000]
```

Plot heatmap of reconstruction error.

### Final Project: Complete Autoencoder Study

1. Train 5+ different architectures
2. Run all analyses from this module
3. Create detailed comparison report
4. Identify best architecture for each use case:
   - Compression
   - Anomaly detection
   - Feature extraction
   - Generation (interpolation quality)

---

## Summary

| Analysis | Purpose | Key Output |
|----------|---------|------------|
| **Architecture Comparison** | Compare models systematically | MSE, params, speed plots |
| **Latent Geometry** | Understand latent space structure | Distance correlation, intrinsic dim |
| **Latent Arithmetic** | Test semantic properties | Interpolation, vector math |
| **Failure Analysis** | Identify weaknesses | Best/worst samples, per-class errors |

**Key Insight:** Comprehensive analysis reveals not just how well an autoencoder performs, but *why* it succeeds or fails, enabling informed architecture and hyperparameter choices.
