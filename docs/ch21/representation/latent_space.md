# Latent Space

Understanding the geometry, structure, and information-theoretic properties of learned latent representations.

---

## Overview

**What you'll learn:**

- Latent space definition and properties (compactness, smoothness, disentanglement)
- Reconstruction loss functions and their effect on latent space geometry
- Information bottleneck principle and optimal latent dimensionality
- Latent space geometry analysis: distance preservation, neighborhood preservation, intrinsic dimensionality
- Visualization techniques for high-dimensional latent spaces
- Failure mode analysis: identifying what autoencoders struggle to encode

---

## Mathematical Foundation

### Definition

The **latent space** $\mathcal{Z}$ is the lower-dimensional representation learned by the encoder:

$$z = f_\theta(x) \in \mathbb{R}^k$$

where $k < d$ (latent dimension < input dimension).

### Properties of Good Latent Spaces

| Property | Description | Benefit |
|----------|-------------|---------|
| **Compactness** | Similar inputs map to nearby points | Generalization |
| **Smoothness** | Small changes in $z$ → small changes in $\hat{x}$ | Interpolation |
| **Disentanglement** | Different factors map to different dimensions | Interpretability |

### Information Bottleneck

The latent space acts as an **information bottleneck**: the encoder must discard irrelevant information while preserving what is needed for reconstruction. Smaller $k$ forces more compression and higher reconstruction error, creating a fundamental trade-off:

$$\text{Compression} \uparrow \iff \text{Fidelity} \downarrow$$

---

## Reconstruction Loss and Latent Geometry

### How Loss Choice Shapes the Latent Space

Different reconstruction losses impose different implicit assumptions about the data distribution, which in turn shapes the geometry of the learned latent space:

| Loss | Mathematical Form | Assumption | Effect on Latent Space |
|------|-------------------|------------|----------------------|
| **MSE** | $\frac{1}{n}\sum\|x - \hat{x}\|^2$ | Gaussian noise | Tends to produce blurry reconstructions; latent codes spread smoothly |
| **BCE** | $-\sum[x\log\hat{x} + (1-x)\log(1-\hat{x})]$ | Bernoulli data | Sharper reconstructions; requires sigmoid output |
| **MAE** | $\frac{1}{n}\sum|x - \hat{x}|$ | Laplacian noise | Robust to outliers; sparser gradients; sharper edges |

### Comparing Loss Functions

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class AutoencoderWithAnalysis(nn.Module):
    """Autoencoder with methods for latent space analysis."""
    
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def compare_loss_functions(model_class, train_loader, test_loader, device):
    """Compare different reconstruction loss functions."""
    
    losses = {
        'MSE': nn.MSELoss(),
        'BCE': nn.BCELoss(),
        'L1': nn.L1Loss()
    }
    
    results = {}
    
    for loss_name, criterion in losses.items():
        print(f"\nTraining with {loss_name} loss...")
        
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        for epoch in range(10):
            model.train()
            epoch_loss = 0
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                
                optimizer.zero_grad()
                recon, _ = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
        
        model.eval()
        with torch.no_grad():
            test_images, _ = next(iter(test_loader))
            test_images = test_images[:10].view(10, -1).to(device)
            recon, _ = model(test_images)
        
        results[loss_name] = {
            'model': model,
            'train_losses': train_losses,
            'reconstructions': recon.cpu().numpy()
        }
    
    return results
```

---

## Latent Space Visualization

```python
def visualize_latent_space(model, test_loader, device):
    """
    Visualize the latent space using t-SNE for high-dimensional
    latent codes, or direct plotting for 2D latent spaces.
    """
    model.eval()
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.view(images.size(0), -1).to(device)
            z = model.encode(images)
            latents.append(z.cpu().numpy())
            labels.append(lbls.numpy())
    
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    
    if model.latent_dim == 2:
        # Direct 2D visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents[:, 0], latents[:, 1], 
                             c=labels, cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter, label='Digit')
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.title('2D Latent Space')
    else:
        # Use t-SNE for higher dimensions
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latents_2d = tsne.fit_transform(latents[:5000])
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                             c=labels[:5000], cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter, label='Digit')
        plt.title(f't-SNE of {model.latent_dim}D Latent Space')
    
    plt.savefig('latent_space_visualization.png', dpi=150)
    plt.show()
```

---

## Optimal Latent Dimensionality

```python
def find_optimal_latent_dim(train_loader, test_loader, device, 
                            dims=[2, 4, 8, 16, 32, 64, 128, 256]):
    """
    Find optimal latent dimension by sweeping across dimensions
    and measuring reconstruction error.
    
    The "elbow" in the error curve indicates where additional
    dimensions provide diminishing returns.
    """
    results = []
    
    for dim in dims:
        model = AutoencoderWithAnalysis(latent_dim=dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(15):
            model.train()
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                optimizer.zero_grad()
                recon, _ = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.view(images.size(0), -1).to(device)
                recon, _ = model(images)
                test_loss += criterion(recon, images).item()
        
        test_loss /= len(test_loader)
        results.append({'dim': dim, 'error': test_loss})
        print(f"Latent dim {dim}: Test MSE = {test_loss:.6f}")
    
    return results
```

---

## Latent Space Geometry Analysis

### Key Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Distance correlation** | How well latent distances preserve original distances | Close to 1.0 |
| **Neighborhood preservation** | Fraction of k-nearest neighbors preserved in latent space | Close to 1.0 |
| **Intrinsic dimensionality** | Effective dimensions used in latent space | < latent_dim |

```python
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def analyze_latent_space_geometry(model, test_loader, device, num_samples=2000):
    """
    Analyze geometric properties of learned latent space:
    distance preservation, neighborhood preservation, and
    intrinsic dimensionality.
    """
    model.eval()
    
    original_data = []
    latent_data = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            if len(original_data) * test_loader.batch_size >= num_samples:
                break
            
            images_flat = images.view(images.size(0), -1)
            original_data.append(images_flat.numpy())
            
            images_device = images_flat.to(device)
            latent = model.encode(images_device)
            latent_flat = latent.view(latent.size(0), -1)
            latent_data.append(latent_flat.cpu().numpy())
    
    X_original = np.concatenate(original_data, axis=0)[:num_samples]
    X_latent = np.concatenate(latent_data, axis=0)[:num_samples]
    
    # 1. Distance preservation
    sample_idx = np.random.choice(num_samples, min(500, num_samples), replace=False)
    dist_original = cdist(X_original[sample_idx], X_original[sample_idx])
    dist_latent = cdist(X_latent[sample_idx], X_latent[sample_idx])
    
    dist_orig_flat = dist_original[np.triu_indices_from(dist_original, k=1)]
    dist_lat_flat = dist_latent[np.triu_indices_from(dist_latent, k=1)]
    
    correlation = np.corrcoef(dist_orig_flat, dist_lat_flat)[0, 1]
    print(f"Distance correlation: {correlation:.4f} (1.0 = perfect)")
    
    # 2. Neighborhood preservation
    k = 10
    preservation_scores = []
    
    for i in range(min(100, len(sample_idx))):
        neighbors_orig = set(np.argsort(dist_original[i])[:k+1])
        neighbors_lat = set(np.argsort(dist_latent[i])[:k+1])
        overlap = len(neighbors_orig & neighbors_lat) / (k + 1)
        preservation_scores.append(overlap)
    
    print(f"Neighborhood preservation: {np.mean(preservation_scores):.4f}")
    
    # 3. Intrinsic dimensionality via PCA on latent space
    pca = PCA()
    pca.fit(X_latent)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    dim_95 = np.argmax(cumsum_var >= 0.95) + 1
    dim_99 = np.argmax(cumsum_var >= 0.99) + 1
    
    print(f"Intrinsic dim (95% var): {dim_95}/{X_latent.shape[1]}")
    print(f"Intrinsic dim (99% var): {dim_99}/{X_latent.shape[1]}")
    
    return {
        'distance_correlation': correlation,
        'neighborhood_preservation': np.mean(preservation_scores),
        'intrinsic_dim_95': dim_95,
        'intrinsic_dim_99': dim_99
    }
```

---

## Reconstruction Quality Analysis

```python
def analyze_reconstruction_quality(model, test_loader, device):
    """
    Analyze reconstruction quality across different inputs.
    Identifies which classes are easiest/hardest to reconstruct.
    """
    model.eval()
    
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            recon, _ = model(images)
            
            errors = torch.mean((images - recon) ** 2, dim=1)
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    # Error by class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(all_errors), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(all_errors):.4f}')
    axes[0].set_xlabel('Reconstruction Error (MSE)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Reconstruction Errors')
    axes[0].legend()
    
    error_by_digit = [all_errors[all_labels == d] for d in range(10)]
    axes[1].boxplot(error_by_digit, labels=range(10))
    axes[1].set_xlabel('Digit')
    axes[1].set_ylabel('Reconstruction Error')
    axes[1].set_title('Reconstruction Error by Digit Class')
    
    plt.tight_layout()
    plt.savefig('reconstruction_analysis.png', dpi=150)
    plt.show()
    
    return all_errors, all_labels
```

---

## Failure Mode Analysis

```python
def analyze_failure_modes(model, test_loader, device, num_best=10, num_worst=10):
    """
    Identify best and worst reconstructions to understand
    what the autoencoder struggles with.
    """
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
    
    best_indices = np.argsort(all_errors)[:num_best]
    worst_indices = np.argsort(all_errors)[-num_worst:][::-1]
    
    print(f"Best reconstruction error: {all_errors[best_indices[0]]:.6f}")
    print(f"Worst reconstruction error: {all_errors[worst_indices[0]]:.6f}")
    print(f"Mean reconstruction error: {np.mean(all_errors):.6f}")
    
    print("\nError by digit class:")
    for digit in range(10):
        digit_errors = all_errors[all_labels == digit]
        print(f"  Digit {digit}: {np.mean(digit_errors):.6f} "
              f"± {np.std(digit_errors):.6f}")
    
    # Visualize best and worst
    fig, axes = plt.subplots(4, num_best, figsize=(15, 6))
    
    for i in range(num_best):
        idx = best_indices[i]
        axes[0, i].imshow(all_images[idx, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(all_reconstructions[idx].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        
        idx = worst_indices[i]
        axes[2, i].imshow(all_images[idx, 0], cmap='gray')
        axes[2, i].axis('off')
        axes[3, i].imshow(all_reconstructions[idx].reshape(28, 28), cmap='gray')
        axes[3, i].axis('off')
    
    axes[0, 0].set_ylabel('Best\nOriginal')
    axes[1, 0].set_ylabel('Best\nRecon')
    axes[2, 0].set_ylabel('Worst\nOriginal')
    axes[3, 0].set_ylabel('Worst\nRecon')
    
    plt.suptitle('Best vs Worst Reconstructions')
    plt.tight_layout()
    plt.savefig('failure_analysis.png', dpi=150)
    plt.show()
```

---

## Quantitative Finance Application

In quantitative finance, latent space analysis is directly relevant to factor model evaluation:

- **Factor dimension selection:** The optimal latent dimension corresponds to the number of statistical factors explaining asset returns — analogous to selecting the number of PCA components
- **Distance preservation:** Good latent spaces preserve the correlation structure of assets, meaning similar assets remain nearby in latent space
- **Reconstruction error by asset:** Assets with high reconstruction error may be poorly explained by common factors, indicating idiosyncratic risk or potential alpha opportunities

---

## Exercises

### Exercise 1: Loss Function Comparison
Train autoencoders with MSE, BCE, and L1 losses. Compare latent space geometry (distance preservation) and reconstruction quality.

### Exercise 2: Latent Dimension Sweep
Plot reconstruction error vs latent dimension. Find the "elbow" point where additional dimensions provide diminishing returns.

### Exercise 3: Latent Space Arithmetic
Encode MNIST digits, compute mean latent vectors per class, and explore arithmetic: $z_{\text{new}} = z_3 + (z_8 - z_0)$. Does the result look like a valid digit?

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Latent space** | Lower-dimensional learned representation capturing essential data structure |
| **MSE loss** | Gaussian noise assumption, tends toward blurry outputs |
| **BCE loss** | Natural for binary data, sharper outputs |
| **Information bottleneck** | Fundamental trade-off between compression and reconstruction fidelity |
| **Distance preservation** | Good latent spaces maintain relative distances from input space |
| **Intrinsic dimensionality** | Often much smaller than the nominal latent dimension |
