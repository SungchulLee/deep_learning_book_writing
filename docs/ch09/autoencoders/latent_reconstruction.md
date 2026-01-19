# Latent Space and Reconstruction Loss

Understanding the latent representation and how reconstruction loss shapes learning.

---

## Overview

**Key Concepts:**

- Latent space geometry and structure
- Reconstruction loss functions (MSE, BCE, MAE)
- Information bottleneck principle
- Visualization techniques

**Time:** ~35 minutes  
**Level:** Beginner-Intermediate

---

## The Latent Space

### Definition

The **latent space** $Z$ is the lower-dimensional representation learned by the encoder:

$$z = f_\theta(x) \in \mathbb{R}^k$$

where $k < d$ (latent dimension < input dimension).

### Properties of Good Latent Spaces

| Property | Description | Benefit |
|----------|-------------|---------|
| **Compactness** | Similar inputs map to nearby points | Generalization |
| **Smoothness** | Small changes in $z$ → small changes in $\hat{x}$ | Interpolation |
| **Disentanglement** | Different factors map to different dimensions | Interpretability |

---

## Reconstruction Loss Functions

### Mean Squared Error (MSE)

$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \hat{x}_i\|^2$$

**Properties:**
- Penalizes large errors more than small errors
- Assumes Gaussian noise model
- Tends to produce blurry reconstructions

### Binary Cross-Entropy (BCE)

$$\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [x_i \log(\hat{x}_i) + (1-x_i) \log(1-\hat{x}_i)]$$

**Properties:**
- Natural for binary/normalized data
- Requires sigmoid output
- Often gives sharper reconstructions

### Mean Absolute Error (MAE / L1)

$$\mathcal{L}_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \hat{x}_i|$$

**Properties:**
- More robust to outliers
- Produces sparser gradients
- Can lead to sharper edges

---

## PyTorch Implementation

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
        
        # Evaluate
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


def visualize_latent_space(model, test_loader, device, method='direct'):
    """Visualize the latent space with different techniques."""
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


def analyze_reconstruction_quality(model, test_loader, device):
    """Analyze reconstruction quality across different inputs."""
    model.eval()
    
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            recon, _ = model(images)
            
            # Per-sample MSE
            errors = torch.mean((images - recon) ** 2, dim=1)
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    # Error by class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(all_errors), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(all_errors):.4f}')
    axes[0].set_xlabel('Reconstruction Error (MSE)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Reconstruction Errors')
    axes[0].legend()
    
    # Error by digit
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

## Information Bottleneck

### The Principle

The latent space acts as an **information bottleneck**:

- **Compression:** Encoder must discard irrelevant information
- **Preservation:** Must keep information needed for reconstruction
- **Trade-off:** Smaller $k$ → more compression, higher error

### Optimal Latent Dimension

```python
def find_optimal_latent_dim(train_loader, test_loader, device, 
                            dims=[2, 4, 8, 16, 32, 64, 128, 256]):
    """Find optimal latent dimension by reconstruction error."""
    
    results = []
    
    for dim in dims:
        model = AutoencoderWithAnalysis(latent_dim=dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train
        for epoch in range(15):
            model.train()
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                optimizer.zero_grad()
                recon, _ = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
        
        # Evaluate
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

## Exercises

### Exercise 1: Loss Function Comparison
Train autoencoders with MSE, BCE, and L1 losses. Compare reconstruction quality visually and numerically.

### Exercise 2: Latent Dimension Analysis
Plot reconstruction error vs latent dimension. Find the "elbow" point.

### Exercise 3: Latent Space Arithmetic
Encode digits, compute mean latent vectors per class, and explore: $z_{new} = z_3 + (z_8 - z_0)$

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Latent space** | Lower-dimensional learned representation |
| **MSE loss** | Gaussian assumption, blurry outputs |
| **BCE loss** | Binary data, sharper outputs |
| **Bottleneck** | Trade-off between compression and quality |

---

## Next: Undercomplete vs Overcomplete

The next section explores different autoencoder configurations based on latent dimension.
