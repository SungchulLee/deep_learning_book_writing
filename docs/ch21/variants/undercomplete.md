# Undercomplete and Overcomplete Autoencoders

Understanding how the relationship between latent dimension and input dimension fundamentally shapes autoencoder learning dynamics.

---

## Overview

**What you'll learn:**

- Undercomplete autoencoders ($\text{dim}(z) < \text{dim}(x)$) and their natural regularization
- Overcomplete autoencoders ($\text{dim}(z) \geq \text{dim}(x)$) and the identity mapping problem
- Regularization strategies for overcomplete settings
- Practical guidelines for choosing latent dimensionality

---

## Mathematical Foundation

### Undercomplete Autoencoder

$$\text{dim}(z) < \text{dim}(x)$$

The standard configuration. The bottleneck forces compression and prevents the network from learning the trivial identity mapping. This provides **natural regularization** through the information bottleneck:

- **Forces compression:** Must learn efficient encoding that retains only the most informative features
- **Prevents identity mapping:** Cannot simply copy input through the bottleneck
- **Natural regularization:** The capacity constraint itself constrains the representation

### Overcomplete Autoencoder

$$\text{dim}(z) \geq \text{dim}(x)$$

With more latent dimensions than input dimensions, the network can potentially learn the identity function $f(x) = x$, $g(z) = z$, achieving zero reconstruction error without learning anything useful. Overcomplete autoencoders **require explicit regularization** — sparsity constraints, denoising, or contractive penalties — to learn meaningful representations.

---

## The Identity Mapping Problem

### Why It's a Problem

With overcomplete representation, the autoencoder can learn:

$$f(x) = x, \quad g(z) = z$$

This achieves **zero reconstruction error** but learns **nothing useful** about the data's structure.

### When It Occurs

| Condition | Identity Risk | Mitigation |
|-----------|---------------|------------|
| `latent_dim >= input_dim` | High | Sparsity, denoising, contractive penalty |
| Linear activations | Very high | Use nonlinear activations |
| No regularization | High | Add explicit regularization |
| Excessive capacity | Medium | Reduce hidden layer widths |

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

class FlexibleAutoencoder(nn.Module):
    """
    Autoencoder that can be undercomplete or overcomplete.
    
    The compression_ratio property indicates whether the 
    bottleneck is narrower (undercomplete) or wider (overcomplete)
    than the input.
    """
    
    def __init__(self, input_dim=784, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    
    @property
    def compression_ratio(self):
        return self.input_dim / self.latent_dim
    
    @property
    def is_undercomplete(self):
        return self.latent_dim < self.input_dim


def compare_completeness(train_loader, test_loader, device):
    """
    Compare undercomplete and overcomplete autoencoders
    across a range of latent dimensions.
    """
    input_dim = 784
    
    configs = {
        'Very Undercomplete (k=16)': 16,
        'Undercomplete (k=64)': 64,
        'Undercomplete (k=256)': 256,
        'Equal (k=784)': 784,
        'Overcomplete (k=1024)': 1024,
    }
    
    results = {}
    
    for name, latent_dim in configs.items():
        print(f"\nTraining {name}...")
        
        model = FlexibleAutoencoder(input_dim, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training
        train_losses = []
        for epoch in range(15):
            model.train()
            epoch_loss = 0
            for images, _ in train_loader:
                images = images.view(images.size(0), -1).to(device)
                
                optimizer.zero_grad()
                recon, z = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluation
        model.eval()
        test_loss = 0
        latent_norms = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.view(images.size(0), -1).to(device)
                recon, z = model(images)
                test_loss += criterion(recon, images).item()
                latent_norms.extend(torch.norm(z, dim=1).cpu().numpy())
        
        results[name] = {
            'latent_dim': latent_dim,
            'train_losses': train_losses,
            'test_loss': test_loss / len(test_loader),
            'mean_latent_norm': np.mean(latent_norms),
            'model': model
        }
        
        print(f"  Final test loss: {results[name]['test_loss']:.6f}")
        print(f"  Mean latent norm: {results[name]['mean_latent_norm']:.4f}")
    
    return results


def analyze_identity_mapping(model, test_loader, device):
    """
    Check if model is learning identity mapping by examining
    reconstruction error and input-latent correlation.
    """
    model.eval()
    
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images[:100].view(100, -1).to(device)
        recon, z = model(images)
        
        # Check 1: Reconstruction error
        recon_error = torch.mean((images - recon) ** 2).item()
        
        # Check 2: Are latent representations just scaled inputs?
        if z.shape[1] >= images.shape[1]:
            correlation = torch.corrcoef(
                torch.cat([images.flatten().unsqueeze(0), 
                          z[:, :images.shape[1]].flatten().unsqueeze(0)])
            )[0, 1].item()
        else:
            correlation = 0.0
        
        # Check 3: Latent space variance
        latent_var = torch.var(z, dim=0).mean().item()
        
    return {
        'recon_error': recon_error,
        'input_latent_correlation': correlation,
        'latent_variance': latent_var
    }
```

---

## Regularization for Overcomplete Autoencoders

### Why Regularize?

Overcomplete autoencoders need regularization to prevent identity mapping, learn meaningful features, and improve generalization. Without explicit constraints, the extra capacity is wasted on trivially copying inputs.

### Common Strategies

| Strategy | Description | Effect |
|----------|-------------|--------|
| **Sparsity** | Penalize non-zero activations | Forces selective encoding |
| **Denoising** | Reconstruct from corrupted input | Prevents copying |
| **Contractive** | Penalize encoder Jacobian | Robust representations |
| **Dropout** | Random neuron dropping | Prevents co-adaptation |

### Sparse Overcomplete Autoencoder

```python
class SparseOvercompleteAutoencoder(nn.Module):
    """
    Overcomplete autoencoder with L1 sparsity regularization.
    
    Even with latent_dim > input_dim, the sparsity penalty
    ensures only a small subset of neurons activate per input.
    """
    
    def __init__(self, input_dim=784, latent_dim=1024):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()  # ReLU naturally promotes sparsity
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_with_sparsity(model, train_loader, device, sparsity_weight=0.001):
    """Train overcomplete autoencoder with sparsity penalty."""
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    recon_criterion = nn.MSELoss()
    
    for epoch in range(15):
        model.train()
        total_loss = 0
        
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            
            optimizer.zero_grad()
            recon, z = model(images)
            
            # Reconstruction loss
            recon_loss = recon_criterion(recon, images)
            
            # Sparsity penalty (L1 on latent activations)
            sparsity_loss = torch.mean(torch.abs(z))
            
            # Total loss
            loss = recon_loss + sparsity_weight * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.6f}")
    
    return model
```

---

## Visualization

```python
def visualize_comparison(results):
    """Visualize comparison between different configurations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    latent_dims = [results[n]['latent_dim'] for n in names]
    test_losses = [results[n]['test_loss'] for n in names]
    latent_norms = [results[n]['mean_latent_norm'] for n in names]
    
    # Plot 1: Test loss vs latent dimension
    ax = axes[0, 0]
    ax.semilogx(latent_dims, test_losses, 'bo-', markersize=10, linewidth=2)
    ax.axvline(784, color='r', linestyle='--', label='Input dim (784)')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Test MSE Loss')
    ax.set_title('Reconstruction Error vs Latent Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training curves
    ax = axes[0, 1]
    for name in names:
        ax.plot(results[name]['train_losses'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Latent norm vs dimension
    ax = axes[1, 0]
    ax.semilogx(latent_dims, latent_norms, 'go-', markersize=10, linewidth=2)
    ax.axvline(784, color='r', linestyle='--', label='Input dim')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean Latent Norm')
    ax.set_title('Latent Representation Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('completeness_comparison.png', dpi=150)
    plt.show()
```

---

## Guidelines

### When to Use Undercomplete

- Default choice for most applications
- When compression is an explicit goal
- When computational efficiency matters
- When interpretable low-dimensional latent space is desired

### When to Use Overcomplete

- With proper regularization (sparse, denoising, contractive)
- When learning rich, overcomplete feature representations
- In sparse coding applications where dictionary atoms outnumber input dimensions
- When input is already low-dimensional

---

## Quantitative Finance Application

In quantitative finance, the undercomplete vs overcomplete distinction maps directly to factor model design:

- **Undercomplete:** Extract a small number of latent risk factors from high-dimensional asset returns — analogous to statistical factor models (e.g., PCA-based)
- **Overcomplete with sparsity:** Learn an overcomplete dictionary of market regimes or micro-patterns where only a few factors are active at any time — analogous to sparse factor models

---

## Exercises

### Exercise 1: Identity Mapping Detection
Train an overcomplete linear autoencoder (no activations) and verify it learns the identity mapping. Compare with a nonlinear overcomplete autoencoder.

### Exercise 2: Regularization Comparison
Compare unregularized vs sparse vs denoising overcomplete autoencoders on MNIST. Measure reconstruction quality and sparsity of learned representations.

### Exercise 3: Optimal Dimension
For MNIST, find the smallest latent dimension that achieves < 0.01 MSE. Plot the reconstruction error curve as a function of latent dimension to identify the "elbow" point.

---

## Summary

| Type | Latent Dim | Risk | Mitigation |
|------|------------|------|------------|
| **Undercomplete** | $k < d$ | Underfitting if too small | Increase capacity |
| **Equal** | $k = d$ | Identity mapping | Regularization required |
| **Overcomplete** | $k > d$ | Identity mapping | Sparsity, denoising, contractive penalty |

**Key Insight:** The choice between undercomplete and overcomplete is not merely about dimensionality — it determines whether the bottleneck itself provides regularization or whether explicit regularization must be imposed to prevent trivial solutions.
