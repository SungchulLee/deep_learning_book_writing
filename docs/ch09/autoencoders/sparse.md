# Sparse Autoencoder

Learn sparse representations by penalizing hidden unit activations, leading to more interpretable features.

---

## Overview

**Key Concepts:**

- L1 regularization on activations
- KL divergence sparsity constraint
- Learning interpretable features
- Sparse vs. dense representations
- Relationship to sparse coding

**Time:** ~50 minutes  
**Level:** Intermediate

---

## Mathematical Foundation

### Standard vs. Sparse Autoencoder Loss

| Type | Loss Function |
|------|---------------|
| Standard | $\mathcal{L} = \|x - f(x)\|^2$ |
| Sparse (L1) | $\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$ |
| Sparse (KL) | $\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$ |

### L1 Regularization

$$\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$$

Where:
- $h_j$ is the activation of neuron $j$ in the latent layer
- $\lambda$ is sparsity regularization strength
- $\sum_j |h_j|$ encourages many activations to be zero

### KL Divergence Sparsity

$$\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$$

Where:
- $\rho$ is target sparsity level (e.g., 0.05)
- $\hat{\rho}_j$ is average activation of neuron $j$
- $\text{KL}(\rho \| \hat{\rho}_j) = \rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho) \log\frac{1-\rho}{1-\hat{\rho}_j}$
- $\beta$ is sparsity weight

### Why Sparsity?

1. **Selective feature activation** — each input activates only relevant features
2. **Interpretable representations** — features correspond to meaningful patterns
3. **Robustness to noise** — sparse codes are more stable
4. **Better generalization** — prevents overfitting

---

## Part 1: Sparse Autoencoder with L1 Regularization

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class SparseAutoencoder_L1(nn.Module):
    """
    Sparse Autoencoder using L1 regularization on latent activations.
    
    Loss = Reconstruction Loss + λ * L1(latent activations)
    
    The L1 penalty encourages many latent activations to be exactly zero.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_L1, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Often larger latent_dim for overcomplete representations
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()  # ReLU naturally promotes sparsity
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def l1_loss(latent: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 penalty on latent activations.
    
    L1(h) = Σᵢⱼ |hᵢⱼ|
    
    Encourages sparsity by penalizing non-zero activations.
    """
    return torch.mean(torch.abs(latent))
```

---

## Part 2: Sparse Autoencoder with KL Divergence

```python
class SparseAutoencoder_KL(nn.Module):
    """
    Sparse Autoencoder using KL divergence sparsity constraint.
    
    Constrains the average activation of each neuron to be close
    to a target sparsity level ρ (e.g., 0.05).
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_KL, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder with Sigmoid for KL divergence (outputs in [0,1])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Sigmoid()  # Required for KL divergence
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def kl_divergence_loss(latent: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
    """
    Compute KL divergence sparsity penalty.
    
    For each neuron j, we want its average activation ρ̂ⱼ ≈ ρ.
    
    KL(ρ || ρ̂ⱼ) = ρ log(ρ/ρ̂ⱼ) + (1-ρ) log((1-ρ)/(1-ρ̂ⱼ))
    
    Minimized when ρ̂ⱼ = ρ.
    """
    # Average activation for each neuron across batch
    rho_hat = torch.mean(latent, dim=0)
    
    # Avoid log(0)
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    
    # KL divergence for each neuron
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    return torch.sum(kl)
```

---

## Part 3: Training Function

```python
def train_sparse_autoencoder(
    model, train_loader, optimizer, device, epoch,
    sparsity_type='l1', sparsity_weight=0.001, rho=0.05
):
    """
    Train sparse autoencoder for one epoch.
    
    Total Loss = Reconstruction Loss + Sparsity Penalty
    """
    model.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    num_batches = 0
    
    recon_criterion = nn.MSELoss()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, latent = model(images)
        
        # Reconstruction loss
        recon_loss = recon_criterion(reconstructed, images)
        
        # Sparsity penalty
        if sparsity_type == 'l1':
            sparsity_loss = l1_loss(latent)
        elif sparsity_type == 'kl':
            sparsity_loss = kl_divergence_loss(latent, rho)
        
        # Total loss
        total_loss = recon_loss + sparsity_weight * sparsity_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        sparsity_loss_sum += sparsity_loss.item()
        num_batches += 1
    
    return (total_loss_sum / num_batches, 
            recon_loss_sum / num_batches, 
            sparsity_loss_sum / num_batches)
```

---

## Part 4: Sparsity Analysis

### Sparsity Metrics

| Metric | Definition |
|--------|------------|
| **Population sparsity** | For each sample, what fraction of neurons are active? |
| **Lifetime sparsity** | For each neuron, what fraction of samples activate it? |

```python
def analyze_sparsity(model, test_loader, device, num_samples=1000):
    """
    Analyze sparsity of learned representations.
    """
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            if len(all_activations) * test_loader.batch_size >= num_samples:
                break
            images = images.view(images.size(0), -1).to(device)
            _, latent = model(images)
            all_activations.append(latent.cpu().numpy())
    
    all_activations = np.concatenate(all_activations, axis=0)[:num_samples]
    
    # Define "active" as activation > threshold
    threshold = 0.1
    active = all_activations > threshold
    
    # Population sparsity: average fraction of active neurons per sample
    population_sparsity = np.mean(np.mean(active, axis=1))
    
    # Lifetime sparsity: fraction of samples that activate each neuron
    lifetime_sparsity = np.mean(active, axis=0)
    
    return population_sparsity, lifetime_sparsity
```

### Visualization

```python
def visualize_sparsity_analysis(model, test_loader, device):
    """
    Visualize sparsity statistics with three plots:
    1. Histogram of population sparsity across samples
    2. Histogram of lifetime sparsity across neurons
    3. Activation distribution for latent layer
    """
    model.eval()
    
    all_activations = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            _, latent = model(images)
            all_activations.append(latent.cpu().numpy())
            if len(all_activations) >= 20:
                break
    
    all_activations = np.concatenate(all_activations, axis=0)
    
    threshold = 0.1
    active = all_activations > threshold
    population_sparsity_per_sample = np.mean(active, axis=1)
    lifetime_sparsity_per_neuron = np.mean(active, axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Population sparsity
    axes[0].hist(population_sparsity_per_sample, bins=50)
    axes[0].set_title('Population Sparsity (per sample)')
    
    # Lifetime sparsity
    axes[1].hist(lifetime_sparsity_per_neuron, bins=50)
    axes[1].set_title('Lifetime Sparsity (per neuron)')
    
    # Activation distribution
    axes[2].hist(all_activations.flatten(), bins=100)
    axes[2].set_yscale('log')
    axes[2].set_title('Activation Distribution')
    
    plt.savefig('sparsity_analysis.png', dpi=150)
    plt.show()
```

---

## Part 5: Visualize Learned Features

```python
def visualize_learned_features(model, num_features=64):
    """
    Visualize learned features by decoding one-hot latent vectors.
    
    For sparse autoencoders, features are often more interpretable
    than dense autoencoders, showing localized patterns.
    """
    model.eval()
    
    latent_dim = model.latent_dim
    num_features = min(num_features, latent_dim)
    
    features = []
    with torch.no_grad():
        for i in range(num_features):
            # Create one-hot vector
            latent = torch.zeros(1, latent_dim)
            latent[0, i] = 1.0  # Activate only neuron i
            
            # Decode to image space
            feature = model.decoder(latent)
            features.append(feature.cpu().numpy().reshape(28, 28))
    
    # Visualize in grid
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_features):
        axes[i].imshow(features[i], cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Learned Features (Decoder Basis)')
    plt.savefig('learned_features.png', dpi=150)
    plt.show()
```

---

## Part 6: Main Execution

```python
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    input_dim = 784
    latent_dim = 128  # Often larger for sparse AE (overcomplete)
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 15
    
    # Sparsity configuration
    sparsity_type = 'kl'  # Options: 'l1' or 'kl'
    sparsity_weight = 0.01
    rho = 0.05  # Target sparsity (5% activation)
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model
    if sparsity_type == 'l1':
        model = SparseAutoencoder_L1(input_dim, latent_dim).to(device)
    else:
        model = SparseAutoencoder_KL(input_dim, latent_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        total_loss, recon_loss, sparse_loss = train_sparse_autoencoder(
            model, train_loader, optimizer, device, epoch,
            sparsity_type, sparsity_weight, rho
        )
        print(f"Epoch {epoch} - Total: {total_loss:.6f}, "
              f"Recon: {recon_loss:.6f}, Sparse: {sparse_loss:.6f}")
    
    # Analysis
    pop_sparsity, life_sparsity = analyze_sparsity(model, test_loader, device)
    print(f"\nPopulation sparsity: {pop_sparsity:.4f}")
    print(f"Lifetime sparsity (mean): {np.mean(life_sparsity):.4f}")
    print(f"Inactive neurons (<1%): {np.sum(life_sparsity < 0.01)}/{len(life_sparsity)}")
    
    # Visualizations
    visualize_sparsity_analysis(model, test_loader, device)
    visualize_learned_features(model, num_features=64)
    
    torch.save(model.state_dict(), f'sparse_autoencoder_{sparsity_type}.pth')

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Sparsity Weight Tuning

Train models with different sparsity weights:

```python
l1_weights = [0.0001, 0.001, 0.01, 0.1, 1.0]
kl_weights = [0.001, 0.01, 0.1, 1.0, 10.0]
```

**Questions:**
- How does sparsity weight affect reconstruction quality?
- What is the optimal trade-off between sparsity and reconstruction?
- Do stronger constraints lead to more interpretable features?

### Exercise 2: L1 vs KL Comparison

Train two models with similar effective sparsity:
- L1 model with λ = 0.01
- KL model with β = 0.1, ρ = 0.05

Compare training dynamics, final sparsity levels, and feature quality.

### Exercise 3: Overcomplete Representations

Train with different latent dimensions:

```python
latent_dims = [64, 128, 256, 512, 1024]
```

Note: 784 is input dimension, so >784 is "overcomplete"

**Questions:**
- How does overcomplete representation affect feature quality?
- What happens to sparsity as latent_dim increases?

### Exercise 4: Feature Selectivity

Analyze which features activate for which digits:

1. For each digit class (0-9), compute average activation pattern
2. Identify features that are highly selective for specific digits
3. Visualize most and least selective features

### Exercise 5: Reconstruction from Sparse Codes

1. Take a test image and get its latent representation
2. Gradually set more neurons to zero (starting from smallest)
3. Reconstruct from increasingly sparse codes
4. Plot: sparsity level vs. reconstruction quality

**Questions:**
- How many neurons are actually needed for good reconstruction?
- Which neurons are most important?

### Advanced Challenge: Dictionary Learning Connection

1. Extract decoder weights as dictionary atoms
2. Compare with K-SVD or OMP dictionary learning
3. Use learned features for classification task
4. Compare with PCA features

---

## Summary

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **L1** | Penalize $\|h\|_1$ | Simple, fast | May not reach exact target sparsity |
| **KL** | Penalize divergence from target $\rho$ | Precise control over sparsity | Requires sigmoid activation |

**Key Insight:** Sparse autoencoders learn overcomplete bases where only a small subset of neurons are active for any given input, leading to more interpretable and robust features compared to dense autoencoders.
