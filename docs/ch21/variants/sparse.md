# Sparse Autoencoder

Learn sparse representations by penalizing hidden unit activations, leading to more interpretable and robust features.

---

## Overview

**What you'll learn:**

- L1 regularization on latent activations for sparsity
- KL divergence sparsity constraint with target activation rates
- Population sparsity vs lifetime sparsity analysis
- Visualization and interpretation of learned features
- Connection to sparse coding and dictionary learning

---

## Mathematical Foundation

### Standard vs Sparse Autoencoder Loss

| Type | Loss Function |
|------|---------------|
| Standard | $\mathcal{L} = \|x - f(x)\|^2$ |
| Sparse (L1) | $\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$ |
| Sparse (KL) | $\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$ |

### L1 Regularization

$$\mathcal{L} = \|x - f(x)\|^2 + \lambda \sum_j |h_j|$$

where $h_j$ is the activation of neuron $j$ in the latent layer, $\lambda$ controls sparsity strength, and $\sum_j |h_j|$ encourages many activations to be exactly zero.

### KL Divergence Sparsity

$$\mathcal{L} = \|x - f(x)\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$$

where:

- $\rho$ is the target sparsity level (e.g., 0.05 means each neuron active ~5% of the time)
- $\hat{\rho}_j = \frac{1}{n}\sum_{i=1}^{n} h_j(x_i)$ is the average activation of neuron $j$ over the dataset
- $\text{KL}(\rho \| \hat{\rho}_j) = \rho \log\frac{\rho}{\hat{\rho}_j} + (1-\rho) \log\frac{1-\rho}{1-\hat{\rho}_j}$
- $\beta$ is the sparsity weight

The KL penalty is minimized when $\hat{\rho}_j = \rho$, providing precise control over the average activation rate.

### Why Sparsity?

1. **Selective feature activation** — each input activates only a relevant subset of features
2. **Interpretable representations** — individual features correspond to meaningful patterns
3. **Robustness to noise** — sparse codes are more stable under perturbation
4. **Better generalization** — reduced effective model capacity prevents overfitting

---

## Part 1: L1 Sparse Autoencoder

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
    
    Loss = Reconstruction Loss + λ × L1(latent activations)
    
    The L1 penalty encourages many latent activations to be exactly zero,
    producing sparse codes where only a few neurons fire per input.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_L1, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Often uses larger latent_dim for overcomplete representations
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
    
    L1(h) = (1/batch_size) Σᵢⱼ |hᵢⱼ|
    
    Encourages sparsity by penalizing non-zero activations.
    """
    return torch.mean(torch.abs(latent))
```

---

## Part 2: KL Divergence Sparse Autoencoder

```python
class SparseAutoencoder_KL(nn.Module):
    """
    Sparse Autoencoder using KL divergence sparsity constraint.
    
    Constrains the average activation of each neuron to be close
    to a target sparsity level ρ (e.g., 0.05).
    
    Key difference from L1: KL provides precise control over the
    target activation rate, while L1 simply penalizes magnitude.
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
            nn.Sigmoid()  # Required: KL divergence assumes Bernoulli activations
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
    
    This is minimized when ρ̂ⱼ = ρ exactly.
    """
    # Average activation for each neuron across batch
    rho_hat = torch.mean(latent, dim=0)
    
    # Clamp to avoid log(0)
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    
    # KL divergence for each neuron
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    return torch.sum(kl)
```

---

## Part 3: Training

```python
def train_sparse_autoencoder(
    model, train_loader, optimizer, device, epoch,
    sparsity_type='l1', sparsity_weight=0.001, rho=0.05
):
    """
    Train sparse autoencoder for one epoch.
    
    Total Loss = Reconstruction Loss + sparsity_weight × Sparsity Penalty
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
    Analyze sparsity of learned representations using both
    population and lifetime sparsity metrics.
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


def visualize_sparsity_analysis(model, test_loader, device):
    """
    Visualize sparsity statistics:
    1. Histogram of population sparsity across samples
    2. Histogram of lifetime sparsity across neurons
    3. Overall activation distribution
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
    
    axes[0].hist(population_sparsity_per_sample, bins=50)
    axes[0].set_title('Population Sparsity (per sample)')
    axes[0].set_xlabel('Fraction of Active Neurons')
    
    axes[1].hist(lifetime_sparsity_per_neuron, bins=50)
    axes[1].set_title('Lifetime Sparsity (per neuron)')
    axes[1].set_xlabel('Fraction of Activating Samples')
    
    axes[2].hist(all_activations.flatten(), bins=100)
    axes[2].set_yscale('log')
    axes[2].set_title('Activation Distribution')
    axes[2].set_xlabel('Activation Value')
    
    plt.tight_layout()
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
    than dense autoencoders, showing localized patterns rather
    than distributed holistic representations.
    """
    model.eval()
    
    latent_dim = model.latent_dim
    num_features = min(num_features, latent_dim)
    
    features = []
    with torch.no_grad():
        for i in range(num_features):
            # Create one-hot vector: activate only neuron i
            latent = torch.zeros(1, latent_dim)
            latent[0, i] = 1.0
            
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
    
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Learned Features (Decoder Basis)')
    plt.tight_layout()
    plt.savefig('learned_features.png', dpi=150)
    plt.show()
```

---

## Part 6: Complete Training Pipeline

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
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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

if __name__ == "__main__":
    main()
```

---

## Quantitative Finance Application

Sparse autoencoders are particularly valuable in quantitative finance for learning **interpretable factor representations**:

- **Sparse factor models:** Each asset's return is explained by a small subset of latent factors, analogous to sparse factor models where factor loadings are mostly zero
- **Regime-specific features:** Sparsity ensures that different market regimes activate distinct latent factors, enabling regime-conditional analysis
- **Feature selection:** The decoder weights of active neurons reveal which input features drive each latent factor, providing economic interpretability

---

## Exercises

### Exercise 1: Sparsity Weight Tuning

Train models with different sparsity weights (`λ ∈ {0.0001, 0.001, 0.01, 0.1, 1.0}` for L1, `β ∈ {0.001, 0.01, 0.1, 1.0, 10.0}` for KL). How does sparsity weight affect reconstruction quality? What is the optimal trade-off?

### Exercise 2: L1 vs KL Comparison

Train two models with similar effective sparsity (L1 with λ = 0.01, KL with β = 0.1 and ρ = 0.05). Compare training dynamics, final sparsity levels, and learned feature quality.

### Exercise 3: Overcomplete Sparse Representations

Train with different latent dimensions (`[64, 128, 256, 512, 1024]`). How does overcomplete representation affect feature quality with sparsity constraints?

### Exercise 4: Feature Selectivity

For each digit class (0–9), compute the average activation pattern. Identify features that are highly selective for specific digits. Visualize the most and least selective features.

### Exercise 5: Reconstruction from Sparse Codes

Take a test image, progressively zero out latent neurons starting from the smallest activations, and reconstruct. Plot sparsity level vs reconstruction quality to find the minimum number of neurons needed.

---

## Summary

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **L1** | Penalize $\|h\|_1$ | Simple, fast, no activation constraint | May not reach exact target sparsity |
| **KL** | Penalize divergence from target $\rho$ | Precise control over sparsity level | Requires sigmoid activation, slower |

**Key Insight:** Sparse autoencoders learn overcomplete bases where only a small subset of neurons are active for any given input, leading to more interpretable and robust features compared to dense autoencoders. This mirrors the sparse coding hypothesis in neuroscience and connects to dictionary learning in signal processing.
