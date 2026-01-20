# Section 41.3: VAE Architecture and Implementation

Complete Variational Autoencoder implementation in PyTorch.

---

## Overview

**What you'll learn:**

- Complete VAE architecture design
- Encoder: predicting μ and log(σ²)
- Decoder: generating from latent samples
- Loss function implementation
- Training loop and monitoring

**Time:** ~50 minutes  
**Level:** Intermediate

---

## VAE Architecture Overview

```
                    ┌──────────────┐
                    │    Input x   │
                    │   (784-dim)  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Encoder    │
                    │  FC Layers   │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌──────────────┐         ┌──────────────┐
       │  μ (mean)    │         │ logσ² (logvar)│
       │  (32-dim)    │         │   (32-dim)    │
       └──────┬───────┘         └──────┬────────┘
              │                        │
              └────────────┬───────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Reparameterize│
                    │ z = μ + σ·ε  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Decoder    │
                    │  FC Layers   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Output x̂   │
                    │  (784-dim)   │
                    └──────────────┘
```

---

## Part 1: Complete VAE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class VAE(nn.Module):
    """
    Variational Autoencoder for MNIST.
    
    Architecture:
    - Encoder: 784 → 256 → 256 → (μ, logσ²) each 32-dim
    - Decoder: 32 → 256 → 256 → 784
    
    Args:
        input_dim: Input dimension (784 for MNIST)
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 32):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # ============== ENCODER ==============
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ============== DECODER ==============
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input to latent distribution parameters."""
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Full forward pass: encode → reparameterize → decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples from p(z) = N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
```

---

## Part 2: VAE Loss Function

```python
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, 
                       mu: torch.Tensor, logvar: torch.Tensor,
                       beta: float = 1.0,
                       reduction: str = 'sum') -> tuple:
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        recon_x: Reconstructed output [batch_size, input_dim]
        x: Original input [batch_size, input_dim]
        mu: Encoder mean [batch_size, latent_dim]
        logvar: Encoder log-variance [batch_size, latent_dim]
        beta: Weight for KL term (β=1 for standard VAE)
        reduction: 'sum' or 'mean'
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction term
        kl_loss: KL divergence term
    """
    # ============== RECONSTRUCTION LOSS ==============
    # Binary Cross-Entropy for Bernoulli decoder
    # -E_q[log p(x|z)] where p(x|z) = Bernoulli(decoder(z))
    if reduction == 'sum':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
    
    # ============== KL DIVERGENCE ==============
    # KL(q(z|x) || p(z)) where q = N(μ, σ²) and p = N(0, I)
    # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    if reduction == 'mean':
        kl_loss = kl_loss / x.size(0)
    
    # ============== TOTAL LOSS ==============
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def vae_loss_mse(recon_x: torch.Tensor, x: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor,
                  beta: float = 1.0) -> tuple:
    """
    VAE Loss with MSE reconstruction (Gaussian decoder).
    
    Use when output is continuous and not bounded to [0, 1].
    """
    # MSE reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
```

---

## Part 3: Training Functions

```python
def train_vae_epoch(model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer, device: torch.device,
                    beta: float = 1.0) -> dict:
    """
    Train VAE for one epoch.
    
    Args:
        model: VAE model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        beta: KL weight
        
    Returns:
        Dictionary with loss statistics
    """
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_samples = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten images
        data = data.view(data.size(0), -1).to(device)
        batch_size = data.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss_function(
            recon_batch, data, mu, logvar, beta=beta
        )
        
        # Backward pass
        loss.backward()
        
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_samples += batch_size
    
    return {
        'loss': total_loss / num_samples,
        'recon_loss': total_recon / num_samples,
        'kl_loss': total_kl / num_samples
    }


def evaluate_vae(model: nn.Module, test_loader: DataLoader,
                 device: torch.device, beta: float = 1.0) -> dict:
    """Evaluate VAE on test set."""
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            batch_size = data.size(0)
            
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, data, mu, logvar, beta=beta
            )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_samples += batch_size
    
    return {
        'loss': total_loss / num_samples,
        'recon_loss': total_recon / num_samples,
        'kl_loss': total_kl / num_samples
    }
```

---

## Part 4: Visualization Functions

```python
def visualize_reconstructions(model: nn.Module, test_loader: DataLoader,
                               device: torch.device, num_images: int = 10):
    """Visualize original vs reconstructed images."""
    model.eval()
    
    data, _ = next(iter(test_loader))
    data = data[:num_images]
    data_flat = data.view(data.size(0), -1).to(device)
    
    with torch.no_grad():
        recon, _, _ = model(data_flat)
    
    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(data[i].squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Reconstructed
        axes[1, i].imshow(recon[i].cpu().reshape(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig('vae_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_samples(model: nn.Module, device: torch.device,
                      num_samples: int = 100):
    """Generate and visualize random samples from the VAE."""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    # Arrange in grid
    nrow = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(nrow, nrow, figsize=(10, 10))
    
    for i in range(nrow):
        for j in range(nrow):
            idx = i * nrow + j
            axes[i, j].imshow(samples[idx].cpu().reshape(28, 28).numpy(), cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle('Random Samples from VAE', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_latent_space(model: nn.Module, test_loader: DataLoader,
                            device: torch.device):
    """Visualize 2D latent space (requires latent_dim=2)."""
    model.eval()
    
    all_mu = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_mu = np.concatenate(all_mu, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], 
                         c=all_labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space (μ only)')
    plt.savefig('vae_latent_space.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_interpolation(model: nn.Module, test_loader: DataLoader,
                             device: torch.device, num_steps: int = 10):
    """Interpolate between two images in latent space."""
    model.eval()
    
    data, labels = next(iter(test_loader))
    
    # Find two different digits
    idx1, idx2 = 0, 1
    for i in range(len(labels)):
        if labels[i] != labels[0]:
            idx2 = i
            break
    
    x1 = data[idx1:idx1+1].view(1, -1).to(device)
    x2 = data[idx2:idx2+1].view(1, -1).to(device)
    
    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        
        interpolations = []
        for t in np.linspace(0, 1, num_steps):
            z = (1 - t) * mu1 + t * mu2
            recon = model.decode(z)
            interpolations.append(recon.cpu().numpy())
    
    # Plot
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
    for i in range(num_steps):
        axes[i].imshow(interpolations[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    
    axes[0].set_title(f'Digit {labels[idx1].item()}')
    axes[-1].set_title(f'Digit {labels[idx2].item()}')
    plt.suptitle('Latent Space Interpolation', y=1.05)
    plt.savefig('vae_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## Part 5: Complete Training Script

```python
def main():
    """Complete VAE training script."""
    
    # ============== CONFIGURATION ==============
    config = {
        'input_dim': 784,
        'hidden_dim': 256,
        'latent_dim': 32,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 20,
        'beta': 1.0,  # KL weight
    }
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============== DATA ==============
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=2)
    
    # ============== MODEL ==============
    model = VAE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # ============== TRAINING ==============
    train_history = {'loss': [], 'recon': [], 'kl': []}
    test_history = {'loss': [], 'recon': [], 'kl': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_stats = train_vae_epoch(model, train_loader, optimizer, 
                                       device, beta=config['beta'])
        
        # Evaluate
        test_stats = evaluate_vae(model, test_loader, device, 
                                   beta=config['beta'])
        
        # Record history
        train_history['loss'].append(train_stats['loss'])
        train_history['recon'].append(train_stats['recon_loss'])
        train_history['kl'].append(train_stats['kl_loss'])
        
        test_history['loss'].append(test_stats['loss'])
        test_history['recon'].append(test_stats['recon_loss'])
        test_history['kl'].append(test_stats['kl_loss'])
        
        # Print progress
        print(f"Epoch {epoch:2d}/{config['num_epochs']} | "
              f"Train Loss: {train_stats['loss']:.4f} "
              f"(Recon: {train_stats['recon_loss']:.4f}, KL: {train_stats['kl_loss']:.4f}) | "
              f"Test Loss: {test_stats['loss']:.4f}")
    
    # ============== VISUALIZATIONS ==============
    print("\nGenerating visualizations...")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(train_history['loss'], label='Train')
    axes[0].plot(test_history['loss'], label='Test')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    
    axes[1].plot(train_history['recon'], label='Train')
    axes[1].plot(test_history['recon'], label='Test')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    
    axes[2].plot(train_history['kl'], label='Train')
    axes[2].plot(test_history['kl'], label='Test')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('vae_training_curves.png', dpi=150)
    plt.show()
    
    # Other visualizations
    visualize_reconstructions(model, test_loader, device)
    visualize_samples(model, device, num_samples=100)
    visualize_interpolation(model, test_loader, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_history': train_history,
        'test_history': test_history
    }, 'vae_model.pth')
    
    print("\nTraining complete! Model saved to 'vae_model.pth'")


if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Architecture Variations

a) Increase hidden_dim to 512. Does performance improve?
b) Try latent_dim of 2, 8, 64, 128. How does it affect generation?
c) Add batch normalization. Does training become more stable?

### Exercise 2: Loss Monitoring

Track and plot:
- Reconstruction loss per digit class
- KL divergence distribution across latent dimensions
- Which latent dimensions are "active" (variance > threshold)?

### Exercise 3: Generation Quality

a) Generate 1000 samples and visually assess quality
b) Train a classifier on real MNIST, evaluate on generated samples
c) Compare samples from different latent_dim settings

---

## Summary

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Encoder** | Learn q(z\|x) parameters | FC layers → (μ, logvar) |
| **Reparameterize** | Enable backprop through sampling | z = μ + σ·ε |
| **Decoder** | Learn p(x\|z) | FC layers → Sigmoid |
| **Loss** | ELBO = Recon + KL | BCE + KL divergence |

---

## Next: Training VAEs

The next section covers training dynamics, common issues, and solutions.
