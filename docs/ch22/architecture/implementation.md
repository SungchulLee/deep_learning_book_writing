# PyTorch Implementation

Complete Variational Autoencoder implementation with training pipeline and visualization.

---

## Learning Objectives

By the end of this section, you will be able to:

- Implement a complete VAE from scratch in PyTorch
- Build a full training and evaluation pipeline
- Visualize reconstructions, samples, latent space, and interpolations
- Run a self-contained training script for MNIST

---

## Complete VAE Model

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

## Loss Functions

```python
def vae_loss_bce(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with BCE reconstruction (Bernoulli decoder)."""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def vae_loss_mse(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with MSE reconstruction (Gaussian decoder)."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

---

## Training and Evaluation

```python
def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """Train VAE for one epoch."""
    model.train()
    total_loss, total_recon, total_kl, num_samples = 0, 0, 0, 0
    
    for data, _ in train_loader:
        data = data.view(data.size(0), -1).to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss_bce(recon_batch, data, mu, logvar, beta)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_samples += data.size(0)
    
    return {
        'loss': total_loss / num_samples,
        'recon_loss': total_recon / num_samples,
        'kl_loss': total_kl / num_samples
    }


def evaluate(model, test_loader, device, beta=1.0):
    """Evaluate VAE on test set."""
    model.eval()
    total_loss, total_recon, total_kl, num_samples = 0, 0, 0, 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss_bce(recon_batch, data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_samples += data.size(0)
    
    return {
        'loss': total_loss / num_samples,
        'recon_loss': total_recon / num_samples,
        'kl_loss': total_kl / num_samples
    }
```

---

## Visualization Functions

### Reconstructions

```python
def visualize_reconstructions(model, test_loader, device, num_images=10):
    """Visualize original vs reconstructed images."""
    model.eval()
    
    data, _ = next(iter(test_loader))
    data = data[:num_images]
    data_flat = data.view(data.size(0), -1).to(device)
    
    with torch.no_grad():
        recon, _, _ = model(data_flat)
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        axes[0, i].imshow(data[i].squeeze().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        axes[1, i].imshow(recon[i].cpu().reshape(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig('vae_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Random Samples

```python
def visualize_samples(model, device, num_samples=100):
    """Generate and visualize random samples from the VAE."""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
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
```

### Latent Space

```python
def visualize_latent_space(model, test_loader, device):
    """Visualize 2D latent space (requires latent_dim=2 or uses first 2 dims)."""
    model.eval()
    
    all_mu, all_labels = [], []
    
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
    plt.title('VAE Latent Space (μ)')
    plt.savefig('vae_latent_space.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Interpolation

```python
def visualize_interpolation(model, test_loader, device, num_steps=10):
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

## Complete Training Script

```python
def main():
    """Complete VAE training script for MNIST."""
    
    # ============== CONFIGURATION ==============
    config = {
        'input_dim': 784,
        'hidden_dim': 256,
        'latent_dim': 32,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 20,
        'beta': 1.0,
    }
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============== DATA ==============
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
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
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # ============== TRAINING ==============
    history = {'train': [], 'test': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device, config['beta'])
        test_stats = evaluate(model, test_loader, device, config['beta'])
        
        history['train'].append(train_stats)
        history['test'].append(test_stats)
        
        print(f"Epoch {epoch:2d}/{config['num_epochs']} | "
              f"Train Loss: {train_stats['loss']:.4f} "
              f"(R:{train_stats['recon_loss']:.4f}, K:{train_stats['kl_loss']:.4f}) | "
              f"Test Loss: {test_stats['loss']:.4f}")
    
    # ============== VISUALIZATION ==============
    print("\nGenerating visualizations...")
    
    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, key in enumerate(['loss', 'recon_loss', 'kl_loss']):
        axes[i].plot([h[key] for h in history['train']], label='Train')
        axes[i].plot([h[key] for h in history['test']], label='Test')
        axes[i].set_title(key.replace('_', ' ').title())
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('vae_training_curves.png', dpi=150)
    plt.show()
    
    visualize_reconstructions(model, test_loader, device)
    visualize_samples(model, device, num_samples=100)
    visualize_interpolation(model, test_loader, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, 'vae_model.pth')
    
    print("\nTraining complete! Model saved to 'vae_model.pth'")


if __name__ == "__main__":
    main()
```

---

## Summary

| Component | Purpose | Key Code |
|-----------|---------|----------|
| **VAE class** | Complete model with encode/decode/sample | `VAE(input_dim, hidden_dim, latent_dim)` |
| **Loss functions** | BCE and MSE variants | `vae_loss_bce`, `vae_loss_mse` |
| **Training** | Epoch loop with gradient clipping | `train_epoch`, `evaluate` |
| **Visualization** | Reconstructions, samples, latent space, interpolation | Four visualization functions |
| **Full script** | End-to-end MNIST training | `main()` |

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

## What's Next

The [Posterior Collapse](posterior_collapse.md) section analyzes a common training failure mode and how to prevent it.
