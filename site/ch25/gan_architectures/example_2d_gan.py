"""
Simple 2D GAN

Visualize GAN training on 2D toy data.
Perfect for understanding adversarial training dynamics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


class SimpleGenerator(nn.Module):
    """Simple generator for 2D data."""
    
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, z):
        return self.model(z)


class SimpleDiscriminator(nn.Module):
    """Simple discriminator for 2D data."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def generate_data(n_samples=1000, dataset='moons'):
    """Generate 2D toy datasets."""
    if dataset == 'moons':
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
    elif dataset == 'circles':
        from sklearn.datasets import make_circles
        data, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
    elif dataset == 'gaussian':
        # Two Gaussians
        data1 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([2, 2])
        data2 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([-2, -2])
        data = np.vstack([data1, data2])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return torch.tensor(data, dtype=torch.float32)


def visualize_training_step(generator, discriminator, real_data, epoch, 
                           fixed_noise, filename=None):
    """Visualize generator distribution and discriminator decision boundary."""
    generator.eval()
    discriminator.eval()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    with torch.no_grad():
        # Generate fake samples
        fake_data = generator(fixed_noise).cpu().numpy()
    
    real_data_np = real_data.cpu().numpy()
    
    # Plot 1: Real data
    axes[0].scatter(real_data_np[:, 0], real_data_np[:, 1], alpha=0.5, s=20)
    axes[0].set_title('Real Data')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Generated data
    axes[1].scatter(fake_data[:, 0], fake_data[:, 1], alpha=0.5, s=20, color='red')
    axes[1].set_title(f'Generated Data (Epoch {epoch})')
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Discriminator decision boundary
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                         dtype=torch.float32)
    
    with torch.no_grad():
        d_scores = discriminator(points).cpu().numpy()
    
    d_scores = d_scores.reshape(200, 200)
    
    contour = axes[2].contourf(X, Y, d_scores, levels=20, cmap='RdYlBu')
    axes[2].scatter(real_data_np[:, 0], real_data_np[:, 1], 
                   alpha=0.3, s=10, color='blue', label='Real')
    axes[2].scatter(fake_data[:, 0], fake_data[:, 1], 
                   alpha=0.3, s=10, color='red', label='Fake')
    axes[2].set_title('Discriminator Decision Boundary')
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].legend()
    plt.colorbar(contour, ax=axes[2], label='D(x)')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    generator.train()
    discriminator.train()


def train_gan_2d(data, latent_dim=2, n_epochs=1000, batch_size=256, 
                lr=0.0002, device='cpu'):
    """Train simple GAN on 2D data."""
    
    # Initialize networks
    generator = SimpleGenerator(latent_dim=latent_dim).to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Loss
    criterion = nn.BCELoss()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(500, latent_dim, device=device)
    
    # Training loop
    g_losses = []
    d_losses = []
    
    print("Training 2D GAN...")
    pbar = tqdm(range(n_epochs))
    
    for epoch in pbar:
        # Sample batch
        indices = torch.randint(0, len(data), (batch_size,))
        real_batch = data[indices].to(device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # Train Discriminator
        discriminator.zero_grad()
        
        # Real data
        d_real = discriminator(real_batch)
        real_loss = criterion(d_real, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = generator(noise)
        d_fake = discriminator(fake_batch.detach())
        fake_loss = criterion(d_fake, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        generator.zero_grad()
        
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_batch = generator(noise)
        d_fake = discriminator(fake_batch)
        g_loss = criterion(d_fake, real_labels)
        
        g_loss.backward()
        g_optimizer.step()
        
        # Record losses
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        pbar.set_postfix({
            'D_loss': f'{d_loss.item():.4f}',
            'G_loss': f'{g_loss.item():.4f}'
        })
        
        # Visualize progress
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            visualize_training_step(
                generator, discriminator, data, epoch,
                fixed_noise, filename=f'2d_gan_epoch_{epoch:04d}.png'
            )
    
    return generator, discriminator, g_losses, d_losses


def plot_loss_curves(g_losses, d_losses):
    """Plot training loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator', alpha=0.7)
    plt.plot(d_losses, label='Discriminator', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('2d_gan_losses.png', dpi=150)
    plt.close()
    print("Saved loss curves")


def main():
    """Main function for 2D GAN demo."""
    print("=" * 60)
    print("2D GAN Visualization Demo")
    print("=" * 60)
    
    # Configuration
    dataset = 'moons'  # Try: 'moons', 'circles', 'gaussian'
    n_samples = 2000
    latent_dim = 2
    n_epochs = 1000
    
    print(f"\nDataset: {dataset}")
    print(f"Samples: {n_samples}")
    print(f"Epochs: {n_epochs}\n")
    
    # Generate data
    data = generate_data(n_samples, dataset)
    
    # Plot original data
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.5, s=20)
    plt.title('Original Data')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True, alpha=0.3)
    plt.savefig('2d_gan_original_data.png', dpi=150)
    plt.close()
    
    # Train GAN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    generator, discriminator, g_losses, d_losses = train_gan_2d(
        data, latent_dim=latent_dim, n_epochs=n_epochs, device=device
    )
    
    # Plot loss curves
    plot_loss_curves(g_losses, d_losses)
    
    print("\n" + "=" * 60)
    print("Demo complete! Generated files:")
    print("  - 2d_gan_original_data.png: Original dataset")
    print("  - 2d_gan_epoch_*.png: Training progress")
    print("  - 2d_gan_losses.png: Loss curves")
    print("=" * 60)


if __name__ == "__main__":
    main()
