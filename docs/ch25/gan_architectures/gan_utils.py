"""
GAN Utilities

This module contains utility functions for training, visualizing, and evaluating GANs.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Tuple, List


def weights_init(m):
    """
    Initialize network weights following DCGAN paper recommendations.
    
    Conv/ConvTranspose layers: mean=0, std=0.02
    BatchNorm layers: weight=1, bias=0
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_samples(generator: nn.Module, epoch: int, device: str, 
                fixed_noise: torch.Tensor, filename: str = None):
    """
    Generate and save sample images from the generator.
    
    Args:
        generator: Generator network
        epoch: Current epoch number
        device: Device to run on
        fixed_noise: Fixed noise for consistent visualization
        filename: Optional filename, defaults to 'samples_epoch_{epoch}.png'
    """
    generator.eval()
    
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    
    # Denormalize from [-1, 1] to [0, 1]
    fake = (fake + 1) / 2.0
    
    # Create grid
    grid = make_grid(fake, nrow=8, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f'Generated Samples - Epoch {epoch}')
    
    if filename is None:
        filename = f'samples_epoch_{epoch:04d}.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    generator.train()


def plot_training_progress(g_losses: List[float], d_losses: List[float],
                          filename: str = 'training_progress.png'):
    """
    Plot generator and discriminator loss curves.
    
    Args:
        g_losses: List of generator losses
        d_losses: List of discriminator losses
        filename: Output filename
    """
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training progress to {filename}")


def plot_discriminator_outputs(discriminator: nn.Module, real_data: torch.Tensor,
                               generator: nn.Module, noise: torch.Tensor,
                               device: str, filename: str = 'discriminator_outputs.png'):
    """
    Plot histogram of discriminator outputs on real and fake data.
    
    Args:
        discriminator: Discriminator network
        real_data: Real data samples
        generator: Generator network
        noise: Noise for generating fake samples
        device: Device to run on
        filename: Output filename
    """
    discriminator.eval()
    generator.eval()
    
    with torch.no_grad():
        # Get discriminator outputs on real data
        d_real = discriminator(real_data).cpu().numpy()
        
        # Generate fake data and get discriminator outputs
        fake_data = generator(noise)
        d_fake = discriminator(fake_data).cpu().numpy()
    
    # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.hist(d_real, bins=50, alpha=0.5, label='Real', color='blue')
    plt.hist(d_fake, bins=50, alpha=0.5, label='Fake', color='red')
    plt.xlabel('Discriminator Output')
    plt.ylabel('Frequency')
    plt.title('Discriminator Output Distribution')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved discriminator outputs to {filename}")
    
    discriminator.train()
    generator.train()


def interpolate_latent(generator: nn.Module, z1: torch.Tensor, z2: torch.Tensor,
                      steps: int = 10, device: str = 'cpu',
                      filename: str = 'interpolation.png'):
    """
    Generate interpolation between two latent vectors.
    
    Args:
        generator: Generator network
        z1: First latent vector
        z2: Second latent vector
        steps: Number of interpolation steps
        device: Device to run on
        filename: Output filename
    """
    generator.eval()
    
    # Linear interpolation
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated_samples = []
    
    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            sample = generator(z)
            interpolated_samples.append(sample)
    
    # Concatenate and denormalize
    samples = torch.cat(interpolated_samples, dim=0)
    samples = (samples + 1) / 2.0
    
    # Create grid
    grid = make_grid(samples, nrow=steps, padding=2)
    
    # Plot
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Latent Space Interpolation')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved interpolation to {filename}")
    
    generator.train()


def generate_latent_grid(generator: nn.Module, latent_dim: int = 100,
                        grid_size: int = 10, device: str = 'cpu',
                        filename: str = 'latent_grid.png'):
    """
    Generate a grid of samples by varying two latent dimensions.
    
    Args:
        generator: Generator network
        latent_dim: Dimensionality of latent space
        grid_size: Size of the grid (grid_size x grid_size)
        device: Device to run on
        filename: Output filename
    """
    generator.eval()
    
    # Create grid of values for two dimensions
    x = torch.linspace(-2, 2, grid_size)
    y = torch.linspace(-2, 2, grid_size)
    
    samples = []
    
    with torch.no_grad():
        for yi in y:
            for xi in x:
                # Create latent vector (all zeros except two dimensions)
                z = torch.randn(1, latent_dim, device=device) * 0.5
                z[0, 0] = xi
                z[0, 1] = yi
                
                sample = generator(z)
                samples.append(sample)
    
    # Concatenate and denormalize
    samples = torch.cat(samples, dim=0)
    samples = (samples + 1) / 2.0
    
    # Create grid
    grid = make_grid(samples, nrow=grid_size, padding=2)
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Latent Space Grid (z[0] and z[1])')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent grid to {filename}")
    
    generator.train()


class GANLosses:
    """Collection of different GAN loss functions."""
    
    @staticmethod
    def vanilla_gan_loss(d_real: torch.Tensor, d_fake: torch.Tensor,
                        mode: str = 'discriminator') -> torch.Tensor:
        """
        Original GAN loss (binary cross-entropy).
        
        Args:
            d_real: Discriminator output on real data
            d_fake: Discriminator output on fake data
            mode: 'discriminator' or 'generator'
        
        Returns:
            Loss value
        """
        criterion = nn.BCELoss()
        
        if mode == 'discriminator':
            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)
            
            real_loss = criterion(d_real, real_labels)
            fake_loss = criterion(d_fake, fake_labels)
            
            return real_loss + fake_loss
        
        elif mode == 'generator':
            real_labels = torch.ones_like(d_fake)
            return criterion(d_fake, real_labels)
    
    @staticmethod
    def nonsaturating_loss(d_fake: torch.Tensor) -> torch.Tensor:
        """
        Non-saturating generator loss.
        
        Args:
            d_fake: Discriminator output on fake data
        
        Returns:
            Generator loss
        """
        return -torch.mean(torch.log(d_fake + 1e-8))
    
    @staticmethod
    def wasserstein_loss(d_real: torch.Tensor, d_fake: torch.Tensor,
                        mode: str = 'discriminator') -> torch.Tensor:
        """
        Wasserstein GAN loss.
        
        Args:
            d_real: Discriminator (critic) output on real data
            d_fake: Discriminator (critic) output on fake data
            mode: 'discriminator' or 'generator'
        
        Returns:
            Loss value
        """
        if mode == 'discriminator':
            return -(torch.mean(d_real) - torch.mean(d_fake))
        elif mode == 'generator':
            return -torch.mean(d_fake)


def label_smoothing(labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Apply label smoothing to real/fake labels.
    
    Args:
        labels: Original labels (0 or 1)
        smoothing: Smoothing amount
    
    Returns:
        Smoothed labels
    """
    return labels * (1 - smoothing) + smoothing * 0.5


def add_noise_to_inputs(data: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """
    Add noise to discriminator inputs (helps training stability).
    
    Args:
        data: Input data
        noise_std: Standard deviation of noise
    
    Returns:
        Noisy data
    """
    noise = torch.randn_like(data) * noise_std
    return data + noise


def calculate_gradient_penalty(discriminator: nn.Module, real_data: torch.Tensor,
                               fake_data: torch.Tensor, device: str,
                               lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Calculate gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator network
        real_data: Real data samples
        fake_data: Generated fake samples
        device: Device to run on
        lambda_gp: Gradient penalty coefficient
    
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    
    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    
    # Get discriminator output
    d_interpolates = discriminator(interpolates)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Calculate penalty
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def save_checkpoint(generator: nn.Module, discriminator: nn.Module,
                   g_optimizer, d_optimizer, epoch: int,
                   filename: str = 'checkpoint.pth'):
    """
    Save model checkpoint.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        epoch: Current epoch
        filename: Checkpoint filename
    """
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }, filename)
    print(f"Saved checkpoint to {filename}")


def load_checkpoint(generator: nn.Module, discriminator: nn.Module,
                   g_optimizer, d_optimizer, filename: str, device: str):
    """
    Load model checkpoint.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        filename: Checkpoint filename
        device: Device to load on
    
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(filename, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {filename}, epoch {epoch}")
    
    return epoch
