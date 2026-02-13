"""
DCGAN Training on MNIST

Complete training script for DCGAN on MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from dcgan import DCGANGenerator, DCGANDiscriminator
from gan_utils import (
    weights_init, save_samples, plot_training_progress,
    plot_discriminator_outputs, interpolate_latent, save_checkpoint
)


class DCGAN_MNIST:
    """Wrapper class for DCGAN training on MNIST."""
    
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64,
                 batch_size: int = 128, lr: float = 0.0002, beta1: float = 0.5,
                 device: str = None):
        """
        Initialize DCGAN for MNIST.
        
        Args:
            latent_dim: Dimension of latent vector
            feature_maps: Base number of feature maps
            batch_size: Training batch size
            lr: Learning rate
            beta1: Beta1 for Adam optimizer
            device: Device to train on
        """
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.generator = DCGANGenerator(
            latent_dim=latent_dim,
            image_channels=1,
            feature_maps=feature_maps
        ).to(self.device)
        
        self.discriminator = DCGANDiscriminator(
            image_channels=1,
            feature_maps=feature_maps
        ).to(self.device)
        
        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # Optimizers (following DCGAN paper: lr=0.0002, beta1=0.5)
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr, betas=(beta1, 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr, betas=(beta1, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Fixed noise for consistent visualization
        self.fixed_noise = torch.randn(64, latent_dim, device=self.device)
        
        # Count parameters
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")
    
    def get_dataloader(self):
        """Create MNIST dataloader."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    def train_discriminator(self, real_images: torch.Tensor) -> float:
        """
        Train discriminator for one step.
        
        Args:
            real_images: Batch of real images
        
        Returns:
            Discriminator loss
        """
        self.discriminator.zero_grad()
        
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Train on real images
        d_real = self.discriminator(real_images)
        real_loss = self.criterion(d_real, real_labels)
        
        # Train on fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        d_fake = self.discriminator(fake_images.detach())
        fake_loss = self.criterion(d_fake, fake_labels)
        
        # Combined loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self) -> float:
        """
        Train generator for one step.
        
        Returns:
            Generator loss
        """
        self.generator.zero_grad()
        
        # Generate fake images
        noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        
        # Try to fool discriminator
        d_fake = self.discriminator(fake_images)
        real_labels = torch.ones(self.batch_size, 1, device=self.device)
        
        g_loss = self.criterion(d_fake, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train(self, n_epochs: int = 50, save_interval: int = 5,
             d_steps: int = 1):
        """
        Train DCGAN.
        
        Args:
            n_epochs: Number of training epochs
            save_interval: Save samples every N epochs
            d_steps: Number of discriminator steps per generator step
        """
        dataloader = self.get_dataloader()
        
        os.makedirs('samples', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        
        g_losses = []
        d_losses = []
        
        print(f"\nTraining DCGAN for {n_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc="Training")
            
            for i, (real_images, _) in enumerate(pbar):
                real_images = real_images.to(self.device)
                
                # Train Discriminator
                for _ in range(d_steps):
                    d_loss = self.train_discriminator(real_images)
                
                # Train Generator
                g_loss = self.train_generator()
                
                # Record losses
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                num_batches += 1
                
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                
                pbar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}'
                })
            
            # Epoch statistics
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            
            print(f"Average G Loss: {avg_g_loss:.4f}")
            print(f"Average D Loss: {avg_d_loss:.4f}")
            
            # Save samples
            if epoch % save_interval == 0 or epoch == 1:
                save_samples(
                    self.generator, epoch, self.device,
                    self.fixed_noise,
                    filename=f'samples/epoch_{epoch:04d}.png'
                )
            
            # Save checkpoint
            if epoch % 25 == 0:
                save_checkpoint(
                    self.generator, self.discriminator,
                    self.g_optimizer, self.d_optimizer,
                    epoch,
                    filename=f'checkpoints/dcgan_epoch_{epoch}.pth'
                )
        
        # Final outputs
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        
        # Save final samples
        save_samples(
            self.generator, n_epochs, self.device,
            self.fixed_noise,
            filename='final_samples.png'
        )
        
        # Plot training progress
        plot_training_progress(g_losses, d_losses)
        
        # Save final checkpoint
        save_checkpoint(
            self.generator, self.discriminator,
            self.g_optimizer, self.d_optimizer,
            n_epochs,
            filename='dcgan_mnist_final.pth'
        )
        
        return g_losses, d_losses
    
    def generate_samples(self, n_samples: int = 64):
        """Generate samples from trained generator."""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(noise)
            
        return samples
    
    def visualize_discriminator(self, real_data: torch.Tensor):
        """Visualize discriminator outputs."""
        noise = torch.randn(real_data.size(0), self.latent_dim, device=self.device)
        plot_discriminator_outputs(
            self.discriminator, real_data,
            self.generator, noise,
            self.device
        )
    
    def generate_interpolation(self):
        """Generate interpolation between random latent vectors."""
        z1 = torch.randn(1, self.latent_dim, device=self.device)
        z2 = torch.randn(1, self.latent_dim, device=self.device)
        
        interpolate_latent(
            self.generator, z1, z2,
            steps=10, device=self.device
        )


def main():
    """Main training script."""
    print("=" * 60)
    print("DCGAN Training on MNIST")
    print("=" * 60)
    
    # Configuration
    config = {
        'latent_dim': 100,
        'feature_maps': 64,
        'batch_size': 128,
        'lr': 0.0002,
        'beta1': 0.5,
        'n_epochs': 50,
        'save_interval': 5,
        'd_steps': 1,  # Number of D updates per G update
    }
    
    print("\nConfiguration:")
    print("-" * 60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("-" * 60)
    
    # Initialize DCGAN
    dcgan = DCGAN_MNIST(
        latent_dim=config['latent_dim'],
        feature_maps=config['feature_maps'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        beta1=config['beta1']
    )
    
    # Train
    g_losses, d_losses = dcgan.train(
        n_epochs=config['n_epochs'],
        save_interval=config['save_interval'],
        d_steps=config['d_steps']
    )
    
    # Generate interpolation
    print("\nGenerating interpolation...")
    dcgan.generate_interpolation()
    
    print("\n" + "=" * 60)
    print("All done! Check the following:")
    print("  - samples/ : Generated images during training")
    print("  - final_samples.png : Final generated samples")
    print("  - training_progress.png : Loss curves")
    print("  - interpolation.png : Latent space interpolation")
    print("  - checkpoints/ : Model checkpoints")
    print("=" * 60)


if __name__ == "__main__":
    main()
