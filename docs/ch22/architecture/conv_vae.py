"""
Convolutional Variational Autoencoder (ConvVAE)
Uses convolutional layers for better spatial feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional VAE for image data.
    
    Args:
        latent_dim (int): Latent space dimension
        img_channels (int): Number of input image channels (1 for grayscale, 3 for RGB)
        img_size (int): Input image size (assumes square images)
    """
    
    def __init__(self, latent_dim=128, img_channels=1, img_size=28):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.flatten_size = 128 * 4 * 4
        
        # Latent space parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        Encode input image to latent distribution parameters.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent representation to image.
        
        Args:
            z: Latent vector
            
        Returns:
            reconstruction: Reconstructed image
        """
        h = self.decoder_input(z)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        """
        Full forward pass.
        
        Args:
            x: Input image tensor
            
        Returns:
            reconstruction: Reconstructed image
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        VAE Loss function.
        
        Args:
            reconstruction: Reconstructed output
            x: Original input
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            beta: Weight for KL divergence term
            
        Returns:
            loss: Total VAE loss
            bce: Reconstruction loss
            kld: KL divergence
        """
        # Reconstruction loss
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        Generate samples from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated image samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    # Test the model
    model = ConvVAE(latent_dim=128, img_channels=1, img_size=28)
    x = torch.randn(32, 1, 28, 28)  # Batch of 32 grayscale 28x28 images
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # Test sampling
    samples = model.sample(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
