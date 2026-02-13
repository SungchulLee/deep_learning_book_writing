"""
Simple Autoencoder
Basic deterministic encoder-decoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    """
    Basic Autoencoder for dimensionality reduction and reconstruction.
    
    Args:
        input_dim (int): Input dimension (e.g., 784 for MNIST)
        hidden_dim (int): Hidden layer dimension
        latent_dim (int): Latent space dimension
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super(SimpleAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode -> decode"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction
    
    def loss_function(self, reconstruction, x):
        """Compute MSE reconstruction loss"""
        return F.mse_loss(reconstruction, x, reduction='sum')


if __name__ == '__main__':
    # Test the model
    model = SimpleAutoencoder(input_dim=784, latent_dim=32)
    x = torch.randn(32, 784)  # Batch of 32 samples
    
    reconstruction = model(x)
    loss = model.loss_function(reconstruction, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Loss: {loss.item():.4f}")
