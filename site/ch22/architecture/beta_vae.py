"""
β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
Introduces β hyperparameter for controlling disentanglement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    """
    β-VAE for learning disentangled representations.
    
    The β parameter controls the trade-off between reconstruction
    and disentanglement. Higher β encourages more disentangled
    latent representations.
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        latent_dim (int): Latent space dimension
        beta (float): Weight for KL divergence (default: 4.0)
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, beta=4.0):
        super(BetaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
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
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor
            
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
        Reparameterization trick.
        
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
        Decode latent representation to output.
        
        Args:
            z: Latent vector
            
        Returns:
            reconstruction: Reconstructed output
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            reconstruction: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar):
        """
        β-VAE Loss = Reconstruction Loss + β * KL Divergence
        
        Higher β values encourage more disentangled representations
        by increasing the penalty on the KL divergence term.
        
        Args:
            reconstruction: Reconstructed output
            x: Original input
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            
        Returns:
            loss: Total β-VAE loss
            bce: Reconstruction loss
            kld: KL divergence
        """
        # Reconstruction loss
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # β-VAE loss
        return BCE + self.beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        Generate samples from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def traverse_latent_dimension(self, dim_idx, num_steps=10, range_limit=3.0, device='cpu'):
        """
        Traverse a single latent dimension to visualize what it encodes.
        This is useful for understanding disentanglement.
        
        Args:
            dim_idx: Index of the latent dimension to traverse
            num_steps: Number of steps in the traversal
            range_limit: Range of values to traverse [-range_limit, range_limit]
            device: Device to generate on
            
        Returns:
            traversals: Generated samples along the dimension
        """
        # Start with zeros
        z = torch.zeros(num_steps, self.latent_dim).to(device)
        
        # Vary only the specified dimension
        values = torch.linspace(-range_limit, range_limit, num_steps).to(device)
        z[:, dim_idx] = values
        
        # Decode
        traversals = self.decode(z)
        return traversals


class ConvBetaVAE(nn.Module):
    """
    Convolutional β-VAE for image data.
    
    Args:
        latent_dim (int): Latent space dimension
        beta (float): Weight for KL divergence
        img_channels (int): Number of input image channels
    """
    
    def __init__(self, latent_dim=128, beta=4.0, img_channels=1):
        super(ConvBetaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        self.flatten_size = 128 * 4 * 4
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar):
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + self.beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def traverse_latent_dimension(self, dim_idx, num_steps=10, range_limit=3.0, device='cpu'):
        z = torch.zeros(num_steps, self.latent_dim).to(device)
        values = torch.linspace(-range_limit, range_limit, num_steps).to(device)
        z[:, dim_idx] = values
        return self.decode(z)


if __name__ == '__main__':
    # Test β-VAE
    print("Testing β-VAE...")
    model = BetaVAE(input_dim=784, latent_dim=10, beta=4.0)
    x = torch.randn(32, 784)
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    print(f"Beta: {model.beta}")
    
    # Test latent traversal
    traversals = model.traverse_latent_dimension(dim_idx=0, num_steps=10)
    print(f"Latent traversal shape: {traversals.shape}")
    
    # Test Conv β-VAE
    print("\nTesting Convolutional β-VAE...")
    conv_model = ConvBetaVAE(latent_dim=64, beta=10.0)
    x_img = torch.randn(32, 1, 28, 28)
    
    reconstruction, mu, logvar = conv_model(x_img)
    loss, bce, kld = conv_model.loss_function(reconstruction, x_img, mu, logvar)
    
    print(f"Input shape: {x_img.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Beta: {conv_model.beta}")
