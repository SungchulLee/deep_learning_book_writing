"""
Conditional Variational Autoencoder (cVAE)
Allows controlled generation by conditioning on labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for controlled generation.
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        latent_dim (int): Latent space dimension
        num_classes (int): Number of classes for conditioning
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder: takes input + one-hot label
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: takes latent + one-hot label
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        """
        Encode input conditioned on class label.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            c: One-hot encoded labels [batch_size, num_classes]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
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
    
    def decode(self, z, c):
        """
        Decode latent representation conditioned on class label.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            c: One-hot encoded labels [batch_size, num_classes]
            
        Returns:
            reconstruction: Reconstructed output
        """
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, labels):
        """
        Full forward pass with conditioning.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            labels: Class labels [batch_size] (will be one-hot encoded)
            
        Returns:
            reconstruction: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        # One-hot encode labels if they're not already
        if len(labels.shape) == 1:
            c = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            c = labels
        
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, c)
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
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, class_label, num_samples, device='cpu'):
        """
        Generate samples conditioned on a specific class.
        
        Args:
            class_label: Class to generate (int)
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        c = F.one_hot(labels, num_classes=self.num_classes).float()
        samples = self.decode(z, c)
        return samples
    
    def interpolate(self, class1, class2, num_steps=10, device='cpu'):
        """
        Interpolate between two classes in latent space.
        
        Args:
            class1: Starting class
            class2: Ending class
            num_steps: Number of interpolation steps
            device: Device to generate on
            
        Returns:
            interpolations: Interpolated samples
        """
        # Sample latent vectors for both classes
        z = torch.randn(1, self.latent_dim).to(device)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps).to(device)
        
        interpolations = []
        for alpha in alphas:
            # Interpolate class labels
            c1 = F.one_hot(torch.tensor([class1]), num_classes=self.num_classes).float().to(device)
            c2 = F.one_hot(torch.tensor([class2]), num_classes=self.num_classes).float().to(device)
            c = (1 - alpha) * c1 + alpha * c2
            
            # Generate sample
            sample = self.decode(z, c)
            interpolations.append(sample)
        
        return torch.cat(interpolations, dim=0)


if __name__ == '__main__':
    # Test the model
    model = ConditionalVAE(input_dim=784, latent_dim=32, num_classes=10)
    x = torch.randn(32, 784)
    labels = torch.randint(0, 10, (32,))
    
    reconstruction, mu, logvar = model(x, labels)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # Test conditional sampling
    samples = model.sample(class_label=5, num_samples=10)
    print(f"Generated samples (class 5) shape: {samples.shape}")
    
    # Test interpolation
    interpolations = model.interpolate(class1=3, class2=8, num_steps=10)
    print(f"Interpolated samples shape: {interpolations.shape}")
