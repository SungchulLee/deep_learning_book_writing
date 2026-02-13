"""
Convolutional Conditional Variational Autoencoder (ConvCVAE)
Combines convolutional architecture with conditional generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvConditionalVAE(nn.Module):
    """
    Convolutional Conditional VAE for conditional image generation.
    
    Args:
        latent_dim (int): Latent space dimension
        num_classes (int): Number of classes for conditioning
        img_channels (int): Number of input image channels
        img_size (int): Input image size (assumes square images)
    """
    
    def __init__(self, latent_dim=128, num_classes=10, img_channels=1, img_size=28):
        super(ConvConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Label embedding for spatial conditioning
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # Encoder - takes image + embedded label as extra channel
        self.encoder = nn.Sequential(
            # Input: img_channels + 1 (for embedded label)
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.flatten_size = 128 * 4 * 4
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder input: latent + one-hot class
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_size)
        
        # Decoder
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
    
    def encode(self, x, labels):
        """
        Encode input image conditioned on class label.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            labels: Class labels [batch_size]
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        batch_size = x.size(0)
        
        # Embed labels and reshape to spatial format
        c_embedded = self.label_embedding(labels)
        c_embedded = c_embedded.view(batch_size, 1, self.img_size, self.img_size)
        
        # Concatenate image and embedded label
        x_combined = torch.cat([x, c_embedded], dim=1)
        
        # Encode
        h = self.encoder(x_combined)
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
    
    def decode(self, z, labels):
        """
        Decode latent representation conditioned on class label.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Class labels [batch_size]
            
        Returns:
            reconstruction: Reconstructed image
        """
        # One-hot encode labels
        c_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Concatenate latent code and condition
        z_combined = torch.cat([z, c_onehot], dim=1)
        
        # Decode
        h = self.decoder_input(z_combined)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x, labels):
        """
        Full forward pass with conditioning.
        
        Args:
            x: Input image tensor
            labels: Class labels
            
        Returns:
            reconstruction: Reconstructed image
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels)
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
            samples: Generated image samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        samples = self.decode(z, labels)
        return samples
    
    def interpolate_classes(self, z, class1, class2, num_steps=10):
        """
        Interpolate between two classes with fixed latent code.
        
        Args:
            z: Fixed latent code [1, latent_dim]
            class1: Starting class
            class2: Ending class
            num_steps: Number of interpolation steps
            
        Returns:
            interpolations: Interpolated samples
        """
        device = z.device
        interpolations = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # Create soft labels for interpolation
            c1_onehot = F.one_hot(torch.tensor([class1]), num_classes=self.num_classes).float().to(device)
            c2_onehot = F.one_hot(torch.tensor([class2]), num_classes=self.num_classes).float().to(device)
            c_interpolated = (1 - alpha) * c1_onehot + alpha * c2_onehot
            
            # Decode
            z_combined = torch.cat([z, c_interpolated], dim=1)
            h = self.decoder_input(z_combined)
            sample = self.decoder(h)
            interpolations.append(sample)
        
        return torch.cat(interpolations, dim=0)


if __name__ == '__main__':
    # Test the model
    model = ConvConditionalVAE(latent_dim=128, num_classes=10, img_channels=1, img_size=28)
    x = torch.randn(32, 1, 28, 28)
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
    samples = model.sample(class_label=7, num_samples=10)
    print(f"Generated samples (class 7) shape: {samples.shape}")
