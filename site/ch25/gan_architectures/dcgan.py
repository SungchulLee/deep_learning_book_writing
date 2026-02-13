"""
Deep Convolutional GAN (DCGAN)

Implementation of DCGAN following the guidelines from:
"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
(Radford et al., 2016)

Key architectural principles:
1. Replace pooling with strided convolutions
2. Use batch normalization in both G and D
3. Remove fully connected hidden layers
4. Use ReLU in G (except output: Tanh)
5. Use LeakyReLU in D
"""

import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator Network.
    
    Maps latent vector z to image using transposed convolutions.
    """
    
    def __init__(self, latent_dim: int = 100, image_channels: int = 1, 
                 feature_maps: int = 64):
        """
        Args:
            latent_dim: Dimension of latent vector z
            image_channels: Number of output channels (1 for grayscale, 3 for RGB)
            feature_maps: Number of feature maps in first layer (scales by 2x each layer)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # For 28x28 images (MNIST), we need to start at 7x7 and upsample to 28x28
        # For 64x64 images, start at 4x4 and upsample to 64x64
        
        # Initial projection and reshape
        self.project = nn.Sequential(
            nn.Linear(latent_dim, feature_maps * 8 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 8 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Convolutional layers (upsampling)
        self.main = nn.Sequential(
            # Input: (feature_maps*8) x 7 x 7
            
            # Layer 1: Upsample to 14x14
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # Output: (feature_maps*4) x 14 x 14
            
            # Layer 2: Upsample to 28x28
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # Output: (feature_maps*2) x 28 x 28
            
            # Layer 3: Final convolution to image channels
            nn.Conv2d(feature_maps * 2, image_channels,
                     kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # Output: image_channels x 28 x 28
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate image from latent vector.
        
        Args:
            z: Latent vector, shape (batch_size, latent_dim) or (batch_size, latent_dim, 1, 1)
        
        Returns:
            Generated images, shape (batch_size, channels, height, width)
        """
        # Flatten if needed
        if z.dim() == 4:
            z = z.view(z.size(0), -1)
        
        # Project and reshape
        x = self.project(z)
        x = x.view(x.size(0), -1, 7, 7)
        
        # Generate image
        return self.main(x)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator Network.
    
    Classifies images as real or fake using strided convolutions.
    """
    
    def __init__(self, image_channels: int = 1, feature_maps: int = 64):
        """
        Args:
            image_channels: Number of input channels (1 for grayscale, 3 for RGB)
            feature_maps: Number of feature maps in first layer (scales by 2x each layer)
        """
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: image_channels x 28 x 28
            
            # Layer 1: No batch norm on input layer (DCGAN guideline)
            nn.Conv2d(image_channels, feature_maps, 
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: feature_maps x 14 x 14
            
            # Layer 2
            nn.Conv2d(feature_maps, feature_maps * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (feature_maps*2) x 7 x 7
            
            # Layer 3
            nn.Conv2d(feature_maps * 2, feature_maps * 4,
                     kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (feature_maps*4) x 4 x 4
            
            # Output layer: Convolve to single value
            nn.Conv2d(feature_maps * 4, 1,
                     kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Classify image as real or fake.
        
        Args:
            img: Input image, shape (batch_size, channels, height, width)
        
        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        output = self.main(img)
        return output.view(-1, 1)


class DCGAN64Generator(nn.Module):
    """
    DCGAN Generator for 64x64 images (following original paper more closely).
    """
    
    def __init__(self, latent_dim: int = 100, image_channels: int = 3,
                 feature_maps: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8,
                             kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 4 x 4
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: feature_maps x 32 x 32
            
            nn.ConvTranspose2d(feature_maps, image_channels,
                             kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: image_channels x 64 x 64
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate 64x64 image from latent vector."""
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


class DCGAN64Discriminator(nn.Module):
    """
    DCGAN Discriminator for 64x64 images (following original paper).
    """
    
    def __init__(self, image_channels: int = 3, feature_maps: int = 64):
        super().__init__()
        
        self.main = nn.Sequential(
            # Input: image_channels x 64 x 64
            nn.Conv2d(image_channels, feature_maps,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps x 32 x 32
            
            nn.Conv2d(feature_maps, feature_maps * 2,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8,
                     kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*8) x 4 x 4
            
            nn.Conv2d(feature_maps * 8, 1,
                     kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Classify 64x64 image."""
        output = self.main(img)
        return output.view(-1, 1)


def test_dcgan():
    """Test DCGAN architectures."""
    print("Testing DCGAN for 28x28 images (MNIST)...")
    
    # Test 28x28 version
    gen = DCGANGenerator(latent_dim=100, image_channels=1, feature_maps=64)
    disc = DCGANDiscriminator(image_channels=1, feature_maps=64)
    
    # Test forward pass
    z = torch.randn(16, 100)
    fake_imgs = gen(z)
    print(f"Generator output shape: {fake_imgs.shape}")
    
    d_output = disc(fake_imgs)
    print(f"Discriminator output shape: {d_output.shape}")
    
    # Count parameters
    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    print("\nTesting DCGAN for 64x64 images...")
    
    # Test 64x64 version
    gen64 = DCGAN64Generator(latent_dim=100, image_channels=3, feature_maps=64)
    disc64 = DCGAN64Discriminator(image_channels=3, feature_maps=64)
    
    z = torch.randn(16, 100, 1, 1)
    fake_imgs = gen64(z)
    print(f"Generator output shape: {fake_imgs.shape}")
    
    d_output = disc64(fake_imgs)
    print(f"Discriminator output shape: {d_output.shape}")
    
    g_params = sum(p.numel() for p in gen64.parameters())
    d_params = sum(p.numel() for p in disc64.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_dcgan()
