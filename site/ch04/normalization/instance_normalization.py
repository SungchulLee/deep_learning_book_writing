"""
Instance Normalization Implementation and Examples
===================================================

Instance Normalization normalizes each sample and each channel independently.
Widely used in style transfer and GANs where batch statistics shouldn't mix.

Paper: "Instance Normalization: The Missing Ingredient for Fast Stylization" 
       (Ulyanov et al., 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm2dNumPy:
    """
    Instance Normalization implementation from scratch using NumPy.
    Normalizes each sample and each channel independently.
    """
    
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        Args:
            num_features: Number of channels (C)
            eps: Small constant for numerical stability
            affine: If True, learn gamma and beta parameters
        """
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            # Learnable parameters per channel
            self.gamma = np.ones((1, num_features, 1, 1))
            self.beta = np.zeros((1, num_features, 1, 1))
        
    def forward(self, x):
        """
        Forward pass of Instance Normalization.
        
        Args:
            x: Input of shape (N, C, H, W)
            
        Returns:
            Normalized output of same shape
        """
        # Calculate mean and variance per instance and per channel
        # Average over spatial dimensions (H, W) for each (N, C)
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Apply affine transformation if enabled
        if self.affine:
            x_normalized = self.gamma * x_normalized + self.beta
        
        return x_normalized


class StyleTransferNetwork(nn.Module):
    """
    Style transfer network using Instance Normalization.
    InstanceNorm is crucial for style transfer as it removes instance-specific
    contrast information, making style transfer more effective.
    """
    
    def __init__(self):
        super(StyleTransferNetwork, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with Instance Normalization.
    """
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorWithInstanceNorm(nn.Module):
    """
    GAN Generator using Instance Normalization.
    Common in image-to-image translation (e.g., CycleGAN, Pix2Pix).
    """
    
    def __init__(self, input_channels=3, output_channels=3, ngf=64):
        super(GeneratorWithInstanceNorm, self).__init__()
        
        # Initial convolution
        model = [
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(9):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.Conv2d(ngf, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


def demonstrate_instance_norm():
    """
    Demonstrate how Instance Normalization works.
    """
    print("=" * 60)
    print("Instance Normalization Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create sample data: 2 images, 3 channels, 4x4 spatial
    batch_size, channels, height, width = 2, 3, 4, 4
    
    # Create data with different statistics per image and channel
    x = np.random.randn(batch_size, channels, height, width)
    
    # Make each channel have different scales
    x[0, 0] *= 10   # Image 1, Channel 1: large values
    x[0, 1] *= 1    # Image 1, Channel 2: normal values
    x[0, 2] *= 0.1  # Image 1, Channel 3: small values
    
    x[1, 0] *= 5    # Image 2, Channel 1: medium values
    x[1, 1] *= 15   # Image 2, Channel 2: very large values
    x[1, 2] *= 2    # Image 2, Channel 3: normal values
    
    print("\nOriginal data statistics:")
    for n in range(batch_size):
        print(f"\nImage {n}:")
        for c in range(channels):
            mean = np.mean(x[n, c])
            std = np.std(x[n, c])
            print(f"  Channel {c}: mean={mean:6.2f}, std={std:6.2f}")
    
    # Apply instance normalization
    instance_norm = InstanceNorm2dNumPy(channels)
    x_normalized = instance_norm.forward(x)
    
    print("\nAfter Instance Normalization:")
    for n in range(batch_size):
        print(f"\nImage {n}:")
        for c in range(channels):
            mean = np.mean(x_normalized[n, c])
            std = np.std(x_normalized[n, c])
            print(f"  Channel {c}: mean={mean:6.2f}, std={std:6.2f}")
    
    print("\nKey observations:")
    print("- Each (image, channel) pair is normalized independently")
    print("- Mean ≈ 0 and Std ≈ 1 for EACH channel of EACH image")
    print("- No mixing of statistics across samples or channels")


def compare_all_normalizations():
    """
    Compare Batch Norm, Layer Norm, and Instance Norm side by side.
    """
    print("\n" + "=" * 60)
    print("Comparing All Normalization Methods")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create sample data: (N=2, C=3, H=4, W=4)
    x = torch.randn(2, 3, 4, 4) * 10
    
    print("\nInput shape: (N=2, C=3, H=4, W=4)")
    print("N=batch, C=channels, H=height, W=width")
    
    # Batch Normalization (normalizes over N, H, W for each C)
    bn = nn.BatchNorm2d(3)
    bn.eval()
    x_bn = bn(x)
    
    # Layer Normalization (normalizes over C, H, W for each N)
    ln = nn.LayerNorm([3, 4, 4])
    x_ln = ln(x)
    
    # Instance Normalization (normalizes over H, W for each N and C)
    instance_norm = nn.InstanceNorm2d(3, affine=False)
    x_in = instance_norm(x)
    
    # Group Normalization with G=3 (each channel is its own group)
    gn = nn.GroupNorm(3, 3)  # Similar to InstanceNorm when each channel is a group
    x_gn = gn(x)
    
    print("\n" + "-" * 60)
    print("Statistics after normalization:")
    print("-" * 60)
    
    print("\nBatch Norm:")
    print(f"  Normalizes over: (N, H, W) for each C")
    print(f"  Mean per channel: {x_bn.mean(dim=(0, 2, 3))}")
    print(f"  Std per channel:  {x_bn.std(dim=(0, 2, 3))}")
    
    print("\nLayer Norm:")
    print(f"  Normalizes over: (C, H, W) for each N")
    print(f"  Mean per sample: {x_ln.mean(dim=(1, 2, 3))}")
    print(f"  Std per sample:  {x_ln.std(dim=(1, 2, 3))}")
    
    print("\nInstance Norm:")
    print(f"  Normalizes over: (H, W) for each N and C")
    for n in range(2):
        print(f"  Sample {n}:")
        for c in range(3):
            mean = x_in[n, c].mean()
            std = x_in[n, c].std()
            print(f"    Channel {c}: mean={mean:.4f}, std={std:.4f}")
    
    print("\n" + "=" * 60)
    print("Summary of Normalization Methods")
    print("=" * 60)
    
    comparison = """
    Method          | Normalizes Over | Use Case
    ----------------|-----------------|----------------------------------
    Batch Norm      | N, H, W         | CNNs, large batches
    Layer Norm      | C, H, W         | RNNs, Transformers, small batches
    Instance Norm   | H, W            | Style transfer, GANs
    Group Norm      | (H, W, C/G)     | Small batches, when BN fails
    """
    print(comparison)


def demonstrate_style_transfer_example():
    """
    Show why Instance Norm is important for style transfer.
    """
    print("\n" + "=" * 60)
    print("Why Instance Norm for Style Transfer?")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Simulate content image (bright)
    content = torch.randn(1, 3, 32, 32) + 2.0
    
    # Simulate style image (dark)
    style = torch.randn(1, 3, 32, 32) - 2.0
    
    print("\nOriginal statistics:")
    print(f"Content image mean: {content.mean():.4f}, std: {content.std():.4f}")
    print(f"Style image mean:   {style.mean():.4f}, std: {style.std():.4f}")
    
    # With Batch Normalization (mixes statistics across images)
    bn = nn.BatchNorm2d(3)
    bn.eval()
    combined_bn = torch.cat([content, style], dim=0)
    normalized_bn = bn(combined_bn)
    
    print("\nWith Batch Normalization (not ideal):")
    print(f"Normalized content mean: {normalized_bn[0].mean():.4f}")
    print(f"Normalized style mean:   {normalized_bn[1].mean():.4f}")
    print("→ Statistics are mixed across images!")
    
    # With Instance Normalization (independent)
    instance_norm = nn.InstanceNorm2d(3, affine=False)
    content_in = instance_norm(content)
    style_in = instance_norm(style)
    
    print("\nWith Instance Normalization (ideal):")
    print(f"Normalized content mean: {content_in.mean():.4f}")
    print(f"Normalized style mean:   {style_in.mean():.4f}")
    print("→ Each image normalized independently!")
    
    print("\nKey insight:")
    print("Instance Norm removes instance-specific contrast information,")
    print("allowing the network to focus on transferring style features")
    print("without being influenced by the original image's brightness/contrast.")


if __name__ == "__main__":
    demonstrate_instance_norm()
    compare_all_normalizations()
    demonstrate_style_transfer_example()
    
    print("\n" + "=" * 60)
    print("When to use Instance Normalization:")
    print("=" * 60)
    print("✓ Style transfer networks")
    print("✓ GANs (especially image-to-image translation)")
    print("✓ When each sample should be processed independently")
    print("✓ When batch statistics shouldn't mix")
    print("✓ Real-time applications (no running statistics needed)")
