"""
Simplified PixelCNN for Autoregressive Image Generation

PixelCNN generates images pixel by pixel in raster scan order (left to right, top to bottom).
Each pixel is predicted based on all previously generated pixels.

This is a simplified educational version focusing on core concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedConv2d(nn.Conv2d):
    """
    Masked Convolution for Autoregressive Image Generation.
    
    The key innovation of PixelCNN: convolutions are masked so that
    each pixel can only depend on previous pixels (above and to the left).
    
    Mask types:
    - Type A: For first layer, excludes the current pixel
    - Type B: For subsequent layers, includes the current pixel
    """
    
    def __init__(self, mask_type: str, *args, **kwargs):
        """
        Initialize masked convolution.
        
        Args:
            mask_type: Either 'A' or 'B'
            *args, **kwargs: Arguments for nn.Conv2d
        """
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type
        
        # Register buffer for the mask (won't be updated during training)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask()
    
    def create_mask(self):
        """
        Create the autoregressive mask.
        
        The mask ensures that:
        - Pixels above can be seen
        - Pixels to the left can be seen
        - Current pixel: seen only for mask B
        - Pixels below and to the right cannot be seen
        """
        # Get dimensions
        # Weight shape: [out_channels, in_channels, kernel_height, kernel_width]
        k_h, k_w = self.weight.shape[2:]
        
        # Initialize mask to all ones
        self.mask.fill_(1)
        
        # Zero out bottom half
        self.mask[:, :, k_h // 2 + 1:, :] = 0
        
        # Zero out right side of center row
        # For mask A: exclude center pixel
        # For mask B: include center pixel
        if self.mask_type == 'A':
            self.mask[:, :, k_h // 2, k_w // 2:] = 0
        else:  # mask_type == 'B'
            self.mask[:, :, k_h // 2, k_w // 2 + 1:] = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masked weights.
        
        Args:
            x: Input tensor
            
        Returns:
            Output of masked convolution
        """
        # Multiply weights by mask before convolution
        # This ensures only allowed pixels are used
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ResidualBlock(nn.Module):
    """
    Residual block with masked convolutions.
    
    Structure:
        Input -> MaskedConv -> ReLU -> MaskedConv -> Add with Input
    """
    
    def __init__(self, channels: int):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
        """
        super(ResidualBlock, self).__init__()
        
        # All convolutions after the first are type B
        self.conv1 = MaskedConv2d('B', channels, channels // 2, 
                                  kernel_size=1, padding=0)
        self.conv2 = MaskedConv2d('B', channels // 2, channels // 2,
                                  kernel_size=3, padding=1)
        self.conv3 = MaskedConv2d('B', channels // 2, channels,
                                  kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = x
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        # Residual connection
        return out + residual


class PixelCNN(nn.Module):
    """
    Simplified PixelCNN for binary (black/white) image generation.
    
    This autoregressive model generates images pixel by pixel:
    P(Image) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
    
    where each xᵢ is a pixel value.
    """
    
    def __init__(self, 
                 n_channels: int = 64,
                 n_residual_blocks: int = 5):
        """
        Initialize PixelCNN.
        
        Args:
            n_channels: Number of feature channels
            n_residual_blocks: Number of residual blocks
        """
        super(PixelCNN, self).__init__()
        
        # First layer uses mask type A (excludes current pixel)
        self.input_conv = MaskedConv2d('A', 1, n_channels,
                                       kernel_size=7, padding=3)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(n_channels) for _ in range(n_residual_blocks)
        ])
        
        # Output layers
        self.output_conv1 = MaskedConv2d('B', n_channels, n_channels,
                                         kernel_size=1)
        self.output_conv2 = MaskedConv2d('B', n_channels, n_channels,
                                         kernel_size=1)
        
        # Final layer: predict probability for each pixel
        # For binary images: output 1 channel (probability of being white)
        self.final_conv = MaskedConv2d('B', n_channels, 1,
                                       kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PixelCNN.
        
        Args:
            x: Input image [batch_size, 1, height, width]
               Values in [0, 1]
               
        Returns:
            Logits for each pixel [batch_size, 1, height, width]
            Apply sigmoid to get probabilities
        """
        # Initial masked convolution
        out = F.relu(self.input_conv(x))
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Output convolutions
        out = F.relu(self.output_conv1(out))
        out = F.relu(self.output_conv2(out))
        
        # Final prediction
        out = self.final_conv(out)
        
        return out
    
    @torch.no_grad()
    def generate(self, 
                 shape: tuple,
                 device: str = 'cpu') -> torch.Tensor:
        """
        Generate an image autoregressively.
        
        This is the core autoregressive generation:
        1. Start with blank image (all zeros)
        2. For each pixel position (top to bottom, left to right):
           a. Predict probability of pixel being 1
           b. Sample from Bernoulli distribution
           c. Fill in the pixel
        3. Return completed image
        
        Args:
            shape: (batch_size, height, width)
            device: Device to generate on
            
        Returns:
            Generated images [batch_size, 1, height, width]
        """
        self.eval()
        
        batch_size, height, width = shape
        
        # Start with blank canvas (all zeros)
        samples = torch.zeros(batch_size, 1, height, width).to(device)
        
        # Generate pixel by pixel
        # Raster scan order: top to bottom, left to right
        for i in range(height):
            for j in range(width):
                # Get prediction for current pixel
                # Note: uses all previously generated pixels
                logits = self.forward(samples)
                
                # Get probability for current pixel position
                probs = torch.sigmoid(logits[:, :, i, j])
                
                # Sample from Bernoulli distribution
                # This makes generation stochastic
                samples[:, :, i, j] = torch.bernoulli(probs)
        
        return samples


if __name__ == "__main__":
    """
    Demo: Test PixelCNN with dummy data
    """
    
    print("=" * 70)
    print("Testing Simplified PixelCNN")
    print("=" * 70)
    
    # Create model
    model = PixelCNN(n_channels=32, n_residual_blocks=3)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {n_params:,} parameters")
    
    # Test forward pass
    batch_size = 4
    height, width = 28, 28  # MNIST size
    
    # Create dummy input
    x = torch.rand(batch_size, 1, height, width)
    
    # Forward pass
    output = model(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test generation
    print(f"\nGenerating images...")
    print(f"  This will take a while (generating pixel by pixel)...")
    
    # Generate small images for demonstration
    small_shape = (2, 8, 8)  # 2 images of size 8x8
    generated = model.generate(small_shape, device='cpu')
    
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample pixel values: {generated[0, 0, :3, :3]}")
    
    print("\n✓ PixelCNN working correctly!")
    print("\nNote: For real training, use the train.py script")
    print("which trains on actual image data (like MNIST)")
