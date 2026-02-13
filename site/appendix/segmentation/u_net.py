#!/usr/bin/env python3
"""
================================================================================
U-Net - Convolutional Networks for Biomedical Image Segmentation
================================================================================

Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox (University of Freiburg)
Link: https://arxiv.org/abs/1505.04597

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
U-Net became the de facto standard for image segmentation, especially in 
medical imaging. Its elegant encoder-decoder architecture with skip connections
enables precise localization while maintaining contextual understanding.

- **ISBI 2015 Winner**: Cell segmentation challenge
- **Medical Imaging Standard**: Used in radiology, pathology, microscopy
- **Foundation for Modern Segmentation**: Influenced countless architectures
- **Diffusion Model Component**: U-Net is the backbone of Stable Diffusion!

================================================================================
KEY INNOVATIONS
================================================================================

1. **Symmetric Encoder-Decoder Architecture**
   ─────────────────────────────────────────────────────────────────────────────
   - Encoder: Progressively downsamples, capturing context
   - Decoder: Progressively upsamples, recovering spatial resolution
   - Creates characteristic "U" shape

2. **Skip Connections (Crucial!)**
   ─────────────────────────────────────────────────────────────────────────────
   Direct connections from encoder to decoder at each level:
   - Preserve fine-grained spatial information
   - Help gradient flow during training
   - Combine "what" (semantic) with "where" (spatial)

3. **Extensive Data Augmentation**
   ─────────────────────────────────────────────────────────────────────────────
   Original paper emphasized elastic deformations:
   - Critical for biomedical images with limited training data
   - Simulates natural tissue deformations

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

The U-Net has a distinctive U-shaped architecture:

Encoder (Contracting Path)          Decoder (Expanding Path)
════════════════════════════════════════════════════════════════════════════════
                                    
    Input Image (572×572×1)         Output Segmentation (388×388×n_classes)
           │                                    ▲
           ▼                                    │
    ┌─────────────────┐                  ┌─────────────────┐
    │ Conv 3×3 + ReLU │                  │ Conv 3×3 + ReLU │
    │ Conv 3×3 + ReLU │──────────────────│ Conv 3×3 + ReLU │
    │    (64 ch)      │   Skip Connect   │                 │
    └────────┬────────┘                  └────────▲────────┘
             │ MaxPool 2×2                        │ UpConv 2×2
             ▼                                    │
    ┌─────────────────┐                  ┌─────────────────┐
    │ Conv 3×3 + ReLU │                  │ Conv 3×3 + ReLU │
    │ Conv 3×3 + ReLU │──────────────────│ Conv 3×3 + ReLU │
    │   (128 ch)      │   Skip Connect   │                 │
    └────────┬────────┘                  └────────▲────────┘
             │ MaxPool 2×2                        │ UpConv 2×2
             ▼                                    │
    ┌─────────────────┐                  ┌─────────────────┐
    │ Conv 3×3 + ReLU │                  │ Conv 3×3 + ReLU │
    │ Conv 3×3 + ReLU │──────────────────│ Conv 3×3 + ReLU │
    │   (256 ch)      │   Skip Connect   │                 │
    └────────┬────────┘                  └────────▲────────┘
             │ MaxPool 2×2                        │ UpConv 2×2
             ▼                                    │
    ┌─────────────────┐                  ┌─────────────────┐
    │ Conv 3×3 + ReLU │                  │ Conv 3×3 + ReLU │
    │ Conv 3×3 + ReLU │──────────────────│ Conv 3×3 + ReLU │
    │   (512 ch)      │   Skip Connect   │                 │
    └────────┬────────┘                  └────────▲────────┘
             │ MaxPool 2×2                        │ UpConv 2×2
             ▼                                    │
    ┌─────────────────────────────────────────────────────┐
    │                    Bottleneck                        │
    │              Conv 3×3 + ReLU × 2                     │
    │                 (1024 ch)                            │
    └─────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════════════

Skip Connections (concatenation):
- Encoder features are concatenated with decoder features
- Doubles the channel count before convolutions
- Preserves fine-grained spatial information lost during downsampling

================================================================================
MATHEMATICAL FOUNDATIONS
================================================================================

**Convolution Operation:**
For 2D convolution maintaining spatial size (same padding):
    output_size = input_size
    padding = (kernel_size - 1) / 2 = (3 - 1) / 2 = 1

**Max Pooling (2×2, stride 2):**
    output_size = input_size / 2
    
**Transposed Convolution (Up-convolution):**
    output_size = (input_size - 1) × stride - 2 × padding + kernel_size
    For kernel=2, stride=2, padding=0:
    output_size = (input_size - 1) × 2 + 2 = 2 × input_size

**Skip Connection (Concatenation):**
    output_channels = encoder_channels + decoder_channels

**Loss Function (typically):**
Cross-entropy for multi-class segmentation:
    L = -Σ_i Σ_c y_{ic} log(p_{ic})
    
Dice Loss for class imbalance:
    L_dice = 1 - (2|X ∩ Y| + ε) / (|X| + |Y| + ε)

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch04: Semantic Segmentation (U-Net architecture)
- Ch14: Diffusion Models (U-Net as denoiser backbone)
- Ch04: Medical Image Segmentation (primary application)

Related architectures:
- FCN: Fully Convolutional Networks (fcn.py)
- DeepLab: Atrous convolutions (deeplabv3.py)
- PSPNet: Pyramid pooling (pspnet.py)

================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple


class DoubleConv(nn.Module):
    """
    Double Convolution Block (basic building block of U-Net)
    
    Structure: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU
    
    This block maintains spatial dimensions (uses padding=1 with kernel=3).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Number of channels between convs (default: out_channels)
    
    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None
    ):
        super(DoubleConv, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling Block: MaxPool → DoubleConv
    
    Reduces spatial dimensions by 2× and increases channels.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    
    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H/2, W/2)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling Block: UpConv → Concat (skip) → DoubleConv
    
    Increases spatial dimensions by 2× and reduces channels.
    Concatenates with corresponding encoder features (skip connection).
    
    Args:
        in_channels: Number of input channels (from decoder)
        out_channels: Number of output channels
        bilinear: Use bilinear upsampling instead of transposed conv. Default: True
    
    Shape:
        - Input: (N, in_channels, H, W) + skip: (N, in_channels//2, 2H, 2W)
        - Output: (N, out_channels, 2H, 2W)
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        # Upsampling method
        if bilinear:
            # Bilinear upsampling + 1×1 conv to reduce channels
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Transposed convolution (learnable upsampling)
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection
        
        Args:
            x1: Input from previous decoder layer (lower resolution)
            x2: Skip connection from encoder (higher resolution)
            
        Returns:
            Concatenated and convolved features
        """
        # Upsample x1 to match x2's spatial dimensions
        x1 = self.up(x1)
        
        # Handle potential size mismatch due to odd dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Pad x1 if necessary
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Convolution: 1×1 conv to map to desired number of classes
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (classes)
        bilinear: Use bilinear upsampling instead of transposed conv. Default: True
        base_channels: Number of channels in first layer. Default: 64
    
    Example:
        >>> model = UNet(in_channels=3, out_channels=2)  # Binary segmentation
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 2, 256, 256])
    
    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        bilinear: bool = True,
        base_channels: int = 64
    ):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Factor for reducing channels in decoder when using bilinear upsampling
        factor = 2 if bilinear else 1
        
        # ====================================================================
        # ENCODER (Contracting Path)
        # ====================================================================
        # Level 1: Input → 64 channels
        self.inc = DoubleConv(in_channels, base_channels)
        
        # Level 2: 64 → 128 channels, /2 spatial
        self.down1 = Down(base_channels, base_channels * 2)
        
        # Level 3: 128 → 256 channels, /4 spatial
        self.down2 = Down(base_channels * 2, base_channels * 4)
        
        # Level 4: 256 → 512 channels, /8 spatial
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        # Level 5 (Bottleneck): 512 → 1024/512 channels, /16 spatial
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # ====================================================================
        # DECODER (Expanding Path)
        # ====================================================================
        # Level 4: 1024 → 512 channels, ×2 spatial
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        
        # Level 3: 512 → 256 channels, ×4 spatial
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        
        # Level 2: 256 → 128 channels, ×8 spatial
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        
        # Level 1: 128 → 64 channels, ×16 spatial
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # ====================================================================
        # OUTPUT
        # ====================================================================
        self.outc = OutConv(base_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor (N, in_channels, H, W)
            
        Returns:
            Segmentation logits (N, out_channels, H, W)
        """
        # ====================================================================
        # ENCODER with skip connection storage
        # ====================================================================
        x1 = self.inc(x)      # (N, 64, H, W)
        x2 = self.down1(x1)   # (N, 128, H/2, W/2)
        x3 = self.down2(x2)   # (N, 256, H/4, W/4)
        x4 = self.down3(x3)   # (N, 512, H/8, W/8)
        x5 = self.down4(x4)   # (N, 1024, H/16, W/16) - Bottleneck
        
        # ====================================================================
        # DECODER with skip connections
        # ====================================================================
        x = self.up1(x5, x4)  # (N, 512, H/8, W/8)
        x = self.up2(x, x3)   # (N, 256, H/4, W/4)
        x = self.up3(x, x2)   # (N, 128, H/2, W/2)
        x = self.up4(x, x1)   # (N, 64, H, W)
        
        # ====================================================================
        # OUTPUT
        # ====================================================================
        logits = self.outc(x)  # (N, out_channels, H, W)
        
        return logits
    
    def get_encoder_features(self, x: torch.Tensor) -> dict:
        """
        Extract encoder features at each level (for visualization/analysis)
        
        Returns:
            Dictionary of features at each encoder level
        """
        features = {}
        
        x1 = self.inc(x)
        features['level1'] = x1
        
        x2 = self.down1(x1)
        features['level2'] = x2
        
        x3 = self.down2(x2)
        features['level3'] = x3
        
        x4 = self.down3(x3)
        features['level4'] = x4
        
        x5 = self.down4(x4)
        features['bottleneck'] = x5
        
        return features


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# DEMO AND TESTING
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("U-Net Model Summary")
    print("=" * 70)
    
    # Create model
    model = UNet(in_channels=3, out_channels=1, bilinear=True)
    total_params, trainable_params = count_parameters(model)
    
    print(f"Configuration:")
    print(f"  in_channels: 3 (RGB)")
    print(f"  out_channels: 1 (binary segmentation)")
    print(f"  bilinear upsampling: True")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Feature extraction demo
    print("\n" + "=" * 70)
    print("Encoder Feature Shapes")
    print("=" * 70)
    
    with torch.no_grad():
        features = model.get_encoder_features(x)
    
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print("=" * 70)
