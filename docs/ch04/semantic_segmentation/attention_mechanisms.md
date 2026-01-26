# Attention Mechanisms in Semantic Segmentation

## Learning Objectives

By the end of this section, you will be able to:

- Understand how attention mechanisms enhance segmentation networks
- Implement channel attention (SE blocks) and spatial attention modules
- Apply CBAM (Convolutional Block Attention Module) to U-Net architectures
- Use self-attention for capturing long-range dependencies
- Implement attention gates for skip connections

## Introduction

Attention mechanisms allow neural networks to focus on relevant features while suppressing irrelevant ones. In semantic segmentation, attention helps models:

1. **Focus on important channels** (feature types)
2. **Highlight relevant spatial locations** (where to look)
3. **Capture long-range dependencies** (relate distant pixels)
4. **Weight skip connections** (filter encoder features)

## Channel Attention: Squeeze-and-Excitation

Channel attention learns to emphasize informative feature channels.

```python
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    
    1. Squeeze: Global average pooling to get channel-wise statistics
    2. Excitation: FC layers to learn channel interdependencies
    3. Scale: Reweight original features by learned channel weights
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Squeeze: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        y = self.squeeze(x).view(b, c)
        
        # Excitation: (B, C) -> (B, C)
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale: element-wise multiplication
        return x * y.expand_as(x)
```

## Spatial Attention

Spatial attention learns to focus on important regions.

```python
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Uses channel-wise max and average pooling to compute spatial
    attention weights, highlighting important spatial locations.
    
    Args:
        kernel_size: Size of the convolution kernel (default: 7)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.sigmoid(self.conv(concat))  # (B, 1, H, W)
        
        return x * attention
```

## CBAM: Combining Channel and Spatial Attention

CBAM applies channel attention followed by spatial attention sequentially.

```python
class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Sequentially applies channel attention and spatial attention
    to refine feature representations.
    
    Args:
        channels: Number of input/output channels
        reduction: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention convolution
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x
```

## Attention Gate for Skip Connections

Attention gates filter skip connection features before concatenation with decoder features.

```python
class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.
    
    Filters encoder features based on decoder context, learning
    to focus on relevant spatial regions and suppress irrelevant ones.
    
    Args:
        gate_channels: Number of channels in gating signal (decoder)
        skip_channels: Number of channels in skip connection (encoder)
        inter_channels: Number of intermediate channels
    """
    def __init__(self, gate_channels: int, skip_channels: int, 
                 inter_channels: int = None):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from decoder (coarser resolution)
            skip: Skip connection from encoder (finer resolution)
        
        Returns:
            Attention-weighted skip connection features
        """
        # Upsample gate if needed
        if gate.shape[2:] != skip.shape[2:]:
            gate = nn.functional.interpolate(
                gate, size=skip.shape[2:], mode='bilinear', align_corners=True
            )
        
        # Compute attention
        g = self.W_g(gate)
        x = self.W_x(skip)
        attention = self.psi(self.relu(g + x))
        
        return skip * attention
```

## Attention U-Net Implementation

```python
class AttentionUNet(nn.Module):
    """
    U-Net with attention gates on skip connections.
    
    Attention gates help the decoder focus on relevant encoder features,
    improving segmentation accuracy especially for small objects.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1,
                 base_features: int = 64):
        super().__init__()
        
        features = [base_features * (2**i) for i in range(5)]
        
        # Encoder
        self.enc1 = self._double_conv(in_channels, features[0])
        self.enc2 = self._double_conv(features[0], features[1])
        self.enc3 = self._double_conv(features[1], features[2])
        self.enc4 = self._double_conv(features[2], features[3])
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[3], features[4])
        
        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(features[4], features[3], 2, stride=2)
        self.att4 = AttentionGate(features[3], features[3])
        self.dec4 = self._double_conv(features[4], features[3])
        
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.att3 = AttentionGate(features[2], features[2])
        self.dec3 = self._double_conv(features[3], features[2])
        
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.att2 = AttentionGate(features[1], features[1])
        self.dec2 = self._double_conv(features[2], features[1])
        
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.att1 = AttentionGate(features[0], features[0])
        self.dec1 = self._double_conv(features[1], features[0])
        
        self.out = nn.Conv2d(features[0], num_classes, 1)
    
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)  # Apply attention
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        return self.out(d1)
```

## Self-Attention for Long-Range Dependencies

Self-attention captures relationships between all spatial positions.

```python
class SelfAttention2D(nn.Module):
    """
    Self-attention module for capturing long-range dependencies.
    
    Computes attention between all spatial positions, allowing
    the model to relate distant regions of the feature map.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for key/query dimensions
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.channels = channels
        inter_channels = channels // reduction
        
        self.query = nn.Conv2d(channels, inter_channels, 1)
        self.key = nn.Conv2d(channels, inter_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        # Query, Key, Value projections
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        key = self.key(x).view(B, -1, H * W)  # (B, C', HW)
        value = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        
        # Attention: (B, HW, HW)
        attention = self.softmax(torch.bmm(query, key))
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        return self.gamma * out + x
```

## Summary

Attention mechanisms significantly enhance segmentation networks:

| Attention Type | Purpose | Best Use Case |
|---------------|---------|---------------|
| Channel (SE) | Feature selection | General enhancement |
| Spatial | Region focus | Object localization |
| CBAM | Both | Comprehensive |
| Attention Gate | Skip filtering | U-Net architectures |
| Self-Attention | Long-range | Large objects, context |

Attention U-Net typically improves IoU by 1-3% over standard U-Net, with attention gates being particularly effective for small object segmentation.

## References

1. Hu, J., et al. (2018). Squeeze-and-Excitation Networks. CVPR.
2. Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
3. Oktay, O., et al. (2018). Attention U-Net. MIDL.
4. Wang, X., et al. (2018). Non-local Neural Networks. CVPR.
