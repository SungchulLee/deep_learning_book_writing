# DeepLab: Semantic Image Segmentation with Atrous Convolution

## Learning Objectives

By the end of this section, you will be able to:

- Understand atrous (dilated) convolutions and their role in expanding receptive fields
- Explain Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction
- Implement DeepLabv3 and DeepLabv3+ architectures
- Describe the encoder-decoder structure in DeepLabv3+
- Apply DeepLab to real-world segmentation tasks using pre-trained models

## The DeepLab Family Evolution

DeepLab represents one of the most influential lines of research in semantic segmentation:

| Version | Year | Key Innovation |
|---------|------|----------------|
| DeepLabv1 | 2015 | Atrous convolution + CRF post-processing |
| DeepLabv2 | 2017 | ASPP (Atrous Spatial Pyramid Pooling) |
| DeepLabv3 | 2017 | Improved ASPP + global image pooling |
| DeepLabv3+ | 2018 | Encoder-decoder architecture |

## Atrous (Dilated) Convolution

### The Receptive Field Problem

Standard convolutions face a fundamental trade-off: small kernels have limited receptive fields, while large kernels are computationally expensive. Pooling increases receptive field but loses spatial resolution.

### Atrous Convolution Solution

Atrous convolution inserts "holes" (zeros) between kernel weights, expanding the receptive field **without increasing parameters or reducing resolution**.

For a 1D signal, standard convolution:

$$y[i] = \sum_{k=1}^{K} x[i + k] \cdot w[k]$$

Atrous convolution with dilation rate $r$:

$$y[i] = \sum_{k=1}^{K} x[i + r \cdot k] \cdot w[k]$$

The effective kernel size becomes $k_e = k + (k-1)(r-1)$ while keeping $k^2$ parameters.

```python
import torch
import torch.nn as nn

# Standard 3×3 convolution
conv_standard = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1)

# Atrous convolution with dilation rate 2 (effective 5×5 receptive field)
conv_atrous_r2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)

# Atrous convolution with dilation rate 6 (effective 13×13 receptive field)
conv_atrous_r6 = nn.Conv2d(64, 64, kernel_size=3, padding=6, dilation=6)
```

## Atrous Spatial Pyramid Pooling (ASPP)

ASPP addresses multi-scale object segmentation by applying multiple atrous convolutions with different rates in **parallel**, then combining their outputs:

```python
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    """Single ASPP convolution branch: 3×3 atrous conv → BN → ReLU"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Module):
    """Image-level pooling branch capturing global context."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (typically 256)
        atrous_rates: Dilation rates for the three atrous conv branches
    """
    def __init__(self, in_channels: int, out_channels: int = 256,
                 atrous_rates: tuple = (6, 12, 18)):
        super().__init__()
        
        modules = []
        
        # 1×1 convolution branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Three atrous convolution branches
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global average pooling branch
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features (5 branches × 256 = 1280)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)
```

## DeepLabv3+ Architecture

DeepLabv3+ adds a simple decoder that uses low-level features from the encoder for sharper boundaries:

```python
import torchvision.models as models

class DeepLabv3Plus(nn.Module):
    """
    DeepLabv3+ with encoder-decoder structure.
    
    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone network ('resnet50', 'resnet101')
        output_stride: Output stride for backbone (16 recommended)
        pretrained_backbone: Use ImageNet pretrained weights
    """
    def __init__(self, num_classes: int = 21, backbone: str = 'resnet50',
                 output_stride: int = 16, pretrained_backbone: bool = True):
        super().__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained_backbone)
        else:
            resnet = models.resnet101(pretrained=pretrained_backbone)
        
        low_level_channels = 256
        high_level_channels = 2048
        
        # Low-level feature extraction (H/4 × W/4)
        self.low_level_features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1,
        )
        
        # High-level feature extraction
        self.high_level_features = nn.Sequential(
            resnet.layer2, resnet.layer3, resnet.layer4,
        )
        
        # ASPP
        atrous_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(high_level_channels, 256, atrous_rates)
        
        # Decoder: reduce low-level features to 48 channels
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions (256 from ASPP + 48 low-level = 304)
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        
        # Extract features
        low_level = self.low_level_features(x)
        high_level = self.high_level_features(low_level)
        aspp_out = self.aspp(high_level)
        
        # Upsample ASPP output to low-level resolution
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[2:],
                                 mode='bilinear', align_corners=False)
        
        # Reduce low-level channels and concatenate
        low_level = self.low_level_conv(low_level)
        concat = torch.cat([aspp_up, low_level], dim=1)
        
        # Decode and classify
        decoder_out = self.decoder(concat)
        logits = self.classifier(decoder_out)
        
        # Final upsampling
        return F.interpolate(logits, size=input_size,
                              mode='bilinear', align_corners=False)
```

## Using Pre-trained DeepLab

```python
from torchvision.models.segmentation import deeplabv3_resnet101

# Load pre-trained model (COCO, 21 classes)
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# For different number of classes
num_classes = 10
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

# Inference
with torch.no_grad():
    output = model(input_tensor)['out']
    prediction = output.argmax(dim=1)
```

## Performance on PASCAL VOC 2012

| Model | Backbone | mIoU (val) |
|-------|----------|------------|
| DeepLabv3 | ResNet-101 | 80.5% |
| DeepLabv3+ | ResNet-101 | 82.1% |
| DeepLabv3+ | Xception-65 | 83.4% |

## Summary

DeepLab introduced key innovations that remain foundational in semantic segmentation:

1. **Atrous convolutions**: Efficient receptive field expansion
2. **ASPP**: Multi-scale feature extraction
3. **Encoder-decoder (v3+)**: Sharp boundary recovery

DeepLabv3+ offers excellent accuracy/speed trade-offs and is widely used in production.

## References

1. Chen, L.-C., et al. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation.
2. Chen, L.-C., et al. (2018). Encoder-Decoder with Atrous Separable Convolution. ECCV.
