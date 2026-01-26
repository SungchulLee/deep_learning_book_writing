# ResNet Architecture

## Overview

ResNet (Residual Network) is a family of convolutional neural network architectures built upon residual connections. Introduced by He et al. in 2015, ResNet won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and has since become one of the most influential and widely-used architectures in computer vision.

The key insight of ResNet is that by using skip connections, networks can be trained to unprecedented depths (50, 101, 152, and even 1000+ layers) while maintaining or improving performance.

## ResNet Family Overview

| Model | Block Type | Layers | Parameters | Top-1 Accuracy (ImageNet) |
|-------|------------|--------|------------|---------------------------|
| ResNet-18 | Basic | 18 | ~11.7M | 69.8% |
| ResNet-34 | Basic | 34 | ~21.8M | 73.3% |
| ResNet-50 | Bottleneck | 50 | ~25.6M | 76.1% |
| ResNet-101 | Bottleneck | 101 | ~44.5M | 77.4% |
| ResNet-152 | Bottleneck | 152 | ~60.2M | 78.3% |

## Building Blocks

### Basic Block (ResNet-18, ResNet-34)

The Basic Block consists of two 3×3 convolutional layers with a skip connection:

```
Input ──┬─────────────────────────────────┐
        │                                  │
        ▼                                  │
   [Conv 3×3, stride]                      │
        ▼                                  │
   [BatchNorm]                             │
        ▼                                  │
     [ReLU]                                │ (identity or projection)
        ▼                                  │
   [Conv 3×3, stride=1]                    │
        ▼                                  │
   [BatchNorm]                             │
        ▼                                  │
      (+)  ◄───────────────────────────────┘
        ▼
     [ReLU]
        ▼
     Output
```

**Parameter count per Basic Block:**
- Two 3×3 convolutions: $2 \times (C_{in} \times C_{out} \times 9)$
- With projection: add $C_{in} \times C_{out} \times 1$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Type


class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18 and ResNet-34.
    
    Two 3×3 convolutions with skip connection.
    Expansion factor = 1 (output channels = input channels to block).
    """
    
    expansion: int = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        # First 3×3 convolution (may downsample)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        
        # Second 3×3 convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out
```

### Bottleneck Block (ResNet-50, ResNet-101, ResNet-152)

The Bottleneck Block uses three convolutions: 1×1 to reduce dimensions, 3×3 for processing, and 1×1 to restore dimensions. This "bottleneck" design reduces computation while maintaining representational power.

```
Input ──┬──────────────────────────────────────┐
        │                                       │
        ▼                                       │
   [Conv 1×1, reduce]                           │
        ▼                                       │
   [BatchNorm]                                  │
        ▼                                       │
     [ReLU]                                     │ (identity or projection)
        ▼                                       │
   [Conv 3×3, stride]                           │
        ▼                                       │
   [BatchNorm]                                  │
        ▼                                       │
     [ReLU]                                     │
        ▼                                       │
   [Conv 1×1, expand]                           │
        ▼                                       │
   [BatchNorm]                                  │
        ▼                                       │
      (+)  ◄────────────────────────────────────┘
        ▼
     [ReLU]
        ▼
     Output
```

The expansion factor is 4, meaning the output has 4× the channels of the bottleneck width.

```python
class Bottleneck(nn.Module):
    """
    Bottleneck Block for ResNet-50, ResNet-101, and ResNet-152.
    
    Three convolutions: 1×1 (reduce) → 3×3 (process) → 1×1 (expand)
    Expansion factor = 4 (output channels = 4 × bottleneck width).
    """
    
    expansion: int = 4
    
    def __init__(
        self,
        in_channels: int,
        width: int,  # Bottleneck width (1/4 of output channels)
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Calculate actual width for grouped convolutions (ResNeXt compatibility)
        actual_width = int(width * (base_width / 64.0)) * groups
        
        # 1×1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(
            in_channels, actual_width,
            kernel_size=1, bias=False
        )
        self.bn1 = norm_layer(actual_width)
        
        # 3×3 conv for main processing
        self.conv2 = nn.Conv2d(
            actual_width, actual_width,
            kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=False
        )
        self.bn2 = norm_layer(actual_width)
        
        # 1×1 conv to expand dimensions
        self.conv3 = nn.Conv2d(
            actual_width, width * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(width * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Process
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        # Expand
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out
```

## Complete ResNet Architecture

The full ResNet architecture consists of:

1. **Initial convolution and pooling** (stem)
2. **Four stages** of residual blocks
3. **Global average pooling** and **fully connected** classifier

```
Input Image (3 × 224 × 224)
         │
         ▼
   [Conv 7×7, 64, stride=2]  ──► (64 × 112 × 112)
         │
         ▼
   [BatchNorm + ReLU]
         │
         ▼
   [MaxPool 3×3, stride=2]   ──► (64 × 56 × 56)
         │
         ▼
   ┌─────────────────────────────────────────┐
   │  Stage 1: conv2_x                       │
   │  [Block] × n₁                           │  ──► (64/256 × 56 × 56)
   └─────────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────────┐
   │  Stage 2: conv3_x (stride=2 first)      │
   │  [Block] × n₂                           │  ──► (128/512 × 28 × 28)
   └─────────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────────┐
   │  Stage 3: conv4_x (stride=2 first)      │
   │  [Block] × n₃                           │  ──► (256/1024 × 14 × 14)
   └─────────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────────┐
   │  Stage 4: conv5_x (stride=2 first)      │
   │  [Block] × n₄                           │  ──► (512/2048 × 7 × 7)
   └─────────────────────────────────────────┘
         │
         ▼
   [Global Average Pooling]   ──► (512/2048 × 1 × 1)
         │
         ▼
   [Fully Connected]          ──► (num_classes)
```

### Stage Configuration

| Model | n₁, n₂, n₃, n₄ | Block Type |
|-------|----------------|------------|
| ResNet-18 | 2, 2, 2, 2 | Basic |
| ResNet-34 | 3, 4, 6, 3 | Basic |
| ResNet-50 | 3, 4, 6, 3 | Bottleneck |
| ResNet-101 | 3, 4, 23, 3 | Bottleneck |
| ResNet-152 | 3, 8, 36, 3 | Bottleneck |

### Complete Implementation

```python
from typing import List, Union


class ResNet(nn.Module):
    """
    Complete ResNet implementation supporting all standard variants.
    
    Args:
        block: Block type (BasicBlock or Bottleneck)
        layers: Number of blocks in each of the 4 stages
        num_classes: Number of output classes (default: 1000 for ImageNet)
        in_channels: Number of input channels (default: 3 for RGB)
        groups: Groups for grouped convolutions (default: 1)
        width_per_group: Base width for bottleneck (default: 64)
        norm_layer: Normalization layer (default: BatchNorm2d)
    """
    
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_group
        
        # Stem: Initial convolution and pooling
        self.conv1 = nn.Conv2d(
            in_channels, self.in_planes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four stages of residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a stage of residual blocks.
        
        Args:
            block: Block type
            planes: Base channel count for this stage
            num_blocks: Number of blocks in this stage
            stride: Stride for the first block (downsampling)
        """
        norm_layer = self._norm_layer
        downsample = None
        
        # Create downsample layer if needed
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(planes * block.expansion)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(block(
            self.in_planes, planes, stride, downsample,
            self.groups, self.base_width, norm_layer
        ))
        
        # Update in_planes for subsequent blocks
        self.in_planes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_planes, planes,
                groups=self.groups, base_width=self.base_width,
                norm_layer=norm_layer
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization.
        
        For residual networks, Kaiming initialization is crucial
        for proper convergence of deep networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Four stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classification head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


# Factory functions for standard ResNet variants
def resnet18(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-18: 18 layers with Basic blocks [2, 2, 2, 2]"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def resnet34(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-34: 34 layers with Basic blocks [3, 4, 6, 3]"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet50(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-50: 50 layers with Bottleneck blocks [3, 4, 6, 3]"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def resnet101(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-101: 101 layers with Bottleneck blocks [3, 4, 23, 3]"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def resnet152(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-152: 152 layers with Bottleneck blocks [3, 8, 36, 3]"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
```

## Layer Counting

Understanding how layers are counted in ResNet:

### ResNet-18 Layer Count
- Conv1 (stem): 1 layer
- Stage 1: 2 blocks × 2 convs = 4 layers
- Stage 2: 2 blocks × 2 convs = 4 layers
- Stage 3: 2 blocks × 2 convs = 4 layers
- Stage 4: 2 blocks × 2 convs = 4 layers
- FC: 1 layer
- **Total: 1 + 4 + 4 + 4 + 4 + 1 = 18 layers**

### ResNet-50 Layer Count
- Conv1 (stem): 1 layer
- Stage 1: 3 blocks × 3 convs = 9 layers
- Stage 2: 4 blocks × 3 convs = 12 layers
- Stage 3: 6 blocks × 3 convs = 18 layers
- Stage 4: 3 blocks × 3 convs = 9 layers
- FC: 1 layer
- **Total: 1 + 9 + 12 + 18 + 9 + 1 = 50 layers**

## Computational Complexity Analysis

### FLOPs Calculation

For a convolutional layer with kernel $K \times K$, input channels $C_{in}$, output channels $C_{out}$, and spatial size $H \times W$:

$$\text{FLOPs} = 2 \times K^2 \times C_{in} \times C_{out} \times H \times W$$

### Basic Block FLOPs

For a Basic Block with input size $(C, H, W)$:

$$\text{FLOPs}_{\text{Basic}} = 2 \times (9C^2HW + 9C^2HW) = 36C^2HW$$

### Bottleneck Block FLOPs

For a Bottleneck Block reducing to width $W$ then expanding to $4W$:

$$\text{FLOPs}_{\text{Bottleneck}} = 2 \times (C \cdot W \cdot HW + 9W^2 \cdot HW + W \cdot 4W \cdot HW)$$

When $W = C/4$, this is approximately $17C^2HW/4 \approx 4.25C^2HW$, which is much less than a Basic Block despite having more layers.

## ResNet for Different Input Sizes

### CIFAR-10/100 (32×32 images)

For small images, the standard ResNet architecture is modified:

```python
def resnet_cifar(num_classes: int = 10, depth: int = 20) -> nn.Module:
    """
    ResNet variant for CIFAR-10/100 with 32×32 images.
    
    Modifications from ImageNet ResNet:
    - Initial conv: 3×3 instead of 7×7, no stride
    - No initial max pooling
    - Three stages instead of four
    - Smaller channel counts
    """
    assert (depth - 2) % 6 == 0, "Depth must be 6n+2 for CIFAR ResNet"
    n = (depth - 2) // 6
    
    class CIFARResNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.in_planes = 16
            
            # Smaller initial convolution
            self.conv1 = nn.Conv2d(
                3, 16, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(16)
            
            # Three stages (not four)
            self.layer1 = self._make_layer(16, n, stride=1)   # 32×32
            self.layer2 = self._make_layer(32, n, stride=2)   # 16×16
            self.layer3 = self._make_layer(64, n, stride=2)   # 8×8
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, num_classes)
        
        def _make_layer(self, planes, num_blocks, stride):
            # Similar to standard ResNet
            ...
    
    return CIFARResNet()
```

## Training Best Practices

### Optimizer and Learning Rate

```python
# Standard ImageNet training setup
model = resnet50(num_classes=1000)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# Step decay scheduler (divide by 10 at epochs 30, 60, 90)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)

# Or cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)
```

### Data Augmentation

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
```

## Feature Extraction and Transfer Learning

ResNet is commonly used as a feature extractor for downstream tasks:

```python
def create_feature_extractor(pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet feature extractor with frozen weights.
    """
    model = resnet50(num_classes=1000)
    
    if pretrained:
        # Load pretrained weights
        weights = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        )
        model.load_state_dict(weights)
    
    # Remove classification head
    model.fc = nn.Identity()
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def create_fine_tuned_classifier(
    num_classes: int, 
    freeze_layers: int = 7
) -> nn.Module:
    """
    Create a fine-tuned classifier with selective unfreezing.
    """
    model = resnet50(num_classes=1000)
    weights = torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    )
    model.load_state_dict(weights)
    
    # Replace classifier
    model.fc = nn.Linear(2048, num_classes)
    
    # Freeze early layers
    layers = [model.conv1, model.bn1, model.layer1, model.layer2,
              model.layer3, model.layer4, model.fc]
    
    for layer in layers[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False
    
    return model
```

## Summary

ResNet's key architectural contributions include:

1. **Basic Block**: Two 3×3 convolutions with skip connection (ResNet-18/34)
2. **Bottleneck Block**: 1×1-3×3-1×1 with 4× expansion (ResNet-50/101/152)
3. **Staged architecture**: Four stages with progressive downsampling
4. **Kaiming initialization**: Crucial for training very deep networks
5. **Batch normalization**: Applied after every convolution

These design choices enable training networks with hundreds of layers that achieve state-of-the-art performance across many computer vision tasks.

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV 2015.
3. Goyal, P., et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677.
