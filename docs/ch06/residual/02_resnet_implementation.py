"""
Complete ResNet Implementation
===============================
Full implementation of ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152
Based on: "Deep Residual Learning for Image Recognition" (He et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic Residual Block (used in ResNet-18 and ResNet-34)
    Two 3x3 convolutions with skip connection
    """
    expansion = 1  # Output channels = input channels * expansion
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Block (used in ResNet-50, ResNet-101, ResNet-152)
    Three convolutions: 1x1, 3x3, 1x1 (reduces then expands channels)
    More parameter efficient for deeper networks
    """
    expansion = 4  # Output channels = input channels * 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv (main computation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # Reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # Expand
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet Architecture
    
    Args:
        block: BasicBlock or Bottleneck
        layers: list of number of blocks in each layer
        num_classes: number of output classes
        in_channels: number of input channels (3 for RGB images)
    """
    
    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution (7x7 conv, stride 2)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a layer with multiple residual blocks
        """
        downsample = None
        
        # If dimensions change, create downsample layer
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # First block (might downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes=1000, in_channels=3):
    """ResNet-18: [2, 2, 2, 2] blocks"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes=1000, in_channels=3):
    """ResNet-34: [3, 4, 6, 3] blocks"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def resnet50(num_classes=1000, in_channels=3):
    """ResNet-50: [3, 4, 6, 3] bottleneck blocks"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def resnet101(num_classes=1000, in_channels=3):
    """ResNet-101: [3, 4, 23, 3] bottleneck blocks"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


def resnet152(num_classes=1000, in_channels=3):
    """ResNet-152: [3, 8, 36, 3] bottleneck blocks"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)


def model_summary(model, input_size=(3, 224, 224)):
    """
    Print model summary with layer information
    """
    print("=" * 80)
    print(f"Model Summary: {model.__class__.__name__}")
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, *input_size)
    
    print(f"\nInput shape: {x.shape}")
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ResNet Architecture Comparison")
    print("=" * 80)
    
    # Create different ResNet models
    models = {
        "ResNet-18": resnet18(num_classes=10),  # For CIFAR-10
        "ResNet-34": resnet34(num_classes=10),
        "ResNet-50": resnet50(num_classes=10),
        "ResNet-101": resnet101(num_classes=10),
    }
    
    print("\nModel Parameter Counts:")
    print("-" * 80)
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:15} {params:>15,} parameters")
    
    print("\n" + "=" * 80)
    print("Detailed Summary for ResNet-18")
    print("=" * 80)
    model_summary(models["ResNet-18"])
    
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    print("=" * 80)
    
    model = resnet18(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    
    print(f"Input: {x.shape}")
    output = model(x)
    print(f"Output: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    print("\n✓ ResNet implementation complete!")
    print("=" * 80 + "\n")
