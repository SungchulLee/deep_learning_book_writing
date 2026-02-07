#!/usr/bin/env python3
"""
================================================================================
ResNet - Deep Residual Learning for Image Recognition
================================================================================

Paper: "Deep Residual Learning for Image Recognition" (2015)
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
Link: https://arxiv.org/abs/1512.03385

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
ResNet revolutionized deep learning by solving the **degradation problem**: 
as networks get deeper, accuracy gets saturated and then degrades rapidly.
This was NOT due to overfitting, but due to optimization difficulties.

- **ILSVRC 2015 Winner**: 3.57% top-5 error (first superhuman performance!)
- **152 layers**: Previously, >20 layers caused degradation
- **Most Cited Paper**: One of the most influential deep learning papers
- **Foundation for Modern Architectures**: Nearly all modern CNNs use residual connections

================================================================================
THE KEY INSIGHT: RESIDUAL LEARNING
================================================================================

The Problem:
────────────────────────────────────────────────────────────────────────────────
Plain networks: As depth increases, training error INCREASES (not decreases!)
This is counterintuitive - a deeper network should be at least as good as a 
shallower one (it could just learn identity mappings for extra layers).

The Solution: Learn RESIDUALS instead of direct mappings
────────────────────────────────────────────────────────────────────────────────

Traditional Learning:              Residual Learning:
    ┌───────────┐                     ┌───────────┐
    │   Input   │                     │   Input   │───────────┐
    │     x     │                     │     x     │           │
    └─────┬─────┘                     └─────┬─────┘           │
          │                                 │                 │
          ▼                                 ▼                 │
    ┌───────────┐                     ┌───────────┐           │
    │  Layers   │                     │  Layers   │           │
    │  H(x)     │                     │  F(x)     │           │
    └─────┬─────┘                     └─────┬─────┘           │
          │                                 │                 │
          ▼                                 ▼                 │
    ┌───────────┐                     ┌───────────┐           │
    │  Output   │                     │    (+)    │◄──────────┘
    │   H(x)    │                     │  F(x) + x │    (skip connection)
    └───────────┘                     └───────────┘

Learn: H(x) = F(x) + x, where F(x) is the RESIDUAL

Why This Works:
1. If identity is optimal, F(x)→0 is easier than H(x)→x
2. Gradients flow directly through skip connections (no vanishing!)
3. Each block learns "refinements" to the input, not full transformations

Mathematical Justification (Gradient Flow):
────────────────────────────────────────────────────────────────────────────────
For layer output: y = F(x) + x

Gradient during backprop:
    ∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
           = ∂L/∂y · ∂F/∂x + ∂L/∂y

The "+1" term ensures gradient always flows through, even if ∂F/∂x ≈ 0!
This prevents vanishing gradients in very deep networks.

================================================================================
RESIDUAL BLOCK VARIANTS
================================================================================

1. BASIC BLOCK (ResNet-18, ResNet-34)
   ────────────────────────────────────
   Used for shallower ResNets (≤34 layers)
   
       ┌─────────────────────────────────┐
       │         Input (C channels)      │──────────┐
       └──────────────┬──────────────────┘          │
                      │                              │
                      ▼                              │
       ┌─────────────────────────────────┐          │
       │    3×3 Conv, C filters, BN      │          │
       └──────────────┬──────────────────┘          │
                      │                              │
                      ▼                              │
                    ReLU                             │
                      │                              │
                      ▼                              │
       ┌─────────────────────────────────┐          │
       │    3×3 Conv, C filters, BN      │          │
       └──────────────┬──────────────────┘          │
                      │                              │
                      ▼                              │
       ┌─────────────────────────────────┐          │
       │            Addition             │◄─────────┘
       └──────────────┬──────────────────┘
                      │
                      ▼
                    ReLU
                      │
                      ▼
                   Output

2. BOTTLENECK BLOCK (ResNet-50, ResNet-101, ResNet-152)
   ─────────────────────────────────────────────────────
   Uses 1×1 convolutions to reduce/restore dimensions
   More efficient for deeper networks
   
       ┌─────────────────────────────────┐
       │      Input (4C channels)        │──────────┐
       └──────────────┬──────────────────┘          │
                      │                              │
                      ▼                              │
       ┌─────────────────────────────────┐          │
       │    1×1 Conv, C filters, BN      │ REDUCE   │
       │    (reduce dimensions)          │          │
       └──────────────┬──────────────────┘          │
                      │                              │
                      ▼                              │
                    ReLU                             │
                      │                              │
                      ▼                              │
       ┌─────────────────────────────────┐          │
       │    3×3 Conv, C filters, BN      │ PROCESS  │
       └──────────────┬──────────────────┘          │ (1×1 conv if
                      │                              │  dims differ)
                      ▼                              │
                    ReLU                             │
                      │                              │
                      ▼                              │
       ┌─────────────────────────────────┐          │
       │    1×1 Conv, 4C filters, BN     │ EXPAND   │
       │    (restore dimensions)         │          │
       └──────────────┬──────────────────┘          │
                      │                              │
                      ▼                              │
       ┌─────────────────────────────────┐          │
       │            Addition             │◄─────────┘
       └──────────────┬──────────────────┘
                      │
                      ▼
                    ReLU

   Expansion Factor = 4: Output has 4× channels of internal processing

================================================================================
ARCHITECTURE CONFIGURATIONS
================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│ Model      │ Layers │ Block Type  │ Blocks per Stage │ Parameters │ FLOPs   │
├──────────────────────────────────────────────────────────────────────────────┤
│ ResNet-18  │   18   │ Basic       │ [2, 2, 2, 2]     │    11.7M   │  1.8B   │
│ ResNet-34  │   34   │ Basic       │ [3, 4, 6, 3]     │    21.8M   │  3.6B   │
│ ResNet-50  │   50   │ Bottleneck  │ [3, 4, 6, 3]     │    25.6M   │  4.1B   │
│ ResNet-101 │  101   │ Bottleneck  │ [3, 4, 23, 3]    │    44.5M   │  7.8B   │
│ ResNet-152 │  152   │ Bottleneck  │ [3, 8, 36, 3]    │    60.2M   │ 11.5B   │
└──────────────────────────────────────────────────────────────────────────────┘

Stage-by-Stage Feature Map Sizes (ResNet-50):
─────────────────────────────────────────────
Input:   224×224×3
Conv1:   112×112×64    (7×7 conv, stride 2)
Pool:    56×56×64      (3×3 maxpool, stride 2)
Stage1:  56×56×256     (3 bottleneck blocks)
Stage2:  28×28×512     (4 bottleneck blocks, first with stride 2)
Stage3:  14×14×1024    (6 bottleneck blocks, first with stride 2)
Stage4:  7×7×2048      (3 bottleneck blocks, first with stride 2)
AvgPool: 1×1×2048      (global average pooling)
FC:      1000          (fully connected layer)

================================================================================
MATHEMATICAL FOUNDATIONS
================================================================================

**Residual Function:**
    y = F(x, {W_i}) + x
    
Where F is the residual mapping to be learned, {W_i} are layer weights.

**With Projection (when dimensions differ):**
    y = F(x, {W_i}) + W_s · x
    
Where W_s is a linear projection matrix (implemented as 1×1 conv)

**Batch Normalization:**
    BN(x) = γ · (x - μ_B) / √(σ²_B + ε) + β
    
Applied after each convolution, before activation:
- μ_B, σ²_B: batch mean and variance
- γ, β: learnable scale and shift
- ε: small constant for numerical stability

**Kaiming Initialization (critical for ResNets):**
    Var(w) = 2 / n_in
    
Where n_in is the fan-in (number of input connections).
Accounts for ReLU's effect on variance (halves it due to zeroing negatives).

================================================================================
TRAINING DETAILS (Original Paper)
================================================================================

Optimizer: SGD with momentum
- Momentum: 0.9
- Weight decay: 1×10⁻⁴
- Batch size: 256

Learning Rate Schedule:
- Initial: 0.1
- Divided by 10 at epochs 30, 60, 90
- Total: 90 epochs

Data Augmentation:
- Random 224×224 crops from resized images (256×480)
- Random horizontal flips
- Per-pixel mean subtraction
- Standard color augmentation

Batch Normalization:
- Applied after every convolution
- No dropout used (BN provides sufficient regularization)

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: Residual Connections (gradient flow analysis)
- Ch04: Image Classification (modern architectures)
- Ch08: Transfer Learning (ResNet as feature extractor)
- Ch21: Model Compression (pruning residual connections)
- Ch29: Interpretability (gradient-based attribution)

Related architectures:
- VGGNet: Predecessor without residuals (vgg_net.py)
- DenseNet: Dense connections (densenet.py)
- ResNeXt: Grouped convolutions (res_next.py)

================================================================================
"""

import torch
import torch.nn as nn
from typing import Type, Union, List, Optional, Tuple


class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18 and ResNet-34
    
    Structure: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+skip) → ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution (for downsampling)
        downsample: Module for matching dimensions in skip connection
    
    Note:
        - expansion = 1 (output channels = out_channels)
        - Used for shallower networks (≤34 layers)
    """
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(BasicBlock, self).__init__()
        
        # ====================================================================
        # First Convolution: 3×3, may downsample if stride > 1
        # ====================================================================
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # ====================================================================
        # Second Convolution: 3×3, always stride 1
        # ====================================================================
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # ReLU activation (shared)
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample module for skip connection (if dimensions differ)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Output tensor (potentially with different spatial size if stride > 1)
        """
        # Store identity for skip connection
        identity = x
        
        # Main path: Conv → BN → ReLU → Conv → BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection: match dimensions if necessary
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual addition: F(x) + x
        out += identity
        
        # Final activation
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Residual Block for ResNet-50, ResNet-101, ResNet-152
    
    Structure: Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+skip) → ReLU
    
    The 1×1 convolutions reduce then restore dimensions:
    - First 1×1: Reduce channels (out_channels)
    - 3×3: Process at reduced dimension
    - Second 1×1: Expand to out_channels × expansion
    
    Args:
        in_channels: Number of input channels
        out_channels: Base number of channels (actual output = out_channels × 4)
        stride: Stride for 3×3 convolution (for downsampling)
        downsample: Module for matching dimensions in skip connection
    
    Note:
        - expansion = 4 (output channels = out_channels × 4)
        - More parameter-efficient than basic blocks for deep networks
    """
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(Bottleneck, self).__init__()
        
        # ====================================================================
        # First 1×1 Convolution: REDUCE dimensions
        # ====================================================================
        # e.g., 256 → 64 channels (4× reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # ====================================================================
        # Second 3×3 Convolution: PROCESS at reduced dimension
        # ====================================================================
        # This is where spatial downsampling occurs (if stride > 1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # ====================================================================
        # Third 1×1 Convolution: EXPAND dimensions
        # ====================================================================
        # e.g., 64 → 256 channels (4× expansion)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Shared ReLU
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample for skip connection
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with bottleneck residual connection
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Output tensor with C × expansion channels
        """
        identity = x
        
        # 1×1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3×3 process
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1×1 expand
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual addition
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet (Residual Network) Implementation
    
    Supports ResNet-18, 34, 50, 101, and 152 configurations.
    
    Args:
        block: Block type (BasicBlock or Bottleneck)
        layers: Number of blocks in each of 4 stages
        num_classes: Number of output classes. Default: 1000
        zero_init_residual: Zero-initialize last BN in each block. Default: False
    
    Example:
        >>> model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)  # ResNet-50
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([1, 1000])
    
    Shape:
        - Input: (N, 3, 224, 224)
        - Output: (N, num_classes)
    """
    
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False
    ):
        super(ResNet, self).__init__()
        
        # Track current channel count
        self.in_channels = 64
        
        # ====================================================================
        # STEM: Initial Conv + Pool
        # ====================================================================
        # 7×7 conv, stride 2: 224×224×3 → 112×112×64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 3×3 maxpool, stride 2: 112×112×64 → 56×56×64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ====================================================================
        # RESIDUAL STAGES
        # ====================================================================
        # Stage 1: 56×56×64 → 56×56×(64×expansion)
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # Stage 2: 56×56 → 28×28, channels: 64×exp → 128×exp
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # Stage 3: 28×28 → 14×14, channels: 128×exp → 256×exp
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # Stage 4: 14×14 → 7×7, channels: 256×exp → 512×exp
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # ====================================================================
        # CLASSIFIER HEAD
        # ====================================================================
        # Global Average Pooling: 7×7 → 1×1
        # Much more efficient than FC layers (used in VGG)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final FC layer
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # ====================================================================
        # WEIGHT INITIALIZATION
        # ====================================================================
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a residual stage with multiple blocks
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            out_channels: Base output channels for blocks
            num_blocks: Number of blocks in this stage
            stride: Stride for first block (for downsampling)
            
        Returns:
            Sequential container of blocks
        """
        downsample = None
        
        # Need projection if dimensions change:
        # 1. Spatial dimensions change (stride > 1)
        # 2. Channel dimensions change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        
        # First block may downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # Update channel count
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool):
        """
        Initialize network weights
        
        Uses Kaiming initialization for conv layers, designed for ReLU.
        Optionally zero-initializes the last BN in each residual block.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        # This makes each residual block behave like identity at initialization
        # Improves training stability for very deep networks
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet
        
        Args:
            x: Input tensor of shape (N, 3, 224, 224)
            
        Returns:
            Output logits of shape (N, num_classes)
        """
        # Stem
        x = self.conv1(x)      # (N, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (N, 64, 56, 56)
        
        # Residual stages
        x = self.layer1(x)     # (N, 256, 56, 56) for ResNet-50
        x = self.layer2(x)     # (N, 512, 28, 28)
        x = self.layer3(x)     # (N, 1024, 14, 14)
        x = self.layer4(x)     # (N, 2048, 7, 7)
        
        # Classifier
        x = self.avgpool(x)    # (N, 2048, 1, 1)
        x = torch.flatten(x, 1) # (N, 2048)
        x = self.fc(x)         # (N, num_classes)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier (for transfer learning)
        
        Args:
            x: Input tensor of shape (N, 3, 224, 224)
            
        Returns:
            Feature tensor of shape (N, 512 * expansion)
        """
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


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ResNet18(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-18: 18 layers, ~11.7M parameters"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def ResNet34(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-34: 34 layers, ~21.8M parameters"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ResNet50(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-50: 50 layers, ~25.6M parameters (most commonly used)"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ResNet101(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-101: 101 layers, ~44.5M parameters"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def ResNet152(num_classes: int = 1000, **kwargs) -> ResNet:
    """ResNet-152: 152 layers, ~60.2M parameters"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


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
    print("ResNet Family - Parameter Comparison")
    print("=" * 70)
    
    configs = [
        ('ResNet-18', ResNet18),
        ('ResNet-34', ResNet34),
        ('ResNet-50', ResNet50),
        ('ResNet-101', ResNet101),
        ('ResNet-152', ResNet152),
    ]
    
    for name, model_fn in configs:
        model = model_fn(num_classes=1000)
        total, _ = count_parameters(model)
        print(f"{name}: {total:>15,} parameters")
    
    print("=" * 70)
    print("\nResNet-50 Detailed Test")
    print("=" * 70)
    
    model = ResNet50(num_classes=1000)
    total_params, trainable_params = count_parameters(model)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB):      {total_params * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        features = model.get_features(x)
    
    print(f"Output shape: {logits.shape}")
    print(f"Feature shape: {features.shape}")
    print("=" * 70)
