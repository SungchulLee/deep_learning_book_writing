#!/usr/bin/env python3
"""
================================================================================
VGGNet - Very Deep Convolutional Networks
================================================================================

Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014)
Authors: Karen Simonyan, Andrew Zisserman (Visual Geometry Group, Oxford)
Link: https://arxiv.org/abs/1409.1556

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
VGGNet demonstrated a crucial insight: **network depth matters**. By stacking
many small (3×3) convolution filters, VGGNet showed that going deeper leads to
better representations, establishing the foundation for modern deep learning.

- **ILSVRC 2014 Runner-up**: 7.3% top-5 error (winner GoogLeNet: 6.7%)
- **Most Cited Architecture**: Became the go-to backbone for many tasks
- **Transfer Learning Pioneer**: Pre-trained VGG features used everywhere

================================================================================
KEY DESIGN PRINCIPLES
================================================================================

1. **Small Filters, Deep Networks**
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Why 3×3 convolutions instead of larger filters?                        │
   │                                                                         │
   │ Receptive Field Equivalence:                                           │
   │ • Two 3×3 layers = one 5×5 receptive field                             │
   │ • Three 3×3 layers = one 7×7 receptive field                           │
   │                                                                         │
   │ Parameter Efficiency:                                                   │
   │ • 7×7×C×C = 49C² parameters  vs  3×(3×3×C×C) = 27C² parameters         │
   │ • ~45% fewer parameters for same receptive field!                       │
   │                                                                         │
   │ More Non-linearity:                                                     │
   │ • Three ReLUs instead of one = more discriminative power               │
   │ • Network can learn more complex functions                              │
   └─────────────────────────────────────────────────────────────────────────┘

2. **Homogeneous Architecture**
   - Only 3×3 conv and 2×2 max pooling throughout
   - Doubles channels after each pooling: 64→128→256→512→512
   - Simple, regular structure (easy to implement and understand)

3. **Deep Feature Hierarchy**
   - Early layers: edges, corners, colors
   - Middle layers: textures, patterns
   - Late layers: object parts, semantic concepts

================================================================================
ARCHITECTURE VARIANTS
================================================================================

┌───────────────────────────────────────────────────────────────────────────────┐
│ Variant │ Conv Layers │ FC Layers │ Total Layers │ Parameters │ Notes       │
├───────────────────────────────────────────────────────────────────────────────┤
│ VGG-11  │     8       │    3      │     11       │   ~133M    │ A (LRN)     │
│ VGG-11  │     8       │    3      │     11       │   ~133M    │ A-LRN       │
│ VGG-13  │    10       │    3      │     13       │   ~133M    │ B           │
│ VGG-16  │    13       │    3      │     16       │   ~138M    │ C/D ⭐       │
│ VGG-19  │    16       │    3      │     19       │   ~144M    │ E           │
└───────────────────────────────────────────────────────────────────────────────┘
⭐ VGG-16 (configuration D) is the most commonly used variant

================================================================================
VGG-16 DETAILED ARCHITECTURE
================================================================================

Block-by-Block Structure:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Block   │ Layers           │ Input Size    │ Output Size   │ Parameters   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input   │ -                │ 224×224×3     │ -             │ -            │
│ Block1  │ Conv64×2 + Pool  │ 224×224×3     │ 112×112×64    │ 38,720       │
│ Block2  │ Conv128×2 + Pool │ 112×112×64    │ 56×56×128     │ 221,440      │
│ Block3  │ Conv256×3 + Pool │ 56×56×128     │ 28×28×256     │ 1,475,328    │
│ Block4  │ Conv512×3 + Pool │ 28×28×256     │ 14×14×512     │ 5,899,776    │
│ Block5  │ Conv512×3 + Pool │ 14×14×512     │ 7×7×512       │ 7,079,424    │
│ FC1     │ 25088 → 4096     │ 7×7×512       │ 4096          │ 102,764,544  │
│ FC2     │ 4096 → 4096      │ 4096          │ 4096          │ 16,781,312   │
│ FC3     │ 4096 → 1000      │ 4096          │ 1000          │ 4,097,000    │
└─────────────────────────────────────────────────────────────────────────────┘
Total: ~138 million parameters

CRITICAL OBSERVATION:
- Conv layers: ~15M parameters (11% of total)
- FC layers: ~123M parameters (89% of total) ← Bottleneck for efficiency!
- Modern networks (ResNet, EfficientNet) use Global Average Pooling to avoid
  massive FC layers

================================================================================
MATHEMATICAL FOUNDATIONS
================================================================================

**Output Size Calculation for Convolution:**
    H_out = floor((H_in + 2×padding - kernel_size) / stride) + 1

For VGG's 3×3 conv with padding=1, stride=1:
    H_out = floor((H_in + 2×1 - 3) / 1) + 1 = H_in
    ∴ Spatial dimensions preserved

**Output Size for Max Pooling (2×2, stride 2):**
    H_out = H_in / 2
    ∴ Dimensions halved after each pool

**Receptive Field Growth:**
After each conv layer: RF_new = RF_old + (kernel_size - 1) × stride_product

For VGG-16:
- After Block1: RF = 1 + 2×2 = 5
- After Block2: RF = 5 + 2×4 = 13  (×2 due to pooling)
- After Block3: RF = 13 + 3×8 = 37
- After Block4: RF = 37 + 3×16 = 85
- After Block5: RF = 85 + 3×32 = 181

================================================================================
TRAINING DETAILS (Original Paper)
================================================================================

Optimizer: SGD with momentum
- Momentum: 0.9
- Weight decay: 5×10⁻⁴
- Batch size: 256

Learning Rate Schedule:
- Initial: 0.01
- Divided by 10 when validation accuracy stops improving
- Decreased 3 times total

Data Augmentation:
- Random 224×224 crops from rescaled images
- Random horizontal flips
- Random RGB color shift (PCA jittering)

Training Time: 2-3 weeks on 4 GPUs (NVIDIA Titan Black)

Initialization Strategy (Important!):
- First train VGG-11, then use its weights to initialize deeper networks
- This "pre-training" helps with convergence for very deep networks
- Modern techniques (BatchNorm, better optimizers) make this unnecessary

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: CNN fundamentals (convolution, pooling, receptive field)
- Ch04: Image Classification (classic architectures)
- Ch08: Transfer Learning (VGG as feature extractor)
- Ch21: Model Compression (VGG is often pruned/quantized)
- Ch29: Interpretability (Grad-CAM, feature visualization)

Related architectures:
- AlexNet: Shallower predecessor (alex_net.py)
- ResNet: Adds skip connections (resnet.py)
- DenseNet: Dense connections (densenet.py)

================================================================================
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union


# VGG architecture configurations
# Numbers represent conv filter counts, 'M' represents MaxPool
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """
    VGG Network Implementation (supports VGG11, VGG13, VGG16, VGG19)
    
    This implementation follows the original paper's architecture with optional
    Batch Normalization (a modern addition that significantly helps training).
    
    Args:
        config (str): Architecture variant ('VGG11', 'VGG13', 'VGG16', 'VGG19')
        num_classes (int): Number of output classes. Default: 1000
        batch_norm (bool): Whether to use Batch Normalization. Default: False
        init_weights (bool): Whether to initialize weights. Default: True
        dropout (float): Dropout probability for FC layers. Default: 0.5
    
    Example:
        >>> model = VGG('VGG16', num_classes=1000, batch_norm=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([1, 1000])
    
    Shape:
        - Input: (N, 3, 224, 224) where N is batch size
        - Output: (N, num_classes)
    """
    
    def __init__(
        self,
        config: str = 'VGG16',
        num_classes: int = 1000,
        batch_norm: bool = False,
        init_weights: bool = True,
        dropout: float = 0.5
    ):
        super(VGG, self).__init__()
        
        if config not in VGG_CONFIGS:
            raise ValueError(f"Unknown config: {config}. Choose from {list(VGG_CONFIGS.keys())}")
        
        # Build feature extraction layers from config
        self.features = self._make_layers(VGG_CONFIGS[config], batch_norm)
        
        # Adaptive pooling for input size flexibility
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # ====================================================================
        # CLASSIFIER (Fully Connected Layers)
        # ====================================================================
        # These layers contain ~89% of all parameters!
        # Modern architectures use Global Average Pooling to avoid this
        self.classifier = nn.Sequential(
            # FC1: 7×7×512 = 25088 → 4096
            # This single layer has ~103M parameters
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # FC2: 4096 → 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            # FC3: 4096 → num_classes
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()
    
    def _make_layers(
        self,
        config: List[Union[int, str]],
        batch_norm: bool
    ) -> nn.Sequential:
        """
        Build convolutional layers from configuration
        
        Args:
            config: List of filter counts and 'M' for MaxPool
            batch_norm: Whether to add BatchNorm after each conv
            
        Returns:
            nn.Sequential of conv/bn/relu/pool layers
        """
        layers = []
        in_channels = 3  # RGB input
        
        for v in config:
            if v == 'M':
                # MaxPool: halves spatial dimensions
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # Convolution block: Conv -> [BN] -> ReLU
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                
                if batch_norm:
                    layers.extend([
                        conv,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ])
                else:
                    layers.extend([
                        conv,
                        nn.ReLU(inplace=True)
                    ])
                
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VGG network
        
        Args:
            x: Input tensor of shape (N, 3, 224, 224)
            
        Returns:
            Output logits of shape (N, num_classes)
        """
        # Feature extraction
        # Input:  (N, 3, 224, 224)
        # Output: (N, 512, 7, 7) for VGG16
        x = self.features(x)
        
        # Adaptive pooling (handles slight size variations)
        x = self.avgpool(x)
        
        # Flatten: (N, 512, 7, 7) → (N, 25088)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract intermediate features (useful for transfer learning and visualization)
        
        Args:
            x: Input tensor of shape (N, 3, 224, 224)
            layer_idx: Index of layer to extract features from (-1 for final conv)
            
        Returns:
            Feature tensor
            
        Example:
            >>> features = model.get_features(x, layer_idx=28)  # After block4
        """
        if layer_idx == -1:
            return self.features(x)
        else:
            return self.features[:layer_idx](x)
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming (He) initialization
        
        Kaiming initialization is designed for ReLU activations:
        - Variance of weights = 2 / fan_in
        - Preserves variance through the network
        - Prevents vanishing/exploding activations
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal for conv layers
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',      # Preserve variance in backward pass
                    nonlinearity='relu'  # Account for ReLU
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: scale=1, shift=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Linear):
                # Normal initialization for FC layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def VGG11(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-11 model (8 conv layers)"""
    return VGG('VGG11', num_classes=num_classes, **kwargs)

def VGG13(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-13 model (10 conv layers)"""
    return VGG('VGG13', num_classes=num_classes, **kwargs)

def VGG16(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-16 model (13 conv layers) - Most commonly used"""
    return VGG('VGG16', num_classes=num_classes, **kwargs)

def VGG19(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-19 model (16 conv layers) - Deepest variant"""
    return VGG('VGG19', num_classes=num_classes, **kwargs)

def VGG16_BN(num_classes: int = 1000, **kwargs) -> VGG:
    """VGG-16 with Batch Normalization - Recommended for training from scratch"""
    return VGG('VGG16', num_classes=num_classes, batch_norm=True, **kwargs)


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
    print("VGG Network Family - Parameter Comparison")
    print("=" * 70)
    
    for config in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
        model = VGG(config, num_classes=1000)
        total, trainable = count_parameters(model)
        print(f"{config}: {total:>15,} parameters")
    
    print("=" * 70)
    print("\nVGG-16 Detailed Summary")
    print("=" * 70)
    
    model = VGG16(num_classes=1000)
    total_params, trainable_params = count_parameters(model)
    
    # Count parameters by component
    feature_params = sum(p.numel() for p in model.features.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"Total parameters:      {total_params:,}")
    print(f"├── Feature extractor: {feature_params:,} ({100*feature_params/total_params:.1f}%)")
    print(f"└── Classifier:        {classifier_params:,} ({100*classifier_params/total_params:.1f}%)")
    print(f"Model size (MB):       {total_params * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    print(f"Sample logits (first 5): {logits[0, :5]}")
    print("=" * 70)
