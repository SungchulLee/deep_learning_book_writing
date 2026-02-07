#!/usr/bin/env python3
"""
================================================================================
AlexNet - Deep Convolutional Neural Network
================================================================================

Paper: "ImageNet Classification with Deep Convolutional Neural Networks" (2012)
Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
Link: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
AlexNet is widely credited with sparking the deep learning revolution. Its 
decisive victory in the ImageNet Large Scale Visual Recognition Challenge 
(ILSVRC) 2012 demonstrated that deep convolutional neural networks could 
dramatically outperform traditional computer vision methods.

- **ILSVRC 2012 Winner**: Top-5 error rate of 15.3% (vs 26.2% for second place)
- **Revival of Neural Networks**: Reignited interest after the "AI winter"
- **GPU Acceleration Pioneer**: One of the first to leverage CUDA for training

================================================================================
KEY INNOVATIONS
================================================================================

1. **ReLU Activation Function**
   - Used ReLU: f(x) = max(0, x) instead of tanh/sigmoid
   - Advantage: Does not saturate for positive values
   - Result: 6x faster training than with tanh
   - Mitigates vanishing gradient problem

2. **Dropout Regularization**
   - Randomly drops 50% of neurons during training
   - Prevents co-adaptation of feature detectors
   - Acts as ensemble of exponentially many networks
   - Applied to first two fully connected layers

3. **Data Augmentation**
   - Random 224×224 crops from 256×256 images
   - Horizontal reflections (mirror images)
   - PCA-based color augmentation (AlexNet-specific)
   - Increased training set size without new data

4. **GPU Training**
   - Trained on two NVIDIA GTX 580 GPUs (3GB each)
   - Network split across GPUs with cross-communication
   - Reduced training time from weeks to days

5. **Local Response Normalization (LRN)**
   - Normalize responses across neighboring channels
   - Inspired by lateral inhibition in neuroscience
   - Note: Later research showed LRN has minimal impact;
     modern implementations often omit it (as we do here)

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

Layer-by-Layer Breakdown:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer      │ Operation           │ Output Shape    │ Parameters  │ Notes   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input      │ -                   │ 224×224×3       │ -           │ RGB     │
│ Conv1      │ 11×11, stride 4     │ 55×55×96        │ 34,944      │ Large RF│
│ Pool1      │ 3×3 max, stride 2   │ 27×27×96        │ -           │         │
│ Conv2      │ 5×5, stride 1       │ 27×27×256       │ 614,656     │         │
│ Pool2      │ 3×3 max, stride 2   │ 13×13×256       │ -           │         │
│ Conv3      │ 3×3, stride 1       │ 13×13×384       │ 885,120     │         │
│ Conv4      │ 3×3, stride 1       │ 13×13×384       │ 1,327,488   │         │
│ Conv5      │ 3×3, stride 1       │ 13×13×256       │ 884,992     │         │
│ Pool5      │ 3×3 max, stride 2   │ 6×6×256         │ -           │         │
│ FC1        │ 9216 → 4096         │ 4096            │ 37,752,832  │ Dropout │
│ FC2        │ 4096 → 4096         │ 4096            │ 16,781,312  │ Dropout │
│ FC3        │ 4096 → 1000         │ 1000            │ 4,097,000   │ Output  │
└─────────────────────────────────────────────────────────────────────────────┘
Total Parameters: ~62 million (mostly in FC layers)

Note: This implementation uses 64 filters in Conv1 (vs original 96) for 
computational efficiency. The original paper split computation across 2 GPUs.

================================================================================
MATHEMATICAL FOUNDATIONS
================================================================================

**Convolution Operation:**
For 2D convolution with input I, kernel K, bias b:
    (I * K)(i,j) = Σ_m Σ_n I(i+m, j+n) · K(m,n) + b

**Output Size Calculation:**
    output_size = floor((input_size - kernel_size + 2*padding) / stride) + 1

**ReLU Activation:**
    ReLU(x) = max(0, x)
    
    Derivative: dReLU/dx = 1 if x > 0, else 0
    (Note: undefined at x=0, typically treated as 0)

**Dropout (during training):**
    y = x · mask / (1 - p)
    where mask ~ Bernoulli(1-p), p = dropout probability
    
    The division by (1-p) ensures expected output equals input.

================================================================================
TRAINING DETAILS (Original Paper)
================================================================================

- Optimizer: SGD with momentum 0.9
- Learning Rate: 0.01, reduced by 10x when validation error plateaus
- Weight Decay: 0.0005 (L2 regularization)
- Batch Size: 128
- Training Time: ~6 days on two GTX 580 GPUs
- Epochs: ~90

Weight Initialization:
- Biases in conv layers 2, 4, 5 and FC layers: 1 (speeds up early learning)
- Other biases: 0
- Weights: Gaussian with mean=0, std=0.01

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: Convolutional Neural Networks (padding, stride, pooling)
- Ch04: Advanced Image Classification (classic architectures)
- Ch08: Transfer Learning (pretrained feature extraction)
- Ch29: Model Interpretability (Grad-CAM visualization on conv layers)

Related architectures in this repository:
- VGGNet: Deeper with smaller filters (vgg_net.py)
- GoogLeNet: Inception modules (google_net_inception.py)
- ResNet: Residual connections (resnet.py)

================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple


class AlexNet(nn.Module):
    """
    AlexNet Implementation for Image Classification
    
    This is a modernized implementation that follows the original architecture
    principles while incorporating contemporary best practices (e.g., adaptive
    pooling for input size flexibility).
    
    Args:
        num_classes (int): Number of output classes. Default: 1000 (ImageNet)
        dropout (float): Dropout probability for FC layers. Default: 0.5
    
    Example:
        >>> model = AlexNet(num_classes=10)  # For CIFAR-10
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([1, 10])
    
    Shape:
        - Input: (N, 3, 224, 224) where N is batch size
        - Output: (N, num_classes)
    
    References:
        - Original Paper: https://papers.nips.cc/paper/4824
        - PyTorch Vision: torchvision.models.alexnet
    """
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        
        # ====================================================================
        # FEATURE EXTRACTION LAYERS (Convolutional Part)
        # ====================================================================
        # These layers learn hierarchical visual features:
        # - Early layers: edges, colors, textures
        # - Middle layers: parts, patterns
        # - Later layers: object parts, high-level features
        
        self.features = nn.Sequential(
            # ----------------------------------------------------------------
            # Conv Layer 1: Initial Feature Extraction
            # ----------------------------------------------------------------
            # Input:  (batch_size, 3, 224, 224)
            # Output: (batch_size, 64, 55, 55)
            # 
            # Calculation: (224 - 11 + 2*2) / 4 + 1 = 55
            #
            # Why 11x11 kernel?
            # - Large receptive field captures broad low-level features
            # - First layer needs to see larger context
            # - Subsequent layers use smaller kernels (feature abstraction)
            #
            # Why stride=4?
            # - Aggressive downsampling reduces computation
            # - 224→55 is a 4x reduction in each dimension
            # - Some spatial information lost, but acceptable for classification
            nn.Conv2d(
                in_channels=3,       # RGB input
                out_channels=64,     # Number of learned filters
                kernel_size=11,      # Large kernel for first layer
                stride=4,            # Aggressive downsampling
                padding=2            # Maintains reasonable output size
            ),
            # ReLU introduces non-linearity without vanishing gradients
            # inplace=True saves memory by modifying tensor directly
            nn.ReLU(inplace=True),
            
            # Max Pooling: 55x55 -> 27x27
            # Provides translation invariance and further reduces dimensions
            # Calculation: (55 - 3) / 2 + 1 = 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # ----------------------------------------------------------------
            # Conv Layer 2: Deeper Feature Learning
            # ----------------------------------------------------------------
            # Input:  (batch_size, 64, 27, 27)
            # Output: (batch_size, 192, 27, 27)
            #
            # 5x5 kernel with padding=2 preserves spatial dimensions
            # Calculation: (27 - 5 + 2*2) / 1 + 1 = 27
            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                padding=2            # Same convolution (preserves size)
            ),
            nn.ReLU(inplace=True),
            
            # Max Pooling: 27x27 -> 13x13
            # Calculation: (27 - 3) / 2 + 1 = 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # ----------------------------------------------------------------
            # Conv Layers 3, 4, 5: High-Level Feature Extraction
            # ----------------------------------------------------------------
            # No pooling between these layers - deeper feature extraction
            # All use 3x3 kernels with padding=1 (preserves spatial size)
            
            # Conv Layer 3: 13x13x192 -> 13x13x384
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Layer 4: 13x13x384 -> 13x13x256
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv Layer 5: 13x13x256 -> 13x13x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Final Max Pooling: 13x13 -> 6x6
            # Calculation: (13 - 3) / 2 + 1 = 6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # ====================================================================
        # ADAPTIVE POOLING
        # ====================================================================
        # Ensures output is always 6x6 regardless of minor input size variations
        # This makes the model more robust to different input resolutions
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # ====================================================================
        # CLASSIFICATION LAYERS (Fully Connected Part)
        # ====================================================================
        # These layers combine spatial features for final classification
        # Contains majority of parameters (~58M of ~62M total)
        
        self.classifier = nn.Sequential(
            # ----------------------------------------------------------------
            # Dropout for Regularization
            # ----------------------------------------------------------------
            # Randomly zeroes p=0.5 (50%) of elements during training
            # Prevents overfitting by encouraging redundant representations
            # During inference, all neurons active but outputs scaled by (1-p)
            nn.Dropout(p=dropout),
            
            # ----------------------------------------------------------------
            # FC Layer 1: 9216 -> 4096
            # ----------------------------------------------------------------
            # Flattened conv output: 256 channels × 6 × 6 = 9216
            # This layer alone has ~37.7M parameters!
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            
            # Second dropout layer
            nn.Dropout(p=dropout),
            
            # ----------------------------------------------------------------
            # FC Layer 2: 4096 -> 4096
            # ----------------------------------------------------------------
            # Further feature combination
            # ~16.7M parameters
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            
            # ----------------------------------------------------------------
            # FC Layer 3 (Output): 4096 -> num_classes
            # ----------------------------------------------------------------
            # No activation - raw logits for CrossEntropyLoss
            # CrossEntropyLoss = LogSoftmax + NLLLoss (more numerically stable)
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through AlexNet
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        
        Note:
            - Returns raw logits, not probabilities
            - Use softmax for probabilities: F.softmax(logits, dim=1)
            - CrossEntropyLoss expects raw logits
        """
        # Feature extraction: (N, 3, 224, 224) -> (N, 256, 6, 6)
        x = self.features(x)
        
        # Adaptive pooling ensures consistent size
        x = self.avgpool(x)
        
        # Flatten: (N, 256, 6, 6) -> (N, 9216)
        # start_dim=1 preserves batch dimension
        x = torch.flatten(x, start_dim=1)
        
        # Classification: (N, 9216) -> (N, num_classes)
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier (useful for transfer learning)
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, 9216)
        
        Example:
            >>> model = AlexNet(num_classes=1000)
            >>> features = model.get_features(x)
            >>> # Use features for downstream tasks
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return x


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# DEMO AND TESTING
# ============================================================================
if __name__ == "__main__":
    # Create model instance
    model = AlexNet(num_classes=1000)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 70)
    print("AlexNet Model Summary")
    print("=" * 70)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB):      {total_params * 4 / 1024 / 1024:.2f}")
    print("=" * 70)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape:  {x.shape}")
    
    # Evaluation mode (disables dropout)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    print(f"\nSample output (first 5 logits): {logits[0, :5]}")
    
    # Feature extraction demo
    print("\n" + "=" * 70)
    print("Feature Extraction Demo")
    print("=" * 70)
    with torch.no_grad():
        features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    print("=" * 70)
