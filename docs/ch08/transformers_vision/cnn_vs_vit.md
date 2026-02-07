# CNN vs Vision Transformer Comparison

## Introduction

Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) represent two fundamentally different approaches to image understanding. Understanding their differences, strengths, and trade-offs is crucial for selecting the right architecture for a given task.

## Architectural Foundations

### CNN: Local to Global

CNNs build representations through a hierarchy of local operations:

```
Input Image
    ↓ (3×3 conv, stride 1) → Local features, small receptive field
    ↓ (3×3 conv + pool) → Larger receptive field
    ↓ (3×3 conv + pool) → Even larger receptive field
    ↓ ...
    ↓ → Global features through many layers
Output
```

### ViT: Global from Start

ViT processes images as sequences with global attention:

```
Input Image
    ↓ (patch + linear projection) → Sequence of tokens
    ↓ (self-attention) → All tokens attend to all others
    ↓ (self-attention) → Global context from first layer
    ↓ ...
    ↓ → Rich global representations
Output
```

## Implementation Comparison

### Simple CNN

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Traditional CNN architecture for comparison."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 112 → 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 56 → 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 28 → 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

## Detailed Comparison

### 1. Receptive Field

| Aspect | CNN | ViT |
|--------|-----|-----|
| Layer 1 | 3×3 (local) | 224×224 (global) |
| Growth | Gradual (hierarchical) | Constant (global) |
| Pattern | Fixed (kernel-based) | Learned (attention) |

### 2. Computational Complexity

**Convolution Complexity:**
$$O(\text{Conv}) = O(k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W)$$

**Self-Attention Complexity:**
$$O(\text{Attention}) = O(N^2 \cdot D + N \cdot D^2)$$

where $N = \frac{H \times W}{P^2}$ is the number of patches.

### 3. Inductive Biases

| Bias | CNN | ViT |
|------|-----|-----|
| Locality | Strong (built-in) | None (learned) |
| Translation equivariance | Yes | No (requires data) |
| Scale invariance | Partial (with pooling) | None |

### 4. Data Requirements

| Dataset Size | Recommended Architecture |
|--------------|-------------------------|
| Small (<100K) | CNN |
| Medium (100K-1M) | CNN or Hybrid |
| Large (1M-10M) | ViT with heavy augmentation |
| Very Large (>10M) | ViT excels |

### 5. Training Configurations

**CNN:**
```python
cnn_config = {
    'optimizer': 'SGD',
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_schedule': 'step_decay',
    'epochs': 90
}
```

**ViT:**
```python
vit_config = {
    'optimizer': 'AdamW',
    'lr': 3e-4,
    'warmup_epochs': 5,
    'weight_decay': 0.1,
    'lr_schedule': 'cosine',
    'augmentation': 'strong',
    'epochs': 300,
    'label_smoothing': 0.1
}
```

## Benchmark Comparison

### ImageNet Classification

| Model | Params | Top-1 Acc | Training Data |
|-------|--------|-----------|---------------|
| ResNet-50 | 25M | 76.1% | ImageNet-1K |
| ResNet-152 | 60M | 78.3% | ImageNet-1K |
| EfficientNet-B7 | 66M | 84.3% | ImageNet-1K |
| ViT-B/16 | 86M | 77.9% | ImageNet-1K |
| ViT-B/16 | 86M | 84.0% | ImageNet-21K |
| ViT-L/16 | 307M | 87.8% | JFT-300M |

## When to Use Each

### Use CNN When:

1. **Limited Data**: Dataset smaller than 100K images
2. **Speed Critical**: Real-time inference requirements
3. **Edge Deployment**: Resource-constrained devices
4. **Known Local Structure**: Task benefits from locality bias

### Use ViT When:

1. **Large Dataset**: Millions of images available
2. **Transfer Learning**: Leveraging pretrained models
3. **Global Context**: Task requires long-range dependencies
4. **State-of-the-art**: Maximum performance needed

### Use Hybrid When:

1. **Medium Dataset**: 100K - 1M images
2. **Balance Needed**: Both local and global reasoning
3. **Dense Prediction**: Segmentation, detection tasks

## Attention vs Convolution: Deep Dive

Self-attention can be viewed as a data-dependent convolution:

- **Convolution**: Fixed weights, local scope
- **Attention**: Learned weights (content-dependent), global scope

Both compute weighted sums, but attention weights are dynamic:
```
Convolution: y = sum(w_fixed * x_local)
Attention:   y = sum(w_dynamic(q,k) * x_global)
```

## Summary Table

| Aspect | CNN | ViT |
|--------|-----|-----|
| Receptive field | Local → Global | Global always |
| Inductive bias | Strong (locality) | Weak (flexible) |
| Data efficiency | High | Low |
| Scalability | Moderate | High |
| Interpretability | Activation maps | Attention maps |
| Training stability | High | Requires care |
| Inference speed | Fast | Moderate |
| Memory usage | Linear in size | Quadratic in patches |

## Conclusion

CNNs and Vision Transformers represent different trade-offs:

- **CNNs** embed useful biases that help with limited data but may limit flexibility
- **ViTs** learn everything from data, excelling when data is abundant
- **Hybrid approaches** combine strengths of both

## References

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.
2. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
3. Raghu, M., et al. "Do Vision Transformers See Like CNNs?" NeurIPS 2021.
