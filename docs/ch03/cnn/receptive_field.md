# Receptive Field

## Introduction

The **receptive field** of a neuron in a convolutional neural network is the region of the input image that influences the neuron's activation. Understanding receptive fields is crucial for designing effective CNN architectures because it determines:

1. **Context available**: How much spatial context each feature detector can use
2. **Feature scale**: What size patterns the network can recognize
3. **Architecture decisions**: Kernel sizes, depth, and dilation choices

This section provides a rigorous mathematical treatment of receptive field calculation and its implications for network design.

## Definitions

### Local Receptive Field

The **local receptive field** of a neuron is the set of neurons in the previous layer that directly connect to it. For a convolution with kernel size $K$, the local receptive field is $K \times K$.

### Global Receptive Field

The **global receptive field** (or simply "receptive field") is the region of the **original input** that can influence a neuron's output. This accumulates through multiple layers.

### Effective Receptive Field (ERF)

In practice, not all pixels within the theoretical receptive field contribute equally. The **effective receptive field** considers the actual influence of each input pixel, which typically follows a Gaussian distribution centered on the neuron.

## Mathematical Formulation

### Single Layer Receptive Field

For a single convolutional layer with:

- Kernel size: $K$
- Stride: $s$
- Dilation: $d$

The receptive field is:

$$r = d \cdot (K - 1) + 1$$

For standard convolution ($d = 1$):

$$r = K$$

### Multi-Layer Receptive Field

For stacked convolutional layers, the receptive field grows according to:

$$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

where:

- $r_l$: Receptive field after layer $l$
- $r_0 = 1$: Initial receptive field (single pixel)
- $K_l$: Kernel size at layer $l$
- $d_l$: Dilation at layer $l$
- $s_i$: Stride at layer $i$

### Intuitive Understanding

The factor $\prod_{i=1}^{l-1} s_i$ (cumulative stride) is crucial:

1. Each previous stride **amplifies** the receptive field growth
2. A stride-2 layer doubles the "step size" for subsequent layers
3. This is why early downsampling dramatically increases later receptive fields

## Receptive Field Calculation

### Recursive Formula

```python
def calculate_receptive_field(layers):
    """
    Calculate receptive field for a sequence of conv/pool layers.
    
    Args:
        layers: List of tuples (kernel_size, stride, dilation)
    
    Returns:
        Total receptive field size
    """
    receptive_field = 1
    cumulative_stride = 1
    
    for kernel_size, stride, dilation in layers:
        # Add contribution from this layer
        receptive_field += (kernel_size - 1) * dilation * cumulative_stride
        # Update cumulative stride
        cumulative_stride *= stride
    
    return receptive_field

# Example: VGG-style network
vgg_layers = [
    (3, 1, 1),  # Conv 3x3
    (3, 1, 1),  # Conv 3x3
    (2, 2, 1),  # MaxPool 2x2
    (3, 1, 1),  # Conv 3x3
    (3, 1, 1),  # Conv 3x3
    (2, 2, 1),  # MaxPool 2x2
    (3, 1, 1),  # Conv 3x3
    (3, 1, 1),  # Conv 3x3
    (3, 1, 1),  # Conv 3x3
    (2, 2, 1),  # MaxPool 2x2
]

rf = calculate_receptive_field(vgg_layers)
print(f"VGG-style receptive field: {rf}x{rf} pixels")
```

### Jump and Receptive Field Center

For precise localization, we also need to track:

1. **Jump**: How many input pixels correspond to moving one unit in the feature map
2. **Start**: The center of the first feature in input coordinates

```python
def receptive_field_complete(layers):
    """
    Complete receptive field calculation with position tracking.
    
    Returns:
        Dictionary with receptive field, jump, and start position
    """
    n = 1  # Receptive field size
    j = 1  # Jump (cumulative stride)
    r_start = 0.5  # Center of first feature
    
    layer_info = []
    
    for i, (k, s, d) in enumerate(layers):
        # Effective kernel size with dilation
        k_eff = d * (k - 1) + 1
        
        # Update receptive field
        n = n + (k_eff - 1) * j
        
        # Update start position
        r_start = r_start + ((k_eff - 1) / 2 - (s - 1) / 2) * j
        
        # Update jump
        j = j * s
        
        layer_info.append({
            'layer': i + 1,
            'receptive_field': n,
            'jump': j,
            'start': r_start
        })
    
    return layer_info

# Analyze layer by layer
layers = [
    (7, 2, 1),  # Conv 7x7, stride 2
    (3, 2, 1),  # MaxPool 3x3, stride 2
    (3, 1, 1),  # Conv 3x3
    (3, 1, 1),  # Conv 3x3
    (3, 2, 1),  # Conv 3x3, stride 2
]

info = receptive_field_complete(layers)
for layer in info:
    print(f"Layer {layer['layer']}: RF={layer['receptive_field']}, "
          f"Jump={layer['jump']}, Start={layer['start']:.1f}")
```

## Receptive Field Growth Strategies

### Strategy 1: Increase Kernel Size

**Pros**: Direct increase in receptive field  
**Cons**: Quadratic parameter growth, computational cost

$$\text{Parameters} = O(K^2)$$

### Strategy 2: Increase Depth

**Pros**: Linear parameter growth, compositional features  
**Cons**: Vanishing gradients, optimization difficulty

Two 3×3 layers have the same receptive field as one 5×5:

$$r_{two\ 3\times3} = 1 + (3-1) \times 1 + (3-1) \times 1 = 5$$
$$r_{one\ 5\times5} = 1 + (5-1) \times 1 = 5$$

But with fewer parameters:

- Two 3×3 layers: $2 \times 3^2 = 18$ weights
- One 5×5 layer: $5^2 = 25$ weights

### Strategy 3: Increase Stride

**Pros**: Fast receptive field growth, reduces computation  
**Cons**: Information loss, reduced resolution

### Strategy 4: Use Dilation

**Pros**: Exponential RF growth with linear parameters  
**Cons**: Gridding artifacts if not careful

```python
# Dilated convolution receptive fields
standard_layers = [(3, 1, 1)] * 4  # Four 3x3 convs
dilated_layers = [
    (3, 1, 1),   # Dilation 1
    (3, 1, 2),   # Dilation 2
    (3, 1, 4),   # Dilation 4
    (3, 1, 8),   # Dilation 8
]

rf_standard = calculate_receptive_field(standard_layers)
rf_dilated = calculate_receptive_field(dilated_layers)

print(f"Standard (4× 3x3): RF = {rf_standard}")   # RF = 9
print(f"Dilated (d=1,2,4,8): RF = {rf_dilated}")  # RF = 31
```

## Effective Receptive Field

### Gaussian Assumption

Luo et al. (2016) showed that the effective receptive field follows a Gaussian distribution:

$$\text{ERF}(i, j) \propto \exp\left(-\frac{(i - c_i)^2 + (j - c_j)^2}{2\sigma^2}\right)$$

where $(c_i, c_j)$ is the center and $\sigma$ depends on network depth.

### Key Findings

1. **ERF << Theoretical RF**: The effective receptive field is often much smaller than the theoretical maximum
2. **ERF grows with $\sqrt{\text{depth}}$**: Not linearly with depth
3. **Training increases ERF**: Deeper features become more influential
4. **Skip connections help**: Residual connections expand ERF

### Measuring Effective Receptive Field

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def measure_effective_receptive_field(model, input_size=224, target_layer=None):
    """
    Measure the effective receptive field of a CNN by gradient analysis.
    
    This method computes the gradient of a central output unit w.r.t. all inputs.
    """
    # Create input with gradient tracking
    x = torch.zeros(1, 3, input_size, input_size, requires_grad=True)
    
    # Forward pass
    if target_layer:
        # Get activation at specific layer
        features = {}
        def hook(module, input, output):
            features['out'] = output
        handle = target_layer.register_forward_hook(hook)
        _ = model(x)
        handle.remove()
        out = features['out']
    else:
        out = model(x)
    
    # Get center position
    _, _, h, w = out.shape
    center_h, center_w = h // 2, w // 2
    
    # Backward from center output
    out[0, 0, center_h, center_w].backward()
    
    # Get gradient magnitude (ERF)
    erf = x.grad.abs().squeeze().mean(dim=0)  # Average over channels
    
    return erf.detach().numpy()

# Example: Analyze VGG-like network
class SimpleVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
    
    def forward(self, x):
        return self.features(x)

model = SimpleVGG()
erf = measure_effective_receptive_field(model, input_size=64)

# Visualize
plt.figure(figsize=(8, 8))
plt.imshow(erf, cmap='hot')
plt.colorbar(label='Gradient Magnitude')
plt.title('Effective Receptive Field')
plt.xlabel('Width')
plt.ylabel('Height')
plt.savefig('effective_receptive_field.png', dpi=150)
plt.show()
```

## Architecture Analysis

### ResNet Receptive Field

```python
# ResNet-50 receptive field calculation
resnet50_layers = [
    # Stem
    (7, 2, 1),   # Conv1
    (3, 2, 1),   # MaxPool
    
    # Stage 2 (3 blocks, no downsampling in first)
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 1
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 2
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 3
    
    # Stage 3 (4 blocks, stride 2 in first)
    (1, 1, 1), (3, 2, 1), (1, 1, 1),  # Block 1 (downsample)
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 2
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 3
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 4
    
    # Stage 4 (6 blocks)
    (1, 1, 1), (3, 2, 1), (1, 1, 1),  # Block 1 (downsample)
    *[(1, 1, 1), (3, 1, 1), (1, 1, 1)] * 5,  # Blocks 2-6
    
    # Stage 5 (3 blocks)
    (1, 1, 1), (3, 2, 1), (1, 1, 1),  # Block 1 (downsample)
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 2
    (1, 1, 1), (3, 1, 1), (1, 1, 1),  # Block 3
]

rf_resnet = calculate_receptive_field(resnet50_layers)
print(f"ResNet-50 receptive field: {rf_resnet}x{rf_resnet} pixels")
# Significantly larger than 224x224 input!
```

### Comparing Architectures

```python
# Compare receptive fields of different architectures
architectures = {
    'AlexNet-like': [
        (11, 4, 1),  # Conv1
        (3, 2, 1),   # Pool1
        (5, 1, 1),   # Conv2
        (3, 2, 1),   # Pool2
        (3, 1, 1),   # Conv3
        (3, 1, 1),   # Conv4
        (3, 1, 1),   # Conv5
    ],
    
    'VGG-like': [
        (3, 1, 1), (3, 1, 1), (2, 2, 1),  # Block 1
        (3, 1, 1), (3, 1, 1), (2, 2, 1),  # Block 2
        (3, 1, 1), (3, 1, 1), (3, 1, 1), (2, 2, 1),  # Block 3
        (3, 1, 1), (3, 1, 1), (3, 1, 1), (2, 2, 1),  # Block 4
        (3, 1, 1), (3, 1, 1), (3, 1, 1), (2, 2, 1),  # Block 5
    ],
    
    'Dilated': [
        (3, 1, 1),   # d=1
        (3, 1, 2),   # d=2
        (3, 1, 4),   # d=4
        (3, 1, 8),   # d=8
        (3, 1, 16),  # d=16
    ],
}

print("Receptive Field Comparison:")
print("-" * 40)
for name, layers in architectures.items():
    rf = calculate_receptive_field(layers)
    params_proxy = sum(k**2 for k, s, d in layers)
    print(f"{name:15}: RF = {rf:4}, Parameter proxy = {params_proxy}")
```

## Receptive Field and Task Design

### Object Detection

For detecting objects of different sizes, the receptive field at each feature level should match the expected object size:

| Feature Level | Receptive Field | Target Objects |
|---------------|-----------------|----------------|
| High resolution | Small | Small objects |
| Medium resolution | Medium | Medium objects |
| Low resolution | Large | Large objects |

### Semantic Segmentation

For pixel-wise prediction, the receptive field should be large enough to capture context while maintaining spatial precision:

- **DeepLab**: Uses dilated convolutions for large RF without downsampling
- **PSPNet**: Uses pyramid pooling to capture multi-scale context
- **U-Net**: Uses encoder-decoder with skip connections

### Image Classification

For whole-image classification, the final receptive field should ideally cover the entire image. Global average pooling ensures this:

```python
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Feature extractor with large receptive field
        self.features = nn.Sequential(
            # ... convolutional layers ...
        )
        # Global average pooling: RF = entire feature map
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        return self.classifier(x)
```

## PyTorch Utilities

### Receptive Field Calculator Module

```python
import torch
import torch.nn as nn

class ReceptiveFieldTracker(nn.Module):
    """
    Wrapper module that tracks receptive field through a sequential model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rf_info = self._analyze_receptive_field()
    
    def _analyze_receptive_field(self):
        """Analyze receptive field of each layer."""
        info = []
        rf, jump = 1, 1
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                k = module.kernel_size
                k = k[0] if isinstance(k, tuple) else k
                
                s = module.stride
                s = s[0] if isinstance(s, tuple) else s
                
                d = getattr(module, 'dilation', 1)
                d = d[0] if isinstance(d, tuple) else d
                
                k_eff = d * (k - 1) + 1
                rf = rf + (k_eff - 1) * jump
                jump = jump * s
                
                info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'kernel': k,
                    'stride': s,
                    'dilation': d,
                    'receptive_field': rf,
                    'jump': jump
                })
        
        return info
    
    def print_receptive_field(self):
        """Print receptive field analysis."""
        print(f"{'Layer':<30} {'Type':<15} {'K':>3} {'S':>3} {'D':>3} {'RF':>6} {'Jump':>6}")
        print("-" * 75)
        for layer in self.rf_info:
            print(f"{layer['name']:<30} {layer['type']:<15} "
                  f"{layer['kernel']:>3} {layer['stride']:>3} {layer['dilation']:>3} "
                  f"{layer['receptive_field']:>6} {layer['jump']:>6}")
    
    def forward(self, x):
        return self.model(x)

# Example usage
model = nn.Sequential(
    nn.Conv2d(3, 64, 7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(3, stride=2, padding=1),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
)

tracker = ReceptiveFieldTracker(model)
tracker.print_receptive_field()
```

## Summary

Key points about receptive field:

1. **Definition**: The region of input that influences a neuron's activation

2. **Calculation**: 
   $$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

3. **Growth strategies**:
   - Deeper networks (preferred)
   - Larger kernels (costly)
   - Dilated convolutions (efficient)
   - Strided convolutions (aggressive)

4. **Effective vs. Theoretical**: The effective receptive field is typically smaller and Gaussian-shaped

5. **Task implications**: Match receptive field to expected feature sizes

## Exercises

1. **Manual Calculation**: Calculate the receptive field of VGG-16 at each pooling layer.

2. **Architecture Design**: Design a network with receptive field exactly 101×101 using only 3×3 convolutions.

3. **ERF Visualization**: Implement effective receptive field visualization for a pretrained ResNet and compare layers at different depths.

4. **Dilated vs. Deep**: Compare the receptive field growth of (a) 10 standard 3×3 convolutions vs. (b) 5 dilated convolutions with rates 1, 2, 4, 8, 16.

## References

1. Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016). Understanding the effective receptive field in deep convolutional neural networks. *NeurIPS 2016*.

2. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR 2016*.

3. Araujo, A., Norber, W., Hooker, S., & Weinberger, K. (2019). Computing receptive fields of convolutional neural networks. *Distill*.
