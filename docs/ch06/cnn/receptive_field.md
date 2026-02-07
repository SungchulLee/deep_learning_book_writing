# Receptive Field

## Introduction

The **receptive field** of a neuron in a convolutional neural network is the region of the input image that influences the neuron's activation. Understanding receptive fields is fundamental to CNN design because it determines:

1. **Context available**: How much spatial context each feature detector can access
2. **Feature scale**: What size patterns the network can recognize
3. **Architecture decisions**: Optimal kernel sizes, network depth, and dilation choices
4. **Task suitability**: Whether the network can capture the spatial relationships required

This section provides a rigorous mathematical treatment of receptive field calculation, practical computation tools, and design implications.

## Definitions

### Local Receptive Field

The **local receptive field** of a neuron is the set of neurons in the previous layer that directly connect to it. For a convolution with kernel size $K$, the local receptive field is $K \times K$.

```
Layer 1 output neuron sees a 3×3 region of input:

Input:              Layer 1:
■ ■ ■ · ·           ○ · ·
■ ■ ■ · ·     →     · · ·
■ ■ ■ · ·           · · ·
· · · · ·
· · · · ·

Local receptive field = 3×3
```

### Global Receptive Field

The **global receptive field** (or simply "receptive field") is the region of the **original input** that can influence a neuron's output. This accumulates through multiple layers:

```
Input:                Layer 1:              Layer 2:
■ ■ ■ ■ ■             ■ ■ ■                 ○
■ ■ ■ ■ ■    3×3      ■ ■ ■      3×3        
■ ■ ■ ■ ■    →        ■ ■ ■      →          
■ ■ ■ ■ ■                                   
■ ■ ■ ■ ■                                   

Layer 1 RF = 3×3     Layer 2 RF = 5×5
```

### Effective Receptive Field (ERF)

The **theoretical** receptive field is not the same as the **effective** receptive field. In practice, not all pixels within the theoretical receptive field contribute equally to a neuron's activation:

- **Center pixels** have higher influence (more paths through the network)
- **Edge pixels** have lower influence (fewer contributing paths)

The ERF typically follows a Gaussian distribution centered on the neuron, and is often much smaller than the theoretical maximum.

## Mathematical Formulation

### Single Layer Receptive Field

For a single convolutional layer with:
- Kernel size: $K$
- Stride: $s$
- Dilation: $d$

The effective kernel size (accounting for dilation) is:

$$K_{\text{eff}} = d \cdot (K - 1) + 1$$

For standard convolution ($d = 1$): $K_{\text{eff}} = K$

### Multi-Layer Receptive Field

For stacked convolutional layers, the receptive field grows according to:

$$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

where:
- $r_l$: Receptive field after layer $l$
- $r_0 = 1$: Initial receptive field (single pixel)
- $K_l$: Kernel size at layer $l$
- $d_l$: Dilation at layer $l$
- $s_i$: Stride at layer $i$
- $\prod_{i=1}^{l-1} s_i$: Cumulative stride (also called "jump")

### The Cumulative Stride Factor

The factor $\prod_{i=1}^{l-1} s_i$ (cumulative stride or "jump") is crucial for understanding receptive field growth:

1. Each previous stride **amplifies** the receptive field growth of subsequent layers
2. A stride-2 layer doubles the "step size" for all later layers
3. This is why early downsampling dramatically increases receptive fields in deeper layers

### Simplified Formulas

**For uniform architecture** (same kernel $k$, stride $s$ throughout):

$$r_L = 1 + \sum_{l=1}^{L} (k - 1) \cdot d_l \cdot s^{l-1}$$

**For all stride-1 convolutions** (no downsampling):

$$r_L = 1 + L \cdot (k - 1)$$

This shows that with stride-1, receptive field grows linearly with depth.

## Computing Receptive Field

### Core Python Implementation

```python
def compute_receptive_field(layers):
    """
    Compute receptive field for a sequence of conv/pool layers.
    
    Args:
        layers: List of dicts with 'kernel', 'stride', 'dilation' keys
    
    Returns:
        Receptive field size and jump (cumulative stride)
    """
    rf = 1  # Start with single pixel
    jump = 1  # Cumulative stride
    
    for layer in layers:
        k = layer.get('kernel', 1)
        s = layer.get('stride', 1)
        d = layer.get('dilation', 1)
        
        # Effective kernel size with dilation
        k_eff = d * (k - 1) + 1
        
        # Update receptive field
        rf = rf + (k_eff - 1) * jump
        
        # Update jump (cumulative stride)
        jump = jump * s
    
    return rf, jump


# Example: VGG-style network
vgg_layers = [
    {'kernel': 3, 'stride': 1},  # Conv 3×3
    {'kernel': 3, 'stride': 1},  # Conv 3×3
    {'kernel': 2, 'stride': 2},  # MaxPool 2×2
    {'kernel': 3, 'stride': 1},  # Conv 3×3
    {'kernel': 3, 'stride': 1},  # Conv 3×3
    {'kernel': 2, 'stride': 2},  # MaxPool 2×2
]

rf, jump = compute_receptive_field(vgg_layers)
print(f"Receptive field: {rf}×{rf}")  # 22×22
print(f"Jump (output stride): {jump}")  # 4
```

### Layer-by-Layer Analysis with Position Tracking

For precise localization, we also track:
- **Jump**: How many input pixels correspond to moving one unit in the feature map
- **Start**: The center of the first feature in input coordinates

```python
def analyze_receptive_field(layers, layer_names=None):
    """
    Detailed receptive field analysis for each layer.
    
    Tracks receptive field size, jump, and center position.
    """
    rf = 1
    jump = 1
    start = 0.5  # Center of first feature (0-indexed)
    
    print(f"{'Layer':<20} {'Kernel':<8} {'Stride':<8} {'Dilation':<8} {'RF':<8} {'Jump':<8}")
    print("-" * 68)
    print(f"{'Input':<20} {'-':<8} {'-':<8} {'-':<8} {rf:<8} {jump:<8}")
    
    for i, layer in enumerate(layers):
        k = layer.get('kernel', 1)
        s = layer.get('stride', 1)
        d = layer.get('dilation', 1)
        
        # Effective kernel size
        k_eff = d * (k - 1) + 1
        
        # Update receptive field
        rf = rf + (k_eff - 1) * jump
        
        # Update start position
        start = start + ((k_eff - 1) / 2) * jump
        
        # Update jump
        jump = jump * s
        
        name = layer_names[i] if layer_names else f"Layer {i+1}"
        print(f"{name:<20} {k:<8} {s:<8} {d:<8} {rf:<8} {jump:<8}")
    
    return rf, jump


# Analyze ResNet-style first few layers
resnet_layers = [
    {'kernel': 7, 'stride': 2},   # Conv1
    {'kernel': 3, 'stride': 2},   # MaxPool
    {'kernel': 3, 'stride': 1},   # Block1 Conv1
    {'kernel': 3, 'stride': 1},   # Block1 Conv2
    {'kernel': 3, 'stride': 2},   # Block2 Conv1 (stride)
    {'kernel': 3, 'stride': 1},   # Block2 Conv2
]

names = ['Conv1 7×7/2', 'MaxPool 3×3/2', 'Block1 3×3', 'Block1 3×3',
         'Block2 3×3/2', 'Block2 3×3']

analyze_receptive_field(resnet_layers, names)
```

**Output:**
```
Layer                Kernel   Stride   Dilation RF       Jump    
--------------------------------------------------------------------
Input                -        -        -        1        1       
Conv1 7×7/2          7        2        1        7        2       
MaxPool 3×3/2        3        2        1        11       4       
Block1 3×3           3        1        1        19       4       
Block1 3×3           3        1        1        27       4       
Block2 3×3/2         3        2        1        35       8       
Block2 3×3           3        1        1        51       8       
```

## Receptive Field Growth Strategies

### Strategy 1: Increase Kernel Size

**Pros**: Direct increase in receptive field  
**Cons**: Quadratic parameter growth, computational cost

$$\text{Parameters} \propto K^2$$

### Strategy 2: Increase Depth (Preferred)

**Pros**: Linear parameter growth, compositional features  
**Cons**: Vanishing gradients, optimization difficulty

Two 3×3 layers have the same receptive field as one 5×5:

$$r_{\text{two } 3\times3} = 1 + (3-1) + (3-1) = 5$$
$$r_{\text{one } 5\times5} = 1 + (5-1) = 5$$

But with **fewer parameters**:
- Two 3×3 layers: $2 \times 3^2 = 18$ weights per channel
- One 5×5 layer: $5^2 = 25$ weights per channel

And **more non-linearity**: Two ReLUs vs. one.

### Strategy 3: Increase Stride

**Pros**: Fast receptive field growth, reduces computation  
**Cons**: Information loss, reduced spatial resolution

### Strategy 4: Use Dilation (Efficient)

**Pros**: Exponential RF growth with linear parameters  
**Cons**: Gridding artifacts if not designed carefully

```python
def compare_receptive_fields():
    """Compare standard vs dilated convolutions."""
    
    # Standard 3×3 convolutions (5 layers)
    standard = [{'kernel': 3, 'stride': 1} for _ in range(5)]
    
    # Dilated convolutions with increasing dilation
    dilated = [
        {'kernel': 3, 'stride': 1, 'dilation': 1},
        {'kernel': 3, 'stride': 1, 'dilation': 2},
        {'kernel': 3, 'stride': 1, 'dilation': 4},
        {'kernel': 3, 'stride': 1, 'dilation': 8},
        {'kernel': 3, 'stride': 1, 'dilation': 16},
    ]
    
    rf_standard, _ = compute_receptive_field(standard)
    rf_dilated, _ = compute_receptive_field(dilated)
    
    print(f"Standard 5 layers (3×3): RF = {rf_standard}")   # 11
    print(f"Dilated 5 layers (d=1,2,4,8,16): RF = {rf_dilated}")  # 63
    print(f"Ratio: {rf_dilated / rf_standard:.1f}x larger with same parameters!")


compare_receptive_fields()
```

### WaveNet-Style Exponential Dilation

```python
def wavenet_receptive_field(num_blocks, layers_per_block, kernel_size=2):
    """
    Calculate receptive field for WaveNet-style architecture.
    
    Dilation pattern: 1, 2, 4, 8, ... repeating for each block.
    """
    layers = []
    for block in range(num_blocks):
        for i in range(layers_per_block):
            dilation = 2 ** i
            layers.append({
                'kernel': kernel_size,
                'stride': 1,
                'dilation': dilation
            })
    
    rf, _ = compute_receptive_field(layers)
    return rf, len(layers)


# WaveNet with 3 blocks of 10 layers each
rf, total_layers = wavenet_receptive_field(3, 10, kernel_size=2)
print(f"WaveNet (3 blocks × 10 layers): {total_layers} layers, RF = {rf}")
# RF = 3 × (2^10 - 1) + 1 = 3069
```

## Effective Receptive Field (ERF)

### The Gap Between Theory and Practice

Luo et al. (2016) showed that the effective receptive field follows a Gaussian distribution:

$$\text{ERF}(i, j) \propto \exp\left(-\frac{(i - c_i)^2 + (j - c_j)^2}{2\sigma^2}\right)$$

where $(c_i, c_j)$ is the center and $\sigma$ depends on network depth.

### Key Empirical Findings

1. **ERF << Theoretical RF**: The effective receptive field is often much smaller than the theoretical maximum
2. **ERF grows with $\sqrt{\text{depth}}$**: Not linearly with depth
3. **Training increases ERF**: Network learns to use more context over training
4. **Skip connections help**: Residual connections expand ERF significantly
5. **Batch normalization**: Can increase ERF by normalizing feature statistics

### Factors Affecting ERF

| Factor | Effect on ERF |
|--------|--------------|
| ReLU activations | Reduces ERF (dead neurons block gradient flow) |
| Batch normalization | Can increase ERF |
| Skip connections | Increases ERF (direct gradient paths) |
| Attention mechanisms | Dynamic, context-dependent ERF |
| Initialization | Affects ERF in early training |
| Training duration | ERF typically grows during training |

### Measuring Effective Receptive Field

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def measure_effective_receptive_field(model, input_size=224, target_layer=None):
    """
    Measure the effective receptive field of a CNN by gradient analysis.
    
    Computes the gradient of a central output unit w.r.t. all inputs.
    The magnitude of this gradient indicates each input pixel's influence.
    """
    model.eval()
    
    # Create input with gradient tracking
    x = torch.zeros(1, 3, input_size, input_size, requires_grad=True)
    
    # Forward pass
    if target_layer:
        # Get activation at specific layer via hook
        features = {}
        def hook(module, input, output):
            features['out'] = output
        handle = target_layer.register_forward_hook(hook)
        _ = model(x)
        output = features['out']
        handle.remove()
    else:
        output = model(x)
    
    # Get center position of output
    if output.dim() == 4:  # Feature map
        h, w = output.shape[2], output.shape[3]
        center_h, center_w = h // 2, w // 2
        
        # Backprop from center neuron (single channel)
        grad_output = torch.zeros_like(output)
        grad_output[0, 0, center_h, center_w] = 1.0
        output.backward(grad_output)
    else:  # Flattened output
        grad_output = torch.zeros_like(output)
        grad_output[0, output.shape[1] // 2] = 1.0
        output.backward(grad_output)
    
    # ERF is absolute gradient w.r.t. input (sum across channels)
    erf = x.grad.abs().sum(dim=1).squeeze().detach().numpy()
    
    return erf


def visualize_erf_concept():
    """Illustrate the concept of effective vs theoretical RF."""
    
    size = 51
    center = size // 2
    
    # Theoretical RF (uniform box)
    theoretical = np.zeros((size, size))
    theoretical[5:46, 5:46] = 1.0  # 41×41 theoretical RF
    
    # Effective RF (Gaussian-like)
    y, x = np.ogrid[-center:size-center, -center:size-center]
    sigma = 8
    effective = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].imshow(theoretical, cmap='Blues')
    axes[0].set_title('Theoretical Receptive Field\n(all pixels equal weight)')
    axes[0].axis('off')
    
    axes[1].imshow(effective, cmap='hot')
    axes[1].set_title('Effective Receptive Field\n(center-weighted, Gaussian)')
    axes[1].axis('off')
    
    # Cross-section comparison
    axes[2].plot(theoretical[center, :], 'b-', linewidth=2, label='Theoretical')
    axes[2].plot(effective[center, :], 'r-', linewidth=2, label='Effective')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Weight')
    axes[2].set_title('Cross-Section Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('erf_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'erf_comparison.png'")
    
    return fig


visualize_erf_concept()
```

## Architecture Analysis

### Common Architectures and Their RF

| Architecture | RF Strategy | Approximate Final RF |
|--------------|-------------|---------------------|
| AlexNet | Large kernels (11×11, 5×5) | 195×195 |
| VGG-16 | Small kernels (3×3), deep | 212×212 |
| ResNet-50 | Skip connections, bottlenecks | 483×483 |
| Inception | Multi-scale parallel branches | Variable per branch |
| U-Net | Encoder-decoder with skips | Full resolution |
| DeepLab | Dilated convolutions (ASPP) | Very large |
| EfficientNet | Compound scaling | Scales with model |

### RF Requirements by Task

| Task | RF Requirement | Rationale |
|------|----------------|-----------|
| Edge detection | Small (3×3 - 7×7) | Local gradients only |
| Texture classification | Medium (30×30 - 100×100) | Texture patterns |
| Object detection | Large (>100×100) | Whole object context |
| Scene understanding | Very large (>200×200) | Global relationships |
| Semantic segmentation | Full image context | Dense prediction needs context |

### ResNet-50 Detailed Analysis

```python
# ResNet-50 receptive field calculation (bottleneck blocks)
def analyze_resnet50():
    layers = [
        # Stem
        {'kernel': 7, 'stride': 2, 'name': 'Conv1'},
        {'kernel': 3, 'stride': 2, 'name': 'MaxPool'},
    ]
    
    # Stage 2 (3 blocks, 256 channels)
    for i in range(3):
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage2.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': 1, 'name': f'Stage2.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage2.Block{i+1}.1x1'},
        ])
    
    # Stage 3 (4 blocks, 512 channels, first has stride 2)
    for i in range(4):
        stride = 2 if i == 0 else 1
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage3.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': stride, 'name': f'Stage3.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage3.Block{i+1}.1x1'},
        ])
    
    # Stage 4 (6 blocks, 1024 channels)
    for i in range(6):
        stride = 2 if i == 0 else 1
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage4.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': stride, 'name': f'Stage4.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage4.Block{i+1}.1x1'},
        ])
    
    # Stage 5 (3 blocks, 2048 channels)
    for i in range(3):
        stride = 2 if i == 0 else 1
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage5.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': stride, 'name': f'Stage5.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage5.Block{i+1}.1x1'},
        ])
    
    rf, jump = compute_receptive_field(layers)
    print(f"ResNet-50 final receptive field: {rf}×{rf} pixels")
    print(f"Final jump (output stride): {jump}")
    print(f"Note: RF ({rf}) > typical input size (224)!")
    
    return rf


analyze_resnet50()
```

### Comparing Architectures

```python
def compare_architectures():
    """Compare receptive fields of different architecture styles."""
    
    architectures = {
        'AlexNet-like': [
            {'kernel': 11, 'stride': 4},  # Conv1
            {'kernel': 3, 'stride': 2},   # Pool1
            {'kernel': 5, 'stride': 1},   # Conv2
            {'kernel': 3, 'stride': 2},   # Pool2
            {'kernel': 3, 'stride': 1},   # Conv3
            {'kernel': 3, 'stride': 1},   # Conv4
            {'kernel': 3, 'stride': 1},   # Conv5
        ],
        
        'VGG-like (16 layers)': [
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
        ],
        
        'Dilated (5 layers)': [
            {'kernel': 3, 'stride': 1, 'dilation': 1},
            {'kernel': 3, 'stride': 1, 'dilation': 2},
            {'kernel': 3, 'stride': 1, 'dilation': 4},
            {'kernel': 3, 'stride': 1, 'dilation': 8},
            {'kernel': 3, 'stride': 1, 'dilation': 16},
        ],
    }
    
    print("Architecture Receptive Field Comparison")
    print("=" * 50)
    print(f"{'Architecture':<25} {'RF':>8} {'Layers':>8} {'Params':>10}")
    print("-" * 50)
    
    for name, layers in architectures.items():
        rf, _ = compute_receptive_field(layers)
        num_layers = len(layers)
        # Proxy for parameters (sum of k^2)
        params_proxy = sum(l.get('kernel', 1)**2 for l in layers)
        print(f"{name:<25} {rf:>8} {num_layers:>8} {params_proxy:>10}")


compare_architectures()
```

## Complete Example: RF-Aware Network Design

```python
import torch
import torch.nn as nn

class RFAwareNetwork(nn.Module):
    """
    Network designed with specific receptive field targets.
    
    Target: ~128×128 receptive field for recognizing objects up to 100×100 pixels.
    
    Design rationale:
    - Start with standard convolutions for fine features
    - Use pooling for controlled downsampling  
    - Switch to dilated convolutions for rapid RF growth without resolution loss
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Layer-by-layer RF calculation:
        # Layer 1: RF = 1 + (3-1)*1 = 3, jump = 1
        # Layer 2: RF = 3 + (3-1)*1 = 5, jump = 1
        # Pool 1:  RF = 5 + (2-1)*1 = 6, jump = 2
        # Layer 3: RF = 6 + (3-1)*2 = 10, jump = 2
        # Layer 4: RF = 10 + (3-1)*2 = 14, jump = 2
        # Pool 2:  RF = 14 + (2-1)*2 = 16, jump = 4
        # Layer 5 (d=2): RF = 16 + (5-1)*4 = 32, jump = 4  [k_eff=5]
        # Layer 6 (d=4): RF = 32 + (9-1)*4 = 64, jump = 4  [k_eff=9]
        # Layer 7 (d=8): RF = 64 + (17-1)*4 = 128, jump = 4 [k_eff=17]
        # Final RF: 128×128 ✓
        
        self.features = nn.Sequential(
            # Block 1: Standard convolutions (fine features)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: Standard convolutions
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: Dilated convolutions (rapid RF growth)
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=8, dilation=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def verify_network_design():
    """Verify the receptive field calculation and test the network."""
    
    layers = [
        {'kernel': 3, 'stride': 1},  # Conv1
        {'kernel': 3, 'stride': 1},  # Conv2
        {'kernel': 2, 'stride': 2},  # Pool1
        {'kernel': 3, 'stride': 1},  # Conv3
        {'kernel': 3, 'stride': 1},  # Conv4
        {'kernel': 2, 'stride': 2},  # Pool2
        {'kernel': 3, 'stride': 1, 'dilation': 2},  # Conv5 d=2
        {'kernel': 3, 'stride': 1, 'dilation': 4},  # Conv6 d=4
        {'kernel': 3, 'stride': 1, 'dilation': 8},  # Conv7 d=8
    ]
    
    names = ['Conv1', 'Conv2', 'Pool1', 'Conv3', 'Conv4', 
             'Pool2', 'Conv5(d=2)', 'Conv6(d=4)', 'Conv7(d=8)']
    
    print("RF-Aware Network Design Verification")
    print("=" * 60)
    rf, jump = analyze_receptive_field(layers, names)
    print(f"\nFinal receptive field: {rf}×{rf} (target: ~128×128) ✓")
    print(f"Output stride: {jump}")
    
    # Test the network
    model = RFAwareNetwork(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    
    print(f"\nNetwork test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    verify_network_design()
```

## PyTorch Utility: Automatic RF Tracker

```python
import torch
import torch.nn as nn

class ReceptiveFieldTracker(nn.Module):
    """
    Wrapper module that automatically analyzes receptive field 
    of any sequential-style model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rf_info = self._analyze_receptive_field()
    
    def _analyze_receptive_field(self):
        """Analyze receptive field of each conv/pool layer."""
        info = []
        rf, jump = 1, 1
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                # Extract kernel size
                k = module.kernel_size
                k = k[0] if isinstance(k, tuple) else k
                
                # Extract stride
                s = module.stride
                s = s[0] if isinstance(s, tuple) else s
                
                # Extract dilation (default 1 for pooling)
                d = getattr(module, 'dilation', 1)
                d = d[0] if isinstance(d, tuple) else d
                
                # Compute effective kernel and update RF
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
        """Print formatted receptive field analysis."""
        print(f"{'Layer':<30} {'Type':<12} {'K':>3} {'S':>3} {'D':>3} {'RF':>6} {'Jump':>6}")
        print("-" * 75)
        for layer in self.rf_info:
            print(f"{layer['name']:<30} {layer['type']:<12} "
                  f"{layer['kernel']:>3} {layer['stride']:>3} {layer['dilation']:>3} "
                  f"{layer['receptive_field']:>6} {layer['jump']:>6}")
        
        if self.rf_info:
            final = self.rf_info[-1]
            print("-" * 75)
            print(f"Final: RF = {final['receptive_field']}×{final['receptive_field']}, "
                  f"Output stride = {final['jump']}")
    
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
    nn.Conv2d(128, 256, 3, stride=2, padding=1),
    nn.ReLU(),
)

tracker = ReceptiveFieldTracker(model)
tracker.print_receptive_field()
```

## Key Takeaways

1. **Definition**: Receptive field is the input region that influences a neuron's activation

2. **Core formula**: 
   $$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

3. **Cumulative stride amplifies growth**: Early downsampling dramatically increases RF in later layers

4. **Four growth strategies** (ranked by efficiency):
   - Dilated convolutions (most efficient for large RF)
   - Deeper networks with small kernels (preferred for learning)
   - Strided convolutions (aggressive, loses resolution)
   - Larger kernels (costly, rarely used)

5. **Effective RF < Theoretical RF**: Due to Gaussian-like influence distribution

6. **ERF grows with $\sqrt{\text{depth}}$**: Not linearly, so very deep networks may have diminishing returns

7. **Task matching**: Design RF to match expected object/pattern sizes

8. **Skip connections expand ERF**: Residual connections help gradients flow to distant inputs

## Exercises

1. **Manual Calculation**: Calculate the receptive field of VGG-16 at each pooling layer.

2. **Architecture Design**: Design a network with receptive field exactly 101×101 using only 3×3 convolutions and stride-1 (hint: how many layers needed?).

3. **ERF Visualization**: Implement effective receptive field visualization for a pretrained ResNet and compare layers at different depths.

4. **Dilated vs. Deep**: Compare the receptive field growth of (a) 10 standard 3×3 convolutions vs. (b) 5 dilated convolutions with rates 1, 2, 4, 8, 16. Which is more efficient?

5. **Optimal Design**: For an object detection task where objects range from 20×20 to 200×200 pixels, design a feature pyramid where each level has an appropriate receptive field.

## References

1. Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016). Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. *NeurIPS*.

2. Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. *ICLR*.

3. Araujo, A., Norberg, W., Hooker, S., & Weinberger, K. (2019). Computing Receptive Fields of Convolutional Neural Networks. *Distill*.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

5. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.
