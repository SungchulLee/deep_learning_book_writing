# Identity Mapping

## Overview

Identity mappings in deep residual networks, introduced by He et al. in "Identity Mappings in Deep Residual Networks" (2016), represent a crucial refinement of the original ResNet design. By reordering components within residual blocks—placing batch normalization and ReLU *before* convolutions rather than after—the skip connection becomes a true identity mapping, enabling cleaner gradient flow and training of 1000+ layer networks.

This section examines why pure identity shortcuts are optimal, how pre-activation block design achieves this, and the mathematical consequences for gradient propagation.

## Motivation: Why Identity Mappings Matter

The original ResNet paper proposed learning residual functions with the formulation:

$$y_l = h(x_l) + F(x_l, W_l)$$
$$x_{l+1} = f(y_l)$$

where $h(x_l)$ is the skip connection, $F$ is the residual function, and $f$ is the post-addition activation function (ReLU).

### Experimenting with Skip Connection Variants

He et al. systematically tested various forms of $h(x_l)$:

| Shortcut Type | $h(x_l)$ | Training Error |
|---------------|----------|----------------|
| Identity (original) | $x_l$ | Best |
| Scaling (0.5) | $0.5 \cdot x_l$ | Worse |
| Gating | $g(x_l) \cdot x_l$ | Worse |
| 1×1 convolution | $W \cdot x_l$ | Worse |
| Dropout | $\text{dropout}(x_l)$ | Worse |

**Key finding**: Pure identity mapping works best. Any modification to the skip connection—even seemingly beneficial ones like learned gating—degrades performance.

### Mathematical Explanation

When the skip connection is identity, the forward propagation unrolls cleanly:

$$x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$$

Any deep layer is directly expressible as a shallow layer plus accumulated residuals. With a non-identity shortcut $h(x_l) = \lambda x_l$, the unrolling becomes:

$$x_L = \lambda^{L-l} x_l + \sum_{i=l}^{L-1} \lambda^{L-1-i} F_i(x_i)$$

The exponential factor $\lambda^{L-l}$ either amplifies ($\lambda > 1$) or attenuates ($\lambda < 1$) the signal, reintroducing the very gradient flow problems that skip connections were designed to solve.

## The Pre-Activation Insight

### Original ResNet Block (Post-Activation)

```
Input ─────┬─────────────────────────────────────┐
           │                                      │
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │ (identity or projection)
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
         (+)  ◄───────────────────────────────────┘
           ▼
        [ReLU]  ◄─── This ReLU affects the next identity path!
           ▼
        Output
```

**Problem**: The ReLU after addition means the signal passing through the skip connection to the next block is *also* passed through ReLU, breaking the pure identity mapping. The output $x_{l+1} = \text{ReLU}(x_l + F(x_l))$ is always non-negative, meaning the identity path is constrained to $\mathbb{R}_{\geq 0}$.

### Pre-Activation ResNet Block

```
Input ─────┬─────────────────────────────────────┐
           │                                      │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │ (pure identity)
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
         (+)  ◄───────────────────────────────────┘
           ▼
        Output (directly connects to next block's input)
```

**Solution**: By moving BN and ReLU before convolutions, the skip connection becomes a true identity mapping. The signal flows unmodified through consecutive blocks, and the output is simply:

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

where $\hat{f}$ denotes the pre-activation (BN followed by ReLU) applied only within the residual branch.

## Mathematical Analysis

### Information Propagation

With pre-activation, the forward propagation becomes:

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

Unrolling this recursion:

$$x_L = x_l + \sum_{i=l}^{L-1} F(\hat{f}(x_i), W_i)$$

This shows that any deep layer $x_L$ is the sum of the shallow layer $x_l$ and all intermediate residual functions. No multiplicative factors appear—the relationship is purely additive.

### Gradient Flow

The gradient with pre-activation:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F_i\right)$$

The "1" term provides a direct gradient path from $\mathcal{L}$ all the way back to $x_l$, unimpeded by any nonlinearity. In the original (post-activation) ResNet, the gradient must pass through the post-addition ReLU at every block, which can zero out negative gradients. Pre-activation eliminates this bottleneck entirely.

### Comparison of Gradient Paths

| Path | Original ResNet | Pre-Activation ResNet |
|------|-----------------|----------------------|
| Identity gradient | Passes through $L-l$ ReLUs | Direct, unmodified |
| Gradient lower bound | $\prod_{i=l}^{L-1} \mathbb{1}[y_i > 0]$ | Always 1 |
| Can vanish? | Yes (dead ReLU cascades) | No (identity is always active) |

## Implementation

### Pre-Activation Basic Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Type, List


class PreActBasicBlock(nn.Module):
    """
    Pre-Activation Basic Block.
    
    Architecture: BN → ReLU → Conv → BN → ReLU → Conv → Add
    
    Unlike the original BasicBlock where BN/ReLU follow convolutions,
    here they precede convolutions, creating cleaner identity mappings.
    """
    
    expansion: int = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActBasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Pre-activation: BN and ReLU before convolutions
        self.bn1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        
        self.bn2 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        
        # Store for skip connection (after first activation)
        # Note: downsample operates on pre-activated input
        identity = out if self.downsample is None else self.downsample(out)
        
        # First convolution (on pre-activated input)
        out = self.conv1(out)
        
        # Second pre-activation and convolution
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # Skip connection (pure addition, no activation after)
        out = out + identity
        
        return out
```

### Pre-Activation Bottleneck Block

```python
class PreActBottleneck(nn.Module):
    """
    Pre-Activation Bottleneck Block.
    
    Architecture: BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → Add
    
    Expansion factor = 4 (standard for bottleneck blocks).
    """
    
    expansion: int = 4
    
    def __init__(
        self,
        in_channels: int,
        width: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActBottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Calculate actual width
        actual_width = int(width * (base_width / 64.0)) * groups
        
        # Pre-activation for 1×1 reduction
        self.bn1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, actual_width,
            kernel_size=1, bias=False
        )
        
        # Pre-activation for 3×3 processing
        self.bn2 = norm_layer(actual_width)
        self.conv2 = nn.Conv2d(
            actual_width, actual_width,
            kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=False
        )
        
        # Pre-activation for 1×1 expansion
        self.bn3 = norm_layer(actual_width)
        self.conv3 = nn.Conv2d(
            actual_width, width * self.expansion,
            kernel_size=1, bias=False
        )
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First pre-activation
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        
        # Skip connection from pre-activated input
        identity = out if self.downsample is None else self.downsample(out)
        
        # 1×1 reduce
        out = self.conv1(out)
        
        # 3×3 process
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # 1×1 expand
        out = self.bn3(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        
        # Addition (no activation after)
        out = out + identity
        
        return out
```

### Complete Pre-Activation ResNet

```python
class PreActResNet(nn.Module):
    """
    Pre-Activation ResNet.
    
    Key differences from original ResNet:
    1. BN/ReLU moved before convolutions in residual blocks
    2. Final BN before classifier (since last block has no post-activation)
    3. Cleaner gradient flow through identity paths
    """
    
    def __init__(
        self,
        block: Type[PreActBasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial convolution (no BN/ReLU — will be in first block)
        self.conv1 = nn.Conv2d(
            in_channels, self.in_planes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final BN and activation (completes the pre-activation pattern)
        self.bn_final = norm_layer(512 * block.expansion)
        self.relu_final = nn.ReLU(inplace=True)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            # Note: No BN in downsample (pre-activation handles normalization)
            downsample = nn.Conv2d(
                self.in_planes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False
            )
        
        layers = []
        layers.append(block(
            self.in_planes, planes, stride, downsample,
            norm_layer=norm_layer
        ))
        
        self.in_planes = planes * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_planes, planes,
                norm_layer=norm_layer
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final pre-activation before pooling
        x = self.bn_final(x)
        x = self.relu_final(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Factory functions
def preact_resnet18(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes)

def preact_resnet50(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes)

def preact_resnet152(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes)
```

## Experimental Results

### Performance Comparison on CIFAR-10

| Depth | Original ResNet | Pre-Activation ResNet |
|-------|-----------------|----------------------|
| 110 | 6.61% error | 6.37% error |
| 164 | 5.93% error | 5.46% error |
| 1001 | Failed to converge | 4.92% error |

**Key observation**: Pre-activation becomes essential as depth increases. At 1001 layers, only the pre-activation variant converges. The improvement is marginal at moderate depths but decisive at extreme depths.

### Activation Ordering Ablation

He et al. tested several activation orderings:

| Ordering | CIFAR-10 Error (164 layers) |
|----------|-----------------------------|
| Post-activation (original) | 5.93% |
| ReLU-only pre-activation | 5.71% |
| BN-only pre-activation | 5.63% |
| Full pre-activation (BN+ReLU) | **5.46%** |

Both BN and ReLU contribute to the improvement, with the full pre-activation combination being optimal.

## Comparison: Original vs Pre-Activation

| Aspect | Original ResNet | Pre-Activation ResNet |
|--------|-----------------|----------------------|
| BN/ReLU position | After convolution | Before convolution |
| Skip connection | Identity + post-ReLU | Pure identity |
| Final layer output | Activated | Requires final BN+ReLU |
| Gradient path | Through $L$ activations | Direct identity path |
| Very deep (1000+) | Fails to converge | Converges successfully |
| Pre-trained availability | Widely available | Less common |

## When to Use Pre-Activation ResNet

### Recommended Scenarios

1. **Very deep networks (100+ layers)**: Essential for convergence at extreme depths
2. **Research on ultra-deep architectures**: Enables 500–1000+ layer networks
3. **When training stability is critical**: More stable training dynamics
4. **Dense prediction tasks**: Cleaner feature representations for segmentation

### When Original ResNet Suffices

1. **Standard depths (18–50 layers)**: Both variants perform similarly
2. **Using pre-trained weights**: Most pre-trained models use original ResNet ordering
3. **Production deployment**: Better framework support and more pre-trained checkpoints available

## Connection to Transformer Architecture

The pre-activation design directly influenced the dominant "Pre-Norm" Transformer architecture. In Pre-Norm Transformers, LayerNorm is applied *before* the attention and feedforward sublayers, creating the same pure identity skip connection pattern:

$$x_{l+1} = x_l + \text{Attention}(\text{LN}(x_l))$$
$$x_{l+2} = x_{l+1} + \text{FFN}(\text{LN}(x_{l+1}))$$

This connection underscores that the identity mapping principle is architecture-agnostic—it benefits any deep network with residual connections, including the sequence models widely used in quantitative finance for time series modeling and natural language processing of financial text.

## Summary

Pre-Activation ResNet introduces a crucial ordering change:

| Change | Impact |
|--------|--------|
| BN before conv | Normalizes input to each convolution |
| ReLU before conv | Activation applied to normalized features |
| Post-addition identity | Pure gradient flow through shortcuts |
| Final BN before classifier | Completes the pre-activation pattern |

The core principle is that **the skip connection should be an unmodified identity mapping**. Any transformation applied to the shortcut path—whether learned (convolution), fixed (scaling), or nonlinear (ReLU)—degrades gradient flow and training performance. This principle holds across depths, architectures, and domains.

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Xiong, R., Yang, Y., He, J., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. (2020). On Layer Normalization in the Transformer Architecture. *ICML 2020*.
