# Weight Normalization

## Overview

Weight Normalization, introduced by Salimans and Kingma in 2016, is a reparameterization technique that decouples the magnitude (norm) of weight vectors from their direction. Unlike Batch or Layer Normalization, Weight Normalization operates on the network weights directly rather than normalizing activations, making it computationally simpler and independent of batch statistics.

## Mathematical Formulation

### Weight Reparameterization

For a weight vector $\mathbf{w}$, Weight Normalization expresses it as:

$$\mathbf{w} = g \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

Where:
- $\mathbf{v}$ is an unnormalized weight vector (learnable)
- $g$ is a scalar magnitude parameter (learnable)
- $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$ is the Euclidean norm

This decouples the **magnitude** $g$ from the **direction** $\frac{\mathbf{v}}{\|\mathbf{v}\|}$.

### For Neural Network Layers

For a layer $y = \phi(\mathbf{w} \cdot \mathbf{x} + b)$, the reparameterized forward pass becomes:

$$y = \phi\left(g \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|} \cdot \mathbf{x} + b\right)$$

### For Convolutional Layers

For a convolutional kernel $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$, Weight Normalization is applied per output channel:

$$\mathbf{W}_{c} = g_c \cdot \frac{\mathbf{V}_{c}}{\|\mathbf{V}_{c}\|}$$

Where $\mathbf{V}_{c} \in \mathbb{R}^{C_{in} \times k \times k}$ and $g_c$ is a scalar for each output channel.

## PyTorch Implementation

### From Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightNormalizedLinear(nn.Module):
    """Linear layer with Weight Normalization."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Direction parameter (unnormalized weights)
        self.v = nn.Parameter(torch.randn(out_features, in_features))
        
        # Magnitude parameter (one per output feature)
        self.g = nn.Parameter(torch.ones(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.normal_(self.v, mean=0, std=0.05)
        # Initialize g to make ||w|| = 1 initially
        with torch.no_grad():
            self.g.copy_(torch.norm(self.v, dim=1))
    
    def forward(self, x):
        # Compute normalized weight
        v_norm = torch.norm(self.v, dim=1, keepdim=True)
        w = self.g.unsqueeze(1) * self.v / v_norm
        
        return F.linear(x, w, self.bias)


class WeightNormalizedConv2d(nn.Module):
    """2D Convolution with Weight Normalization."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Direction parameter
        self.v = nn.Parameter(torch.randn(
            out_channels, in_channels, *self.kernel_size
        ))
        
        # Magnitude parameter (one per output channel)
        self.g = nn.Parameter(torch.ones(out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.v)
        with torch.no_grad():
            v_flat = self.v.view(self.out_channels, -1)
            self.g.copy_(torch.norm(v_flat, dim=1))
    
    def forward(self, x):
        # Flatten spatial dimensions for norm computation
        v_flat = self.v.view(self.out_channels, -1)
        v_norm = torch.norm(v_flat, dim=1, keepdim=True)
        
        # Compute normalized weight
        w = (self.g.view(-1, 1) / v_norm) * v_flat
        w = w.view_as(self.v)
        
        return F.conv2d(x, w, self.bias, self.stride, self.padding)
```

### Using PyTorch Built-in

PyTorch provides `torch.nn.utils.weight_norm` as a function to wrap existing layers:

```python
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm

# Apply weight normalization to a linear layer
linear = nn.Linear(256, 128)
linear_wn = weight_norm(linear, name='weight')

# Apply to convolutional layer
conv = nn.Conv2d(64, 128, 3, padding=1)
conv_wn = weight_norm(conv, name='weight')

# Inspect the reparameterization
print(f"Parameters: {list(linear_wn.named_parameters())}")
# Output: [('weight_g', ...), ('weight_v', ...), ('bias', ...)]

# Remove weight normalization (fuse back to regular weight)
linear_regular = remove_weight_norm(linear_wn)
```

### Important: Dimension Selection

```python
# Default: normalizes over all dimensions except first (dim=0)
# For Linear: normalizes each row of the weight matrix
conv_wn = weight_norm(nn.Conv2d(64, 128, 3), dim=0)

# Alternative: normalize over different dimension
# For Conv2d, dim=0 means per output channel (standard)
# dim=1 would mean per input channel (unusual)
conv_wn_dim1 = weight_norm(nn.Conv2d(64, 128, 3), dim=1)
```

## Gradient Computation

### Gradient with Respect to g

$$\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial \mathbf{w}} \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

The gradient w.r.t. $g$ is simply the dot product of the weight gradient with the direction.

### Gradient with Respect to v

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}} = \frac{g}{\|\mathbf{v}\|} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{w}} - \frac{g}{\|\mathbf{v}\|^2} \frac{\partial g}{\partial \mathbf{v}} \mathbf{v} \right)$$

$$= \frac{g}{\|\mathbf{v}\|} \left( \frac{\partial \mathcal{L}}{\partial \mathbf{w}} - \frac{\partial \mathcal{L}}{\partial g} \frac{\mathbf{v}}{\|\mathbf{v}\|} \right)$$

The gradient for $\mathbf{v}$ has a projection removed, ensuring updates to $\mathbf{v}$ primarily affect direction, not magnitude.

### Implementation of Backward Pass

```python
class WeightNormFunction(torch.autograd.Function):
    """Custom autograd for Weight Normalization."""
    
    @staticmethod
    def forward(ctx, v, g):
        # Compute normalized weight
        v_norm = torch.norm(v.view(v.size(0), -1), dim=1, keepdim=True)
        v_norm = v_norm.view(-1, *([1] * (v.dim() - 1)))
        
        w = g.view(-1, *([1] * (v.dim() - 1))) * v / v_norm
        
        ctx.save_for_backward(v, g, v_norm)
        return w
    
    @staticmethod
    def backward(ctx, grad_w):
        v, g, v_norm = ctx.saved_tensors
        
        # Flatten for computation
        grad_w_flat = grad_w.view(grad_w.size(0), -1)
        v_flat = v.view(v.size(0), -1)
        v_norm_flat = v_norm.view(-1, 1)
        g_flat = g.view(-1, 1)
        
        # Gradient w.r.t. g
        v_normalized = v_flat / v_norm_flat
        grad_g = (grad_w_flat * v_normalized).sum(dim=1)
        
        # Gradient w.r.t. v
        grad_v = (g_flat / v_norm_flat) * (
            grad_w_flat - grad_g.view(-1, 1) * v_normalized
        )
        grad_v = grad_v.view_as(v)
        
        return grad_v, grad_g
```

## Benefits of Weight Normalization

### 1. Decoupled Learning Dynamics

The gradient updates for $g$ and $\mathbf{v}$ are partially decoupled:

```python
def visualize_decoupling():
    """Visualize how g and v are updated independently."""
    
    torch.manual_seed(42)
    
    layer = WeightNormalizedLinear(10, 5)
    x = torch.randn(32, 10)
    target = torch.randn(32, 5)
    
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    
    print("Training dynamics with Weight Normalization:")
    print("=" * 50)
    
    for step in range(5):
        # Record initial values
        g_before = layer.g.clone()
        v_norm_before = torch.norm(layer.v, dim=1)
        
        # Forward and backward
        optimizer.zero_grad()
        loss = F.mse_loss(layer(x), target)
        loss.backward()
        
        # Check gradients
        g_grad = layer.g.grad.norm().item()
        v_grad = layer.v.grad.norm().item()
        
        optimizer.step()
        
        # Record changes
        g_change = (layer.g - g_before).norm().item()
        v_norm_after = torch.norm(layer.v, dim=1)
        v_direction_change = (v_norm_after / v_norm_before - 1).abs().mean().item()
        
        print(f"Step {step}: loss={loss:.4f}, |∇g|={g_grad:.4f}, |∇v|={v_grad:.4f}")

visualize_decoupling()
```

### 2. No Batch Dependence

```python
def demonstrate_batch_independence():
    """Show that Weight Norm doesn't depend on batch statistics."""
    
    layer = WeightNormalizedLinear(64, 32)
    
    # Same input, different batch contexts
    x_single = torch.randn(1, 64)
    
    outputs = []
    for batch_size in [1, 2, 8, 32]:
        x_batch = torch.randn(batch_size, 64)
        x_batch[0] = x_single[0]
        
        out = layer(x_batch)
        outputs.append(out[0].clone())
    
    print("Weight Norm outputs for same input in different batch sizes:")
    for i, (bs, out) in enumerate(zip([1, 2, 8, 32], outputs)):
        print(f"  Batch size {bs:2d}: {out[:5].tolist()}")
    
    # Check they're identical
    max_diff = max((outputs[0] - out).abs().max().item() for out in outputs[1:])
    print(f"\nMax difference: {max_diff:.2e} (should be ~0)")

demonstrate_batch_independence()
```

### 3. Faster Convergence

Weight Normalization can lead to faster convergence by:
- Better conditioning of the optimization landscape
- More stable gradient magnitudes
- Natural learning rate scaling per weight vector

## Comparison with Batch Normalization

| Aspect | Weight Normalization | Batch Normalization |
|--------|---------------------|---------------------|
| Operates on | Weights | Activations |
| Batch dependent | No | Yes |
| Running statistics | No | Yes |
| Train/eval difference | No | Yes |
| Computational cost | Lower | Higher |
| Regularization effect | Minimal | Yes (batch noise) |
| Works with batch=1 | Yes | Problematic |

### Side-by-side Comparison

```python
def compare_wn_bn():
    """Compare Weight Normalization with Batch Normalization."""
    
    torch.manual_seed(42)
    
    # Networks
    net_wn = nn.Sequential(
        weight_norm(nn.Linear(784, 256)),
        nn.ReLU(),
        weight_norm(nn.Linear(256, 10))
    )
    
    net_bn = nn.Sequential(
        nn.Linear(784, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 16, 64]
    
    print("Output statistics by batch size:")
    print("=" * 60)
    
    for bs in batch_sizes:
        x = torch.randn(bs, 784)
        
        # Weight Norm (same behavior always)
        out_wn = net_wn(x)
        
        # Batch Norm (varies with batch size)
        net_bn.train()
        out_bn = net_bn(x)
        
        print(f"Batch size {bs:2d}:")
        print(f"  WN: mean={out_wn.mean():.4f}, std={out_wn.std():.4f}")
        print(f"  BN: mean={out_bn.mean():.4f}, std={out_bn.std():.4f}")

compare_wn_bn()
```

## Mean-Only Batch Normalization Combination

Weight Normalization is often combined with "mean-only" batch normalization for better results:

```python
class WeightNormWithMeanOnlyBN(nn.Module):
    """
    Weight Normalization combined with mean-only Batch Normalization.
    This combination was found effective in the original paper.
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.linear = weight_norm(nn.Linear(in_features, out_features, bias=False))
        
        # Mean-only batch normalization (no variance scaling)
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))  # Bias/shift
        self.momentum = 0.1
    
    def forward(self, x):
        # Apply weight-normalized linear
        out = self.linear(x)
        
        if self.training:
            # Compute batch mean
            mean = out.mean(dim=0)
            
            # Update running mean
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            
            # Center using batch mean
            out = out - mean
        else:
            # Center using running mean
            out = out - self.running_mean
        
        # Add learnable bias
        out = out + self.beta
        
        return out
```

## Network Architectures with Weight Normalization

### WaveNet-style Network

Weight Normalization was notably used in WaveNet for audio generation:

```python
class WaveNetBlock(nn.Module):
    """WaveNet residual block with Weight Normalization."""
    
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        
        self.dilated_conv = weight_norm(
            nn.Conv1d(channels, 2 * channels, kernel_size,
                     padding=(kernel_size - 1) * dilation // 2,
                     dilation=dilation)
        )
        
        self.residual_conv = weight_norm(
            nn.Conv1d(channels, channels, 1)
        )
        
        self.skip_conv = weight_norm(
            nn.Conv1d(channels, channels, 1)
        )
    
    def forward(self, x):
        # Gated activation
        conv_out = self.dilated_conv(x)
        tanh_out, sigmoid_out = conv_out.chunk(2, dim=1)
        gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
        
        # Residual and skip connections
        residual = self.residual_conv(gated) + x
        skip = self.skip_conv(gated)
        
        return residual, skip
```

### Simple Classifier

```python
class WeightNormClassifier(nn.Module):
    """Image classifier using Weight Normalization."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            weight_norm(nn.Conv2d(3, 64, 3, padding=1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            weight_norm(nn.Conv2d(64, 128, 3, padding=1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            weight_norm(nn.Conv2d(128, 256, 3, padding=1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            weight_norm(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Linear(128, num_classes))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## When to Use Weight Normalization

### Good Use Cases

✅ **Generative models** (WaveNet, PixelCNN)  
✅ **When batch normalization is problematic** (small batches, online learning)  
✅ **Reinforcement learning** (varying batch sizes, off-policy)  
✅ **Simple architectures** where full batch norm overhead isn't needed  
✅ **Recurrent networks** (can be simpler than layer norm)

### Limitations

❌ Less regularization effect than BatchNorm  
❌ May underperform BatchNorm on large-batch image classification  
❌ Doesn't provide the same activation normalization benefits

## Summary

Weight Normalization is a lightweight normalization technique that:

1. **Reparameterizes weights** as magnitude × direction
2. **Decouples** learning of weight magnitude and direction
3. **Has no batch dependence** - same behavior regardless of batch size
4. **Lower computational overhead** than activation-based normalizations

Key properties:
- Operates on **weights**, not activations
- **No running statistics** needed
- **Same train/eval behavior**
- Often combined with **mean-only batch norm** for best results

## References

1. Salimans, T., & Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. *NeurIPS*.

2. van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv preprint arXiv:1609.03499*.
