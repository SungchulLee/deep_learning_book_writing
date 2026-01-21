# Backpropagation in CNNs

## Introduction

**Backpropagation** in Convolutional Neural Networks follows the same chain rule principles as standard neural networks, but requires careful handling of the convolution operation's structure. Understanding CNN backpropagation is essential for debugging gradient issues, implementing custom layers, and optimizing training.

## Forward Pass Review

For a convolutional layer with input $X$, weights $W$, and bias $b$:

$$Y = W * X + b$$

where $*$ denotes cross-correlation (what deep learning calls "convolution").

## Gradient Derivations

### Gradient with Respect to Bias

The simplest gradient—bias adds equally to all spatial positions:

$$\frac{\partial L}{\partial b_k} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}}$$

Sum over all spatial positions of the gradient for each output channel.

### Gradient with Respect to Weights

$$\frac{\partial L}{\partial W_{k,c,m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}} \cdot X_{c, i+m, j+n}$$

This is a **correlation** between the input and the output gradient.

### Gradient with Respect to Input

$$\frac{\partial L}{\partial X_{c,i,j}} = \sum_{k,m,n} \frac{\partial L}{\partial Y_{k, i-m, j-n}} \cdot W_{k,c,m,n}$$

This is a **full convolution** of the output gradient with the **flipped** weights—equivalent to **transposed convolution**.

## Key Insight: Transposed Convolution

The backward pass through convolution w.r.t. input is a transposed convolution:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Forward: conv2d
x = torch.randn(1, 3, 8, 8, requires_grad=True)
w = torch.randn(16, 3, 3, 3, requires_grad=True)
y = F.conv2d(x, w, padding=1)

# Backward: equivalent to conv_transpose2d
grad_output = torch.randn_like(y)
y.backward(grad_output)

# Manual verification
grad_input_manual = F.conv_transpose2d(grad_output, w, padding=1)
print(f"Gradient match: {torch.allclose(x.grad, grad_input_manual, atol=1e-5)}")
```

## PyTorch Implementation

### Complete Custom Conv2d with Backward

```python
class Conv2dManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        return F.conv2d(input, weight, bias, stride, padding)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding
        
        # Gradient w.r.t. input: transposed convolution
        grad_input = F.conv_transpose2d(grad_output, weight, 
                                        stride=stride, padding=padding)
        
        # Gradient w.r.t. weight: correlation of input and grad_output
        grad_weight = F.conv2d(
            input.transpose(0, 1),
            grad_output.transpose(0, 1),
            padding=padding
        ).transpose(0, 1)
        
        # Gradient w.r.t. bias: sum over batch and spatial dims
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None
        
        return grad_input, grad_weight, grad_bias, None, None
```

## Gradient Flow Visualization

```python
def visualize_gradients(model, x, target_layer_name):
    """Visualize gradient magnitudes through a CNN."""
    gradients = {}
    
    def save_grad(name):
        def hook(grad):
            gradients[name] = grad.abs().mean().item()
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.register_hook(save_grad(name))
    
    # Forward + backward
    y = model(x)
    y.sum().backward()
    
    return gradients
```

## Common Issues

### Vanishing Gradients in Deep CNNs

As gradients propagate through many layers, they can vanish. Solutions:

1. **Residual connections**: $y = F(x) + x$
2. **Batch normalization**: Stabilizes gradient magnitudes
3. **Careful initialization**: He/Xavier initialization

### Gradient Checking

```python
def gradient_check(layer, x, eps=1e-5):
    """Numerical gradient verification."""
    x = x.clone().requires_grad_(True)
    y = layer(x)
    y.sum().backward()
    analytic_grad = x.grad.clone()
    
    # Numerical gradient
    numeric_grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone().flatten()
        x_plus[i] += eps
        x_minus = x.clone().flatten()
        x_minus[i] -= eps
        
        y_plus = layer(x_plus.view_as(x)).sum()
        y_minus = layer(x_minus.view_as(x)).sum()
        numeric_grad.flatten()[i] = (y_plus - y_minus) / (2 * eps)
    
    diff = (analytic_grad - numeric_grad).abs().max()
    print(f"Max gradient difference: {diff:.2e}")
    return diff < 1e-4
```

## Pooling Layer Gradients

### Max Pooling

Gradient flows only through the maximum element:

```python
# Max pooling stores indices for backward pass
pool = nn.MaxPool2d(2, return_indices=True)
x = torch.randn(1, 1, 4, 4, requires_grad=True)
y, indices = pool(x)

# Gradient is sparse: only max positions receive gradient
y.sum().backward()
print(f"Non-zero gradients: {(x.grad != 0).sum().item()}")  # = number of output elements
```

### Average Pooling

Gradient distributed equally:

```python
# Each input receives gradient / (pool_size^2)
pool = nn.AvgPool2d(2)
x = torch.randn(1, 1, 4, 4, requires_grad=True)
y = pool(x)
y.sum().backward()
print(f"Gradient values: {x.grad.unique()}")  # All equal to 0.25 (1/4)
```

## Summary

| Component | Forward | Backward (w.r.t. input) |
|-----------|---------|------------------------|
| Conv2d | Cross-correlation | Transposed convolution |
| MaxPool | Select maximum | Route gradient to max position |
| AvgPool | Compute mean | Distribute gradient equally |

## References

1. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning.
2. He, K., et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.
