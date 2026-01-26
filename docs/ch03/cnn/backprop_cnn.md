# Backpropagation in CNNs

## Introduction

**Backpropagation** in Convolutional Neural Networks follows the same chain rule principles as standard neural networks, but requires careful handling of the convolution operation's structure. Understanding CNN backpropagation is essential for:

- Debugging gradient issues during training
- Implementing custom layers
- Architectural innovation
- Optimizing training performance

## Review: General Backpropagation

For a loss function $L$ and layer output $\mathbf{y} = f(\mathbf{x}; \mathbf{w})$:

**Chain Rule**:
$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{w}}$$

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

## Forward Pass Review

For a convolutional layer with input $X$, weights $W$, and bias $b$:

$$Y = W * X + b$$

where $*$ denotes cross-correlation (what deep learning calls "convolution").

---

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

---

## 1D Convolution Gradients (Detailed Derivation)

### Forward Pass

For input $\mathbf{x} \in \mathbb{R}^n$ and kernel $\mathbf{w} \in \mathbb{R}^k$:

$$y_i = \sum_{j=0}^{k-1} x_{i+j} \cdot w_j$$

### Gradient with Respect to Input

$$\frac{\partial L}{\partial x_i} = \sum_{j} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

Position $x_i$ affects outputs $y_{i-k+1}$ through $y_i$ (where valid):

$$\frac{\partial L}{\partial x_i} = \sum_{j=\max(0, i-k+1)}^{\min(i, n-k)} \frac{\partial L}{\partial y_j} \cdot w_{i-j}$$

This is equivalent to **full convolution** of the gradient with a **flipped kernel**:

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} *_{full} \text{flip}(\mathbf{w})$$

### Gradient with Respect to Kernel

$$\frac{\partial L}{\partial w_j} = \sum_{i} \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial w_j} = \sum_{i} \frac{\partial L}{\partial y_i} \cdot x_{i+j}$$

This is the **cross-correlation** of input with output gradient:

$$\frac{\partial L}{\partial \mathbf{w}} = \mathbf{x} \star \frac{\partial L}{\partial \mathbf{y}}$$

### Python Implementation

```python
import numpy as np

def conv1d_forward(x, w):
    """1D convolution (cross-correlation) forward pass."""
    n, k = len(x), len(w)
    out_len = n - k + 1
    y = np.zeros(out_len)
    for i in range(out_len):
        y[i] = np.sum(x[i:i+k] * w)
    return y

def conv1d_backward(x, w, grad_output):
    """
    1D convolution backward pass.
    
    Args:
        x: Input array
        w: Kernel weights
        grad_output: Gradient from subsequent layer (dL/dy)
    
    Returns:
        grad_x: Gradient with respect to input (dL/dx)
        grad_w: Gradient with respect to weights (dL/dw)
    """
    n, k = len(x), len(w)
    out_len = len(grad_output)
    
    # Gradient w.r.t. input: full convolution with flipped kernel
    grad_x = np.zeros(n)
    w_flip = w[::-1]
    
    # Pad grad_output for full convolution
    grad_padded = np.pad(grad_output, (k-1, k-1), mode='constant')
    for i in range(n):
        grad_x[i] = np.sum(grad_padded[i:i+k] * w_flip)
    
    # Gradient w.r.t. weights: correlation of input with grad_output
    grad_w = np.zeros(k)
    for j in range(k):
        grad_w[j] = np.sum(x[j:j+out_len] * grad_output)
    
    return grad_x, grad_w


# Test with numerical gradient verification
np.random.seed(42)
x = np.random.randn(8)
w = np.random.randn(3)

y = conv1d_forward(x, w)
grad_output = np.random.randn(len(y))

grad_x, grad_w = conv1d_backward(x, w, grad_output)

# Verify grad_w numerically
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def loss_wrt_w(w_):
    return np.sum(conv1d_forward(x, w_) * grad_output)

grad_w_numerical = numerical_gradient(loss_wrt_w, w)

print("Analytical grad_w:", grad_w)
print("Numerical grad_w: ", grad_w_numerical)
print("Match:", np.allclose(grad_w, grad_w_numerical))
```

---

## 2D Convolution Gradients

### Forward Pass

For input $\mathbf{X} \in \mathbb{R}^{H \times W}$ and kernel $\mathbf{W} \in \mathbb{R}^{k \times k}$:

$$Y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i+m, j+n} \cdot W_{m,n}$$

### Gradient with Respect to Input

$$\frac{\partial L}{\partial X_{i,j}} = \sum_{m,n} \frac{\partial L}{\partial Y_{i-m, j-n}} \cdot W_{m,n}$$

This is **full 2D convolution** with **180° rotated kernel**:

$$\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} *_{full} \text{rot180}(\mathbf{W})$$

### Gradient with Respect to Kernel

$$\frac{\partial L}{\partial W_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j}} \cdot X_{i+m, j+n}$$

This is **2D cross-correlation** of input with output gradient:

$$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X} \star \frac{\partial L}{\partial \mathbf{Y}}$$

### Python Implementation

```python
import numpy as np

def conv2d_forward(X, W, padding=0, stride=1):
    """2D convolution forward pass."""
    H, W_in = X.shape
    k_h, k_w = W.shape
    
    if padding > 0:
        X = np.pad(X, padding, mode='constant')
        H, W_in = X.shape
    
    H_out = (H - k_h) // stride + 1
    W_out = (W_in - k_w) // stride + 1
    
    Y = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            region = X[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            Y[i, j] = np.sum(region * W)
    
    return Y

def conv2d_backward(X, W, grad_output, padding=0, stride=1):
    """
    2D convolution backward pass.
    
    Args:
        X: Original input
        W: Kernel weights
        grad_output: Gradient from next layer (dL/dY)
        padding: Original padding used
        stride: Original stride used
    
    Returns:
        grad_X: Gradient with respect to input
        grad_W: Gradient with respect to weights
    """
    H_orig, W_orig = X.shape
    k_h, k_w = W.shape
    
    if padding > 0:
        X_padded = np.pad(X, padding, mode='constant')
    else:
        X_padded = X
    
    H, W_in = X_padded.shape
    H_out, W_out = grad_output.shape
    
    # Gradient w.r.t. weights
    grad_W = np.zeros_like(W)
    for m in range(k_h):
        for n in range(k_w):
            for i in range(H_out):
                for j in range(W_out):
                    grad_W[m, n] += grad_output[i, j] * X_padded[i*stride+m, j*stride+n]
    
    # Gradient w.r.t. input (full convolution with rotated kernel)
    W_rot = np.rot90(W, 2)  # 180° rotation
    
    # For stride > 1, need to "dilate" grad_output
    if stride > 1:
        dilated_H = (H_out - 1) * stride + 1
        dilated_W = (W_out - 1) * stride + 1
        grad_dilated = np.zeros((dilated_H, dilated_W))
        grad_dilated[::stride, ::stride] = grad_output
        grad_output = grad_dilated
    
    # Full convolution
    pad_h, pad_w = k_h - 1, k_w - 1
    grad_padded = np.pad(grad_output, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    grad_X_padded = np.zeros_like(X_padded)
    for i in range(H):
        for j in range(W_in):
            region = grad_padded[i:i+k_h, j:j+k_w]
            grad_X_padded[i, j] = np.sum(region * W_rot)
    
    # Remove padding from gradient
    if padding > 0:
        grad_X = grad_X_padded[padding:-padding, padding:-padding]
    else:
        grad_X = grad_X_padded
    
    return grad_X, grad_W
```

---

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

---

## Multi-Channel Convolution Gradients

### Forward Pass

For input $\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W}$, weights $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$:

$$Y_{o,i,j} = \sum_{c=0}^{C_{in}-1} \sum_{m,n} X_{c,i+m,j+n} \cdot W_{o,c,m,n} + b_o$$

### Gradients

**Gradient w.r.t. Input**:
$$\frac{\partial L}{\partial X_{c,i,j}} = \sum_{o=0}^{C_{out}-1} \sum_{m,n} \frac{\partial L}{\partial Y_{o,i-m,j-n}} \cdot W_{o,c,m,n}$$

**Gradient w.r.t. Weights**:
$$\frac{\partial L}{\partial W_{o,c,m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{o,i,j}} \cdot X_{c,i+m,j+n}$$

**Gradient w.r.t. Bias**:
$$\frac{\partial L}{\partial b_o} = \sum_{i,j} \frac{\partial L}{\partial Y_{o,i,j}}$$

### PyTorch Verification

```python
import torch
import torch.nn as nn

# Create conv layer
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)

# Input with gradient tracking
x = torch.randn(2, 3, 8, 8, requires_grad=True)

# Forward
y = conv(x)

# Create upstream gradient
grad_output = torch.randn_like(y)

# Backward
y.backward(grad_output)

# Access gradients
print(f"Input gradient shape: {x.grad.shape}")           # [2, 3, 8, 8]
print(f"Weight gradient shape: {conv.weight.grad.shape}")  # [16, 3, 3, 3]
print(f"Bias gradient shape: {conv.bias.grad.shape}")      # [16]

# Verify bias gradient: sum over batch, height, width
expected_bias_grad = grad_output.sum(dim=(0, 2, 3))
print(f"Bias grad match: {torch.allclose(conv.bias.grad, expected_bias_grad)}")
```

---

## Custom Autograd Function

### Complete Custom Conv2d with Backward

```python
import torch
import torch.nn.functional as F

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

---

## Pooling Layer Gradients

### Max Pooling

**Forward**: $y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}$

**Backward**: Gradient flows only through the maximum element:

$$\frac{\partial L}{\partial x_{m,n}} = \begin{cases}
\frac{\partial L}{\partial y_{i,j}} & \text{if } x_{m,n} = y_{i,j} \\
0 & \text{otherwise}
\end{cases}$$

```python
import torch
import torch.nn as nn

# Max pooling stores indices for backward pass
pool = nn.MaxPool2d(2, return_indices=True)
x = torch.randn(1, 1, 4, 4, requires_grad=True)
y, indices = pool(x)

# Gradient is sparse: only max positions receive gradient
y.sum().backward()
print(f"Non-zero gradients: {(x.grad != 0).sum().item()}")  # = number of output elements
```

### Average Pooling

**Forward**: $y_{i,j} = \frac{1}{k^2} \sum_{(m,n) \in R_{i,j}} x_{m,n}$

**Backward**: Gradient distributed equally:

$$\frac{\partial L}{\partial x_{m,n}} = \frac{1}{k^2} \frac{\partial L}{\partial y_{i,j}}$$

```python
# Each input receives gradient / (pool_size^2)
pool = nn.AvgPool2d(2)
x = torch.randn(1, 1, 4, 4, requires_grad=True)
y = pool(x)
y.sum().backward()
print(f"Gradient values: {x.grad.unique()}")  # All equal to 0.25 (1/4)
```

### NumPy Implementations

```python
import numpy as np

def maxpool2d_forward(X, pool_size=2, stride=2):
    """Max pooling forward pass with index tracking."""
    H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    Y = np.zeros((H_out, W_out))
    indices = np.zeros((H_out, W_out, 2), dtype=int)
    
    for i in range(H_out):
        for j in range(W_out):
            region = X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            max_idx = np.unravel_index(np.argmax(region), region.shape)
            Y[i, j] = region[max_idx]
            indices[i, j] = [i*stride + max_idx[0], j*stride + max_idx[1]]
    
    return Y, indices

def maxpool2d_backward(grad_output, indices, input_shape):
    """Max pooling backward pass."""
    grad_X = np.zeros(input_shape)
    H_out, W_out = grad_output.shape
    
    for i in range(H_out):
        for j in range(W_out):
            max_i, max_j = indices[i, j]
            grad_X[max_i, max_j] += grad_output[i, j]
    
    return grad_X

def avgpool2d_backward(grad_output, input_shape, pool_size=2, stride=2):
    """Average pooling backward pass."""
    grad_X = np.zeros(input_shape)
    H_out, W_out = grad_output.shape
    
    for i in range(H_out):
        for j in range(W_out):
            grad_X[i*stride:i*stride+pool_size, 
                   j*stride:j*stride+pool_size] += grad_output[i, j] / (pool_size ** 2)
    
    return grad_X
```

---

## Gradient Flow Analysis

### Visualizing Gradients Through a CNN

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Simple CNN for demonstrating backpropagation."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def visualize_gradients(model, x, target):
    """Visualize gradient magnitudes through a CNN."""
    gradients = {}
    
    def save_grad(name):
        def hook(grad):
            gradients[name] = grad.abs().mean().item()
        return hook
    
    # Register hooks on weights
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.register_hook(save_grad(name))
    
    # Forward + backward
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    
    # Print gradient statistics
    print("Layer-wise Gradient Magnitudes:")
    print("-" * 40)
    for name, grad_mag in gradients.items():
        print(f"{name:15s}: {grad_mag:.6f}")
    
    return gradients


# Example usage
model = SimpleCNN()
x = torch.randn(4, 1, 28, 28)
target = torch.randint(0, 10, (4,))

visualize_gradients(model, x, target)
```

---

## Common Issues and Solutions

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
    x_flat = x.flatten()
    
    for i in range(min(100, x.numel())):  # Check first 100 elements
        x_plus = x.clone().flatten()
        x_minus = x.clone().flatten()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        y_plus = layer(x_plus.view_as(x)).sum()
        y_minus = layer(x_minus.view_as(x)).sum()
        numeric_grad.flatten()[i] = (y_plus - y_minus) / (2 * eps)
    
    diff = (analytic_grad.flatten()[:100] - numeric_grad.flatten()[:100]).abs().max()
    print(f"Max gradient difference: {diff:.2e}")
    return diff < 1e-4
```

---

## Transposed Convolution for Upsampling

Transposed convolution is the gradient operation of convolution, commonly used in decoder networks:

```python
import torch
import torch.nn as nn

# Regular convolution (downsample)
conv = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)

# Transposed convolution (upsample)
conv_transpose = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

x = torch.randn(1, 64, 32, 32)

# Downsample
y = conv(x)
print(f"Conv: {x.shape} → {y.shape}")  # [1, 64, 32, 32] → [1, 32, 16, 16]

# Upsample
z = conv_transpose(y)
print(f"ConvT: {y.shape} → {z.shape}")  # [1, 32, 16, 16] → [1, 64, 32, 32]
```

---

## Summary of CNN Gradients

| Operation | Forward | Backward (w.r.t. input) | Backward (w.r.t. weight) |
|-----------|---------|------------------------|-------------------------|
| Conv2d | Cross-correlation | Transposed convolution | Correlation with grad |
| MaxPool | Select maximum | Route gradient to max position | N/A |
| AvgPool | Compute mean | Distribute gradient equally | N/A |
| ReLU | $\max(0, x)$ | $\mathbb{1}_{x > 0} \cdot \nabla$ | N/A |

## Key Takeaways

1. **Conv gradient w.r.t. input**: Full convolution with 180° rotated kernel (= transposed convolution)
2. **Conv gradient w.r.t. weights**: Cross-correlation of input with output gradient
3. **Conv gradient w.r.t. bias**: Sum over all spatial positions and batch
4. **Max pooling**: Sparse gradient—flows only to maximum positions
5. **Average pooling**: Uniform gradient distribution ($1/k^2$ to each position)
6. **Transposed convolution** is the gradient operation of regular convolution
7. Understanding gradients helps debug training issues and design custom layers

## References

1. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning."
2. Goodfellow, I., et al. (2016). "Deep Learning." Chapter 9: Convolutional Networks.
3. He, K., et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance."
4. Long, J., et al. (2015). "Fully Convolutional Networks for Semantic Segmentation."
