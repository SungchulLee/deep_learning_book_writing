# 1D Convolutions

## Introduction

While 2D convolutions are the workhorse of image processing, **1D convolutions** operate on sequential data—time series, audio signals, text sequences, and financial data. They apply a learnable kernel that slides along a single spatial (temporal) dimension, extracting local patterns at every position.

1D convolutions are foundational to architectures like WaveNet, Temporal Convolutional Networks (TCN), and are widely used in quantitative finance for processing price sequences, order book snapshots, and other temporal signals.

---

## Mathematical Formulation

### Single-Channel 1D Convolution

For a 1D input $\mathbf{x} \in \mathbb{R}^{n}$ and kernel $\mathbf{w} \in \mathbb{R}^{k}$:

$$y[i] = \sum_{j=0}^{k-1} x[i + j] \cdot w[j]$$

The output size (without padding) is $n - k + 1$.

### Example

Consider input $\mathbf{x} = [1, 2, 3, 4, 5]$ and kernel $\mathbf{w} = [1, 0, -1]$:

```
Position 0: 1×1 + 2×0 + 3×(-1) = 1 - 3 = -2
Position 1: 2×1 + 3×0 + 4×(-1) = 2 - 4 = -2
Position 2: 3×1 + 4×0 + 5×(-1) = 3 - 5 = -2

Output: [-2, -2, -2]
```

This kernel computes a discrete derivative (difference), detecting changes in the signal.

### Multi-Channel Formulation

For an input with $C_{in}$ channels and producing $C_{out}$ channels:

$$Y[o, i] = \sum_{c=0}^{C_{in}-1} \sum_{j=0}^{k-1} X[c, i+j] \cdot W[o, c, j] + b[o]$$

The weight tensor has shape $W \in \mathbb{R}^{C_{out} \times C_{in} \times k}$.

---

## PyTorch `nn.Conv1d`

### Interface

```python
import torch
import torch.nn as nn

# Conv1d signature
conv1d = nn.Conv1d(
    in_channels,    # Number of input channels
    out_channels,   # Number of output channels (filters)
    kernel_size,    # Size of the convolving kernel
    stride=1,       # Stride of the convolution
    padding=0,      # Zero-padding added to both sides
    dilation=1,     # Spacing between kernel elements
    groups=1,       # Number of blocked connections
    bias=True,      # Add learnable bias
    padding_mode='zeros'  # 'zeros', 'reflect', 'replicate', 'circular'
)
```

**Input shape**: $(N, C_{in}, L)$ — batch size, input channels, sequence length

**Output shape**: $(N, C_{out}, L_{out})$ where $L_{out} = \left\lfloor \frac{L + 2p - d(k-1) - 1}{s} \right\rfloor + 1$

### Basic Examples

```python
import torch
import torch.nn as nn

# 1D Convolution Example
# Input: (batch_size, in_channels, length)
x = torch.tensor([[[1., 2., 3., 4., 5.]]])  # Shape: (1, 1, 5)

# Create 1D conv layer: 1 input channel, 1 output channel, kernel size 3
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# Manually set weights to [1, 0, -1] (edge detection kernel)
with torch.no_grad():
    conv1d.weight = nn.Parameter(torch.tensor([[[1., 0., -1.]]]))

output = conv1d(x)
print(f"Input shape: {x.shape}")      # torch.Size([1, 1, 5])
print(f"Output shape: {output.shape}") # torch.Size([1, 1, 3])
print(f"Output: {output}")             # tensor([[[-2., -2., -2.]]])
```

### Multi-Channel Example

```python
# Time series with 8 features (e.g., OHLCV + indicators), length 100
batch_size = 32
x = torch.randn(batch_size, 8, 100)  # (N, C_in, L)

# Extract 32 temporal features with kernel size 5
conv = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=5, padding=2)

output = conv(x)
print(f"Input: {x.shape}")    # [32, 8, 100]
print(f"Output: {output.shape}")  # [32, 32, 100] (same length with padding=2)

params = sum(p.numel() for p in conv.parameters())
print(f"Parameters: {params:,}")  # 8 × 32 × 5 + 32 = 1,312
```

---

## 1D Convolution as Matrix Multiplication

The sliding-window interpretation of 1D convolution can be expressed as multiplication with a **Toeplitz matrix**. For input $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^\top$ and kernel $\mathbf{k} = [k_0, k_1, k_2]^\top$:

$$\mathbf{y} = \mathbf{T}\mathbf{x} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}$$

This matrix is **sparse** and **structured** (constant along diagonals), which is why convolution is far more efficient than a general matrix multiplication—only $k$ unique values need to be stored.

### Transposed Convolution as $\mathbf{T}^\top$

The gradient of convolution w.r.t. the input corresponds to multiplication by $\mathbf{T}^\top$, which is a **transposed convolution** (see [Transposed Convolutions](transposed_conv.md)):

$$\mathbf{T}^\top = \begin{bmatrix}
k_0 & 0 & 0 \\
k_1 & k_0 & 0 \\
k_2 & k_1 & k_0 \\
0 & k_2 & k_1 \\
0 & 0 & k_2
\end{bmatrix}$$

This maps a length-3 vector back to length-5, performing upsampling.

```python
import torch
import torch.nn as nn

# Verify: Conv1d backward = ConvTranspose1d forward
x = torch.randn(1, 1, 5, requires_grad=True)
w = torch.randn(1, 1, 3)

# Forward pass
y = torch.nn.functional.conv1d(x, w)
# y has shape (1, 1, 3)

# Backward pass gives gradient w.r.t. x
grad_output = torch.randn(1, 1, 3)
y.backward(grad_output)

# This is equivalent to transposed convolution
grad_manual = torch.nn.functional.conv_transpose1d(grad_output, w)
print(f"Gradient match: {torch.allclose(x.grad, grad_manual, atol=1e-5)}")
```

---

## Backpropagation in 1D Convolution

### Forward Pass

$$y_i = \sum_{j=0}^{k-1} x_{i+j} \cdot w_j$$

### Gradient with Respect to Input

$$\frac{\partial L}{\partial x_i} = \sum_{j=\max(0, i-k+1)}^{\min(i, n-k)} \frac{\partial L}{\partial y_j} \cdot w_{i-j}$$

This is equivalent to **full convolution** of the gradient with a **flipped kernel**:

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} *_{full} \text{flip}(\mathbf{w})$$

### Gradient with Respect to Kernel

$$\frac{\partial L}{\partial w_j} = \sum_{i} \frac{\partial L}{\partial y_i} \cdot x_{i+j}$$

This is the **cross-correlation** of input with output gradient.

### NumPy Implementation

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
    
    Returns:
        grad_x: Gradient with respect to input (dL/dx)
        grad_w: Gradient with respect to weights (dL/dw)
    """
    n, k = len(x), len(w)
    out_len = len(grad_output)
    
    # Gradient w.r.t. input: full convolution with flipped kernel
    grad_x = np.zeros(n)
    w_flip = w[::-1]
    grad_padded = np.pad(grad_output, (k-1, k-1), mode='constant')
    for i in range(n):
        grad_x[i] = np.sum(grad_padded[i:i+k] * w_flip)
    
    # Gradient w.r.t. weights: correlation of input with grad_output
    grad_w = np.zeros(k)
    for j in range(k):
        grad_w[j] = np.sum(x[j:j+out_len] * grad_output)
    
    return grad_x, grad_w


# Numerical gradient verification
np.random.seed(42)
x = np.random.randn(8)
w = np.random.randn(3)

y = conv1d_forward(x, w)
grad_output = np.random.randn(len(y))

grad_x, grad_w = conv1d_backward(x, w, grad_output)

# Numerical verification
eps = 1e-5
grad_w_numerical = np.zeros_like(w)
for i in range(len(w)):
    w_plus, w_minus = w.copy(), w.copy()
    w_plus[i] += eps
    w_minus[i] -= eps
    grad_w_numerical[i] = (np.sum(conv1d_forward(x, w_plus) * grad_output) - 
                            np.sum(conv1d_forward(x, w_minus) * grad_output)) / (2 * eps)

print("Analytical grad_w:", grad_w)
print("Numerical grad_w: ", grad_w_numerical)
print("Match:", np.allclose(grad_w, grad_w_numerical))
```

---

## Causal Convolution

In many time-series applications, the model must not look into the future—the output at time $t$ should depend only on inputs at times $\leq t$. This requires **causal convolution**.

### Left-Only Padding

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: output at time t depends only on input at times ≤ t.
    
    Achieved by padding only on the left side and removing trailing elements.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
    
    def forward(self, x):
        out = self.conv(x)
        # Remove the right padding to enforce causality
        return out[:, :, :-self.padding] if self.padding > 0 else out


# Verify causality
causal = CausalConv1d(1, 1, kernel_size=3)
x = torch.randn(1, 1, 10)
y = causal(x)
print(f"Input length: {x.shape[2]}, Output length: {y.shape[2]}")
# Both are 10: output[t] depends on input[t-2], input[t-1], input[t]
```

### Dilated Causal Convolution (WaveNet-style)

Stacking causal convolutions with exponentially increasing dilation achieves very large receptive fields while maintaining causality:

```python
class DilatedCausalConv1d(nn.Module):
    """Dilated causal convolution for sequence modeling."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


def build_wavenet_stack(channels, kernel_size=2, num_layers=10):
    """Stack with exponentially increasing dilation."""
    layers = []
    for i in range(num_layers):
        dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        layers.append(DilatedCausalConv1d(channels, channels, kernel_size, dilation))
    return nn.Sequential(*layers)

# Receptive field calculation:
# With K=2 and dilations [1, 2, 4, ..., 512]:
# RF = 1 + sum(d * (K-1)) = 1 + (1+2+4+...+512) = 1024 samples
stack = build_wavenet_stack(64, kernel_size=2, num_layers=10)
print(f"Total layers: 10, Receptive field: 1024 samples")
print(f"At 16kHz audio: {1024/16000:.3f}s of context")
```

---

## Temporal Convolutional Network (TCN)

TCNs combine causal convolutions, dilations, and residual connections for general-purpose sequence modeling:

```python
import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block.
    
    Combines:
    - Dilated causal convolution
    - Weight normalization
    - ReLU activation
    - Dropout for regularization
    - Residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=self.padding)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=self.padding)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1×1 conv if channels change)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                        if in_channels != out_channels else nn.Identity())
    
    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Causal trim
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding]  # Causal trim
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual
        return self.relu(out + self.residual(x))


class TCN(nn.Module):
    """Complete Temporal Convolutional Network."""
    def __init__(self, input_channels, hidden_channels, output_size,
                 kernel_size=3, num_layers=6, dropout=0.2):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else hidden_channels
            layers.append(TCNBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_channels, output_size)
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.network(x)
        # Take the last time step for classification/regression
        out = out[:, :, -1]
        return self.output_layer(out)


# Example: predicting next-step returns from 8-feature price history
model = TCN(input_channels=8, hidden_channels=64, output_size=1,
            kernel_size=3, num_layers=8)

x = torch.randn(32, 8, 256)  # 32 samples, 8 features, 256 time steps
pred = model(x)
print(f"Input: {x.shape}, Prediction: {pred.shape}")  # [32, 1]

# Receptive field: 1 + sum(2^i * (3-1)) for i=0..7 = 1 + 2*(1+2+...+128) = 511
print(f"Receptive field: 511 time steps")
```

---

## 1D Convolution for Financial Time Series

### Feature Extraction from Price Data

```python
import torch
import torch.nn as nn

class FinancialFeatureExtractor(nn.Module):
    """
    Multi-scale 1D convolution for extracting temporal patterns
    from financial time series at different time horizons.
    """
    def __init__(self, input_features, hidden_dim=64):
        super().__init__()
        
        # Short-term patterns (3-day window)
        self.short_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Medium-term patterns (5-day / weekly)
        self.medium_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Long-term patterns (21-day / monthly)
        self.long_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=21, padding=10),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, features, time_steps)
               e.g., features = [open, high, low, close, volume, returns, ...]
        Returns:
            Multi-scale features: (batch, 3*hidden_dim, time_steps)
        """
        short = self.short_conv(x)
        medium = self.medium_conv(x)
        long_term = self.long_conv(x)
        
        return torch.cat([short, medium, long_term], dim=1)


# Example usage
extractor = FinancialFeatureExtractor(input_features=6, hidden_dim=32)
# 6 features: OHLCV + returns, 252 trading days
x = torch.randn(16, 6, 252)
features = extractor(x)
print(f"Multi-scale features: {features.shape}")  # [16, 96, 252]
```

---

## Conv1d vs. Conv2d: When to Use Which

| Criterion | Conv1d | Conv2d |
|-----------|--------|--------|
| Data structure | Sequences, time series | Images, spatial grids |
| Input shape | $(N, C, L)$ | $(N, C, H, W)$ |
| Kernel slides over | 1 dimension (time/position) | 2 dimensions (height × width) |
| Typical kernel sizes | 3, 5, 7, 21 | 3×3, 5×5, 7×7 |
| Example applications | Audio, NLP, finance | Images, video frames, vol surfaces |
| Parameter count | $C_{out} \times C_{in} \times K$ | $C_{out} \times C_{in} \times K^2$ |

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Operation** | Sliding dot product along one spatial dimension |
| **Input shape** | $(N, C_{in}, L)$: batch, channels, sequence length |
| **Output size** | $\lfloor (L + 2p - d(k-1) - 1) / s \rfloor + 1$ |
| **Causal** | Left-only padding ensures no future information leakage |
| **Dilated** | Exponential receptive field growth with constant parameters |
| **Matrix form** | Toeplitz matrix; transpose gives transposed convolution |

## Key Takeaways

1. **Conv1d** operates on sequences with shape $(N, C, L)$, sliding kernels along the temporal dimension
2. **Causal convolutions** (left-padding + trim) ensure outputs depend only on past and present inputs
3. **Dilated causal stacks** achieve exponential receptive field growth—10 layers of kernel-2 dilated convolutions give a receptive field of 1024
4. **TCNs** combine dilated causal convolutions with residual connections for competitive sequence modeling
5. **Multi-scale convolutions** with different kernel sizes capture patterns at different temporal horizons
6. **Backpropagation** through 1D conv produces a full convolution with a flipped kernel (= transposed convolution)

## References

1. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." *arXiv preprint arXiv:1609.03499*.

2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv preprint arXiv:1803.01271*.

3. Lea, C., et al. (2017). "Temporal Convolutional Networks for Action Segmentation and Detection." *CVPR*.

4. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.
