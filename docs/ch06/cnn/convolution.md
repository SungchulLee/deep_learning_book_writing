# Convolution Operation

## Introduction

Convolutional Neural Networks (CNNs) derive their name from the **convolution** operation, a mathematical operation that combines two functions to produce a third. In the context of deep learning, convolution enables neural networks to automatically learn spatial hierarchies of features from input data, making them exceptionally powerful for image recognition, computer vision, and signal processing tasks.

This section provides a rigorous mathematical treatment of discrete convolution and cross-correlation, establishing the theoretical foundations necessary to understand how CNNs extract features from images.

> **Important Terminology Note**: In deep learning literature and frameworks like PyTorch, what is called "convolution" is technically **cross-correlation**. Since the kernels are learned parameters, the flip is irrelevant—the network can learn either the kernel or its flipped version. We follow this convention throughout, using "convolution" to mean cross-correlation unless otherwise specified.

---

## Mathematical Foundations

### Continuous Convolution

Before examining discrete convolution, we briefly review the continuous case. For two continuous functions $f$ and $g$, the convolution $(f * g)$ is defined as:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \cdot g(t - \tau) \, d\tau$$

This operation "slides" function $g$ over function $f$, computing the integral of their pointwise product at each position.

### Discrete Convolution

For discrete sequences, the integral becomes a summation. Given an input signal $x[n]$ and a kernel (filter) $h[m]$, the **discrete convolution** is:

$$(x * h)[n] = \sum_{m=-\infty}^{\infty} x[m] \cdot h[n - m]$$

Key observation: The kernel $h$ is **flipped** (reversed) before sliding across the input. This flip distinguishes true convolution from cross-correlation.

### Cross-Correlation

**Cross-correlation** is similar to convolution but **without flipping** the kernel:

$$(x \star h)[n] = \sum_{m=-\infty}^{\infty} x[m] \cdot h[n + m]$$

Or equivalently, for a finite kernel of size $k$:

$$y[i] = \sum_{j=0}^{k-1} x[i + j] \cdot h[j]$$

This is what CNNs actually compute.

### Comparison Table

| Property | True Convolution | Cross-Correlation (CNN) |
|----------|------------------|-------------------------|
| Kernel flipped | Yes (180°) | No |
| Mathematical notation | $f * g$ | $f \star g$ |
| Commutativity | Yes: $f * g = g * f$ | No |
| Associativity | Yes | No |
| Signal processing | Standard definition | Similarity measure |
| Deep learning | Rarely used | Standard |

### Why CNNs Use Cross-Correlation

1. **Learned Kernels**: Since kernels are learned during training, a flipped version of the optimal kernel would be learned anyway
2. **Computational Simplicity**: Cross-correlation avoids the extra flip operation
3. **Equivalent Results**: For symmetric kernels (common in classical image processing), convolution and cross-correlation are identical

---

## 2D Convolution for Images

### Mathematical Definition

For a 2D input image $I$ of size $H \times W$ and kernel $K$ of size $M \times N$, the discrete 2D convolution (cross-correlation) at position $(i, j)$ is:

$$(I * K)[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[i + m, j + n] \cdot K[m, n]$$

### Output Dimensions

For an input of size $H \times W$ and kernel of size $K \times K$, the output size (without padding) is:

$$H_{out} = H - K + 1$$
$$W_{out} = W - K + 1$$

This reduction in spatial dimensions is a direct consequence of the kernel requiring surrounding pixels to compute each output.

### Visual Interpretation

The convolution operation can be visualized as follows:

1. Position the kernel at the top-left corner of the image
2. Compute the element-wise product between overlapping elements
3. Sum all products to get a single output value
4. Slide the kernel to the next position and repeat

```
Input Image (5×5)          Kernel (3×3)
┌───┬───┬───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 0 │ 1 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 0 │ 1 │ 2 │ 3 │ 1 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 1 │ 2 │ 1 │ 0 │ 0 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     └───┴───┴───┘
│ 0 │ 0 │ 1 │ 2 │ 1 │
├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 0 │ 1 │ 0 │
└───┴───┴───┴───┴───┘

Output at position (0,0):
= 1×1 + 2×0 + 3×(-1) + 0×1 + 1×0 + 2×(-1) + 1×1 + 2×0 + 1×(-1)
= 1 + 0 - 3 + 0 + 0 - 2 + 1 + 0 - 1
= -4
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# 2D Convolution Example
# Input: (batch_size, channels, height, width)
x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
print("Input:")
print(x.squeeze())

# Create 2D conv layer
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# Sobel-like edge detection kernel
with torch.no_grad():
    kernel = torch.tensor([[[[1., 0., -1.],
                             [2., 0., -2.],
                             [1., 0., -1.]]]])
    conv2d.weight = nn.Parameter(kernel)

output = conv2d(x)
print(f"\nOutput shape: {output.shape}")  # torch.Size([1, 1, 2, 2])
print("Output:")
print(output.squeeze())
```

---

## Multi-Channel Convolution

### RGB Images and Feature Maps

Real-world images have multiple channels (e.g., RGB with 3 channels). In CNNs, intermediate layers produce **feature maps** with many channels. Multi-channel convolution handles this naturally.

### Mathematical Formulation

For an input with $C_{in}$ channels and kernel with $C_{in}$ channels, the output at position $(i, j)$ is:

$$(I * K)[i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[c, i + m, j + n] \cdot K[c, m, n]$$

The kernel is now 3-dimensional: $K \in \mathbb{R}^{C_{in} \times M \times N}$.

### Multiple Output Channels

To produce $C_{out}$ output channels, we use $C_{out}$ different kernels, each producing one output channel:

$$Y[o, i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X[c, i + m, j + n] \cdot K[o, c, m, n] + b[o]$$

where:
- $K \in \mathbb{R}^{C_{out} \times C_{in} \times M \times N}$ is the full kernel tensor
- $b \in \mathbb{R}^{C_{out}}$ is the bias vector

### Parameter Count

The number of parameters in a convolutional layer is:

$$\text{Parameters} = C_{out} \times C_{in} \times K_H \times K_W + C_{out}$$

where the last term accounts for biases.

**Example:** A layer with 32 input channels, 64 output channels, and $3 \times 3$ kernels has:

$$64 \times 32 \times 3 \times 3 + 64 = 18{,}496 \text{ parameters}$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Multi-channel convolution
# RGB image: 3 channels, 32x32
batch_size = 4
x = torch.randn(batch_size, 3, 32, 32)

# Conv layer: 3 input channels → 64 output channels
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    padding=1,  # Same padding
    bias=True
)

output = conv(x)
print(f"Input shape: {x.shape}")       # torch.Size([4, 3, 32, 32])
print(f"Output shape: {output.shape}") # torch.Size([4, 64, 32, 32])

# Parameter count
num_params = sum(p.numel() for p in conv.parameters())
print(f"Parameters: {num_params}")     # 64 × 3 × 3 × 3 + 64 = 1,792
```

---

## Properties of Convolution

### Translation Equivariance

A fundamental property of convolution is **translation equivariance**: if the input is shifted, the output shifts by the same amount.

Formally, let $T_{\Delta}$ denote a translation operator. Then:

$$T_{\Delta}(f * g) = (T_{\Delta} f) * g = f * (T_{\Delta} g)$$

This property is crucial for CNNs—a feature detector (kernel) will detect the same feature regardless of where it appears in the image.

```python
import torch
import torch.nn.functional as F

# Demonstrate translation equivariance
kernel = torch.randn(1, 1, 3, 3)

# Original image
img = torch.zeros(1, 1, 10, 10)
img[0, 0, 2:5, 2:5] = 1.0  # Square at position (2,2)

# Translated image
img_shifted = torch.zeros(1, 1, 10, 10)
img_shifted[0, 0, 4:7, 4:7] = 1.0  # Same square at position (4,4)

# Apply convolution
out1 = F.conv2d(img, kernel, padding=1)
out2 = F.conv2d(img_shifted, kernel, padding=1)

# The outputs are shifted versions of each other
# (modulo boundary effects)
```

### Locality (Local Receptive Fields)

Each output value depends only on a **local region** of the input, determined by the kernel size. This induces an inductive bias that local patterns (edges, textures) are important for understanding images.

### Parameter Sharing

The same kernel is applied across all spatial positions:
- **Memory efficient**: Same parameters for entire input
- **Statistical efficiency**: Learns from all positions simultaneously
- **Reduced overfitting**: Fewer parameters than fully connected layers

### Sparse Connectivity

Unlike fully connected layers where each output depends on all inputs, convolutional layers have **sparse connectivity**—each output depends only on a small subset of inputs.

---

## Common Kernels and Their Effects

### Edge Detection Kernels

**Sobel Operators** detect horizontal and vertical edges:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**Laplacian** (second derivative, detects edges in all directions):

$$K_{laplacian} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

### Other Common Kernels

**Sharpening**:
$$K_{sharpen} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

**Gaussian Blur**:
$$K_{blur} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

**Identity** (no change):
$$K_{identity} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

### What CNNs Learn

In trained CNNs, kernels are **learned** through backpropagation, not hand-designed. The network discovers optimal kernels for the task:

- **Early layers**: Edge detectors similar to Sobel operators, color detectors, texture detectors
- **Middle layers**: Part detectors (eyes, wheels, windows)
- **Deep layers**: Object detectors, complex pattern recognizers

---

## Convolution as Matrix Multiplication

The sliding-window view of convolution is intuitive, but convolution can also be expressed as **matrix multiplication**, which is how GPUs actually compute it.

### Toeplitz Matrix Formulation

For a 1D input $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^\top$ and kernel $\mathbf{k} = [k_0, k_1, k_2]^\top$, the convolution $\mathbf{y} = \mathbf{T}\mathbf{x}$ uses the Toeplitz matrix:

$$\mathbf{T} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}$$

### im2col: The Practical Implementation

In practice, 2D convolution is implemented via the **im2col** (image to column) transformation, which rearranges input patches into a matrix so that convolution becomes a single matrix multiplication:

1. Extract each $K \times K$ input patch and flatten it into a column
2. Stack all patches into a matrix $\mathbf{X}_{col} \in \mathbb{R}^{(C_{in} \cdot K^2) \times (H_{out} \cdot W_{out})}$
3. Reshape kernels into $\mathbf{W}_{row} \in \mathbb{R}^{C_{out} \times (C_{in} \cdot K^2)}$
4. Compute $\mathbf{Y} = \mathbf{W}_{row} \cdot \mathbf{X}_{col}$

```python
import torch
import torch.nn.functional as F

def conv2d_via_im2col(x, weight, bias=None, stride=1, padding=0):
    """
    2D convolution implemented via im2col + GEMM.
    
    Args:
        x: Input tensor (N, C_in, H, W)
        weight: Kernel tensor (C_out, C_in, kH, kW)
        bias: Optional bias (C_out,)
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    
    # Apply padding
    if padding > 0:
        x = F.pad(x, [padding]*4)
        _, _, H, W = x.shape
    
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    
    # im2col: extract patches
    # unfold extracts sliding windows along a dimension
    cols = x.unfold(2, kH, stride).unfold(3, kW, stride)  # (N, C_in, H_out, W_out, kH, kW)
    cols = cols.contiguous().view(N, C_in * kH * kW, H_out * W_out)  # (N, C_in*kH*kW, L)
    
    # Reshape weight: (C_out, C_in*kH*kW)
    W_row = weight.view(C_out, -1)
    
    # Matrix multiplication: (C_out, C_in*kH*kW) × (N, C_in*kH*kW, L) → (N, C_out, L)
    out = torch.bmm(W_row.unsqueeze(0).expand(N, -1, -1), cols)
    
    # Reshape to spatial
    out = out.view(N, C_out, H_out, W_out)
    
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out


# Verify against PyTorch
x = torch.randn(2, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
b = torch.randn(16)

out_custom = conv2d_via_im2col(x, w, b, padding=1)
out_pytorch = F.conv2d(x, w, b, padding=1)

print(f"Max difference: {(out_custom - out_pytorch).abs().max().item():.2e}")
```

This matrix-multiplication perspective also clarifies why the **backward pass** through convolution w.r.t. the input is a transposed convolution—it corresponds to multiplication by $\mathbf{T}^\top$.

---

## Implementation Examples

### Manual 2D Convolution

```python
import torch
import torch.nn as nn

def manual_conv2d(input_tensor, kernel, bias=None):
    """
    Implement 2D convolution manually for educational purposes.
    
    Args:
        input_tensor: (batch, in_channels, H, W)
        kernel: (out_channels, in_channels, kH, kW)
        bias: (out_channels,) or None
    
    Returns:
        output: (batch, out_channels, H_out, W_out)
    """
    batch_size, in_channels, H, W = input_tensor.shape
    out_channels, _, kH, kW = kernel.shape
    
    # Calculate output dimensions
    H_out = H - kH + 1
    W_out = W - kW + 1
    
    # Initialize output
    output = torch.zeros(batch_size, out_channels, H_out, W_out)
    
    # Perform convolution
    for b in range(batch_size):           # For each sample in batch
        for oc in range(out_channels):    # For each output channel
            for i in range(H_out):        # For each output row
                for j in range(W_out):    # For each output column
                    # Extract the receptive field
                    receptive_field = input_tensor[b, :, i:i+kH, j:j+kW]
                    # Element-wise multiplication and sum
                    output[b, oc, i, j] = (receptive_field * kernel[oc]).sum()
                    # Add bias if present
                    if bias is not None:
                        output[b, oc, i, j] += bias[oc]
    
    return output


# Test manual implementation
x = torch.randn(2, 3, 8, 8)
kernel = torch.randn(16, 3, 3, 3)
bias = torch.randn(16)

output_manual = manual_conv2d(x, kernel, bias)

# Verify against PyTorch
conv = nn.Conv2d(3, 16, 3, bias=True)
conv.weight.data = kernel
conv.bias.data = bias
output_pytorch = conv(x)

print(f"Max difference: {(output_manual - output_pytorch).abs().max().item():.2e}")
# Should be very small (numerical precision)
```

### Edge Detection Visualization

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def create_test_image():
    """Create a simple image with clear edges."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 1.0  # White square on black background
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

# Define Sobel kernels
sobel_x = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)

sobel_y = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

# Apply convolution
image = create_test_image()
edge_x = F.conv2d(image, sobel_x, padding=1)
edge_y = F.conv2d(image, sobel_y, padding=1)

# Compute edge magnitude
edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

# Results: edge_x detects vertical edges, edge_y detects horizontal edges
# edge_magnitude shows all edges
```

### Comparing Convolution and Cross-Correlation

```python
import torch
import torch.nn.functional as F

def true_convolution(x, kernel):
    """Perform true mathematical convolution (with kernel flip)."""
    flipped_kernel = torch.flip(kernel, dims=[2, 3])
    return F.conv2d(x, flipped_kernel)

def cross_correlation(x, kernel):
    """Perform cross-correlation (what PyTorch calls conv2d)."""
    return F.conv2d(x, kernel)

# Asymmetric kernel to see the difference
kernel = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.]]).view(1, 1, 3, 3)

x = torch.randn(1, 1, 5, 5)

conv_result = true_convolution(x, kernel)
xcorr_result = cross_correlation(x, kernel)

print("For asymmetric kernels, convolution ≠ cross-correlation")
print(f"Difference: {(conv_result - xcorr_result).abs().max().item():.4f}")

# For symmetric kernels, they are identical
symmetric_kernel = torch.tensor([[1., 2., 1.],
                                 [2., 4., 2.],
                                 [1., 2., 1.]]).view(1, 1, 3, 3)

conv_symmetric = true_convolution(x, symmetric_kernel)
xcorr_symmetric = cross_correlation(x, symmetric_kernel)

print("\nFor symmetric kernels, convolution = cross-correlation")
print(f"Difference: {(conv_symmetric - xcorr_symmetric).abs().max().item():.2e}")
```

---

## Computational Complexity

### Direct Convolution

For input size $H \times W$ with $C_{in}$ channels, kernel size $K \times K$, and $C_{out}$ output channels:

$$O(H \times W \times C_{in} \times C_{out} \times K^2)$$

### FFT-Based Convolution

For large kernels, FFT-based convolution can be more efficient:

$$O(H \times W \times \log(HW) \times C_{in} \times C_{out})$$

In practice, for typical CNN kernel sizes ($3 \times 3$, $5 \times 5$), direct convolution is faster due to better memory access patterns and optimized implementations (cuDNN, Winograd).

### Practical Performance

Modern deep learning frameworks employ several optimizations:

- **im2col + GEMM**: Converts convolution to matrix multiplication, leveraging highly optimized BLAS libraries
- **Winograd convolution**: Reduces multiplications for small kernels ($3 \times 3$) by ~2.25×
- **FFT convolution**: Beneficial for kernel sizes > ~11×11
- **cuDNN auto-tuning**: Automatically selects the fastest algorithm for given tensor shapes

---

## Backpropagation Through Convolution

Understanding the gradient computations is essential for implementing custom layers and debugging training:

**Gradient w.r.t. Input** (for propagating gradients backward):
$$\frac{\partial L}{\partial X_{c,i,j}} = \sum_{k,m,n} \frac{\partial L}{\partial Y_{k, i-m, j-n}} \cdot W_{k,c,m,n}$$

This is a **full convolution** of the output gradient with the **flipped** weights—equivalent to a **transposed convolution** (see [Transposed Convolutions](transposed_conv.md)).

**Gradient w.r.t. Weights** (for updating parameters):
$$\frac{\partial L}{\partial W_{k,c,m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}} \cdot X_{c, i+m, j+n}$$

This is a **cross-correlation** between the input and the output gradient.

**Gradient w.r.t. Bias**:
$$\frac{\partial L}{\partial b_k} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}}$$

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Operation** | Element-wise multiply and sum over local regions |
| **CNN Convention** | Uses cross-correlation (no kernel flip) |
| **Multi-channel** | Sum over all input channels, stack for output channels |
| **Key Properties** | Translation equivariance, locality, parameter sharing |
| **Parameters** | $C_{out} \times C_{in} \times K^2 + C_{out}$ |
| **Output Size** | $(H - K + 1) \times (W - K + 1)$ without padding |
| **Implementation** | im2col + GEMM on GPUs for efficiency |

## Key Takeaways

1. **CNNs use cross-correlation**, not true convolution, but the terms are often used interchangeably
2. **Kernels slide over input**, computing dot products at each position
3. **Multi-channel convolution** sums contributions from all input channels
4. **Translation equivariance** makes CNNs robust to object position
5. **Parameter sharing** and **local connectivity** make convolution efficient
6. **Early layers** detect simple features; **deeper layers** detect complex patterns
7. **Convolution is matrix multiplication** via Toeplitz/im2col, enabling GPU acceleration

## Exercises

1. **Manual Implementation**: Implement 2D convolution without using any library functions. Verify your implementation against PyTorch's `F.conv2d`.

2. **Edge Detection**: Apply Sobel operators to real images and visualize the detected edges. Experiment with other edge detection kernels (Prewitt, Laplacian).

3. **Parameter Counting**: Calculate the number of parameters in the first three convolutional layers of VGG-16. Verify by examining `model.parameters()`.

4. **Convolution vs. Cross-Correlation**: Create examples where convolution and cross-correlation produce different results. Under what conditions are they equivalent?

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press. Chapter 9: Convolutional Networks.

3. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.

4. Chellapilla, K., Puri, S., & Simard, P. (2006). "High Performance Convolutional Neural Networks for Document Processing." *Tenth International Workshop on Frontiers in Handwriting Recognition*.
