# Discrete Convolution and Cross-Correlation

## Introduction

Convolutional Neural Networks (CNNs) derive their name from the **convolution** operation, a mathematical operation that combines two functions to produce a third. In the context of deep learning, convolution enables neural networks to automatically learn spatial hierarchies of features from input data, making them exceptionally powerful for image recognition, computer vision, and signal processing tasks.

This section provides a rigorous mathematical treatment of discrete convolution and cross-correlation, establishing the theoretical foundations necessary to understand how CNNs extract features from images.

## Mathematical Foundations

### Continuous Convolution

Before examining discrete convolution, we briefly review the continuous case. For two continuous functions $f$ and $g$, the convolution $(f * g)$ is defined as:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \cdot g(t - \tau) \, d\tau$$

This operation "slides" function $g$ over function $f$, computing the integral of their pointwise product at each position.

### Discrete Convolution in 1D

For discrete sequences, the integral becomes a summation. Given an input signal $x[n]$ and a kernel (filter) $h[m]$, the **discrete convolution** is:

$$(x * h)[n] = \sum_{m=-\infty}^{\infty} x[m] \cdot h[n - m]$$

Key observation: The kernel $h$ is **flipped** (reversed) before sliding across the input. This flip distinguishes convolution from cross-correlation.

**Example:** Consider a simple 1D convolution with:

- Input: $x = [1, 2, 3, 4, 5]$
- Kernel: $h = [1, 0, -1]$

The kernel is first flipped to $h' = [-1, 0, 1]$, then slid across the input:

$$y[n] = \sum_{m} x[m] \cdot h[n - m]$$

### Cross-Correlation in 1D

**Cross-correlation** is similar to convolution but **without flipping** the kernel:

$$(x \star h)[n] = \sum_{m=-\infty}^{\infty} x[m] \cdot h[n + m]$$

This seemingly small difference has significant implications:

| Property | Convolution | Cross-Correlation |
|----------|-------------|-------------------|
| Kernel flipped | Yes | No |
| Commutativity | $f * g = g * f$ | Not commutative |
| Signal processing | Standard definition | Similarity measure |
| Deep learning | Often called "convolution" | Actually used in CNNs |

!!! warning "Important Terminology Note"
    In deep learning literature and frameworks like PyTorch, what is called "convolution" is technically **cross-correlation**. Since the kernels are learned parameters, the flip is irrelevant—the network can learn either the kernel or its flipped version. We will follow this convention throughout, using "convolution" to mean cross-correlation.

## 2D Convolution for Images

### Mathematical Definition

For a 2D input image $I$ and kernel $K$, the discrete 2D convolution (cross-correlation) at position $(i, j)$ is:

$$(I * K)[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[i + m, j + n] \cdot K[m, n]$$

where:

- $I$ is the input image of size $H \times W$
- $K$ is the kernel of size $M \times N$ (typically $M = N$, e.g., $3 \times 3$)
- $(i, j)$ is the position in the output feature map

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

### Output Dimensions

For an input of size $H \times W$ and kernel of size $K \times K$, the output size (without padding) is:

$$H_{out} = H - K + 1$$
$$W_{out} = W - K + 1$$

This reduction in spatial dimensions is a direct consequence of the kernel requiring surrounding pixels to compute each output.

## Multi-Channel Convolution

### RGB Images and Feature Maps

Real-world images have multiple channels (e.g., RGB with 3 channels). In CNNs, intermediate layers produce **feature maps** with many channels. Multi-channel convolution handles this naturally.

### Mathematical Formulation

For an input with $C_{in}$ channels and kernel with $C_{in}$ channels, the output at position $(i, j)$ is:

$$(I * K)[i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[c, i + m, j + n] \cdot K[c, m, n]$$

The kernel is now 3-dimensional: $K \in \mathbb{R}^{C_{in} \times M \times N}$.

### Multiple Output Channels

To produce $C_{out}$ output channels, we use $C_{out}$ different kernels, each producing one output channel:

$$\text{Output}[k, i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[c, i + m, j + n] \cdot K[k, c, m, n] + b[k]$$

where:

- $K \in \mathbb{R}^{C_{out} \times C_{in} \times M \times N}$ is the full kernel tensor
- $b \in \mathbb{R}^{C_{out}}$ is the bias vector

### Parameter Count

The number of parameters in a convolutional layer is:

$$\text{Parameters} = C_{out} \times C_{in} \times K_H \times K_W + C_{out}$$

where the last term accounts for biases.

**Example:** A layer with 32 input channels, 64 output channels, and $3 \times 3$ kernels has:

$$64 \times 32 \times 3 \times 3 + 64 = 18,496 \text{ parameters}$$

## Properties of Convolution

### Translation Equivariance

A fundamental property of convolution is **translation equivariance**: if the input is shifted, the output shifts by the same amount.

Formally, let $T_{\Delta}$ denote a translation operator. Then:

$$T_{\Delta}(f * g) = (T_{\Delta} f) * g = f * (T_{\Delta} g)$$

This property is crucial for CNNs—a feature detector (kernel) will detect the same feature regardless of where it appears in the image.

### Locality

Each output value depends only on a **local region** of the input, determined by the kernel size. This induces an inductive bias that local patterns (edges, textures) are important for understanding images.

### Parameter Sharing

The same kernel is applied across all spatial positions. This dramatically reduces the number of parameters compared to fully connected layers and enables the network to learn features that are useful across the entire image.

### Sparse Connectivity

Unlike fully connected layers where each output depends on all inputs, convolutional layers have **sparse connectivity**—each output depends only on a small subset of inputs.

## Common Kernels and Their Effects

### Edge Detection Kernels

**Sobel Operators** detect horizontal and vertical edges:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

### Sharpening Kernel

$$K_{sharpen} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

### Gaussian Blur

$$K_{blur} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

### Identity Kernel

$$K_{identity} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

!!! note "Learned Kernels"
    In CNNs, kernels are **learned** through backpropagation, not hand-designed. The network discovers optimal kernels for the task at hand. Early layers typically learn edge detectors similar to Sobel operators, while deeper layers learn more complex, task-specific features.

## PyTorch Implementation

### Basic 2D Convolution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a convolutional layer
# in_channels: number of input channels
# out_channels: number of output channels (number of filters)
# kernel_size: size of the convolving kernel
conv_layer = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 different filters
    kernel_size=3,      # 3x3 kernels
    stride=1,           # Move 1 pixel at a time
    padding=0,          # No padding
    bias=True           # Include bias term
)

# Input: (batch_size, channels, height, width)
x = torch.randn(1, 3, 32, 32)  # Single RGB image, 32x32

# Forward pass
output = conv_layer(x)
print(f"Input shape: {x.shape}")       # torch.Size([1, 3, 32, 32])
print(f"Output shape: {output.shape}") # torch.Size([1, 64, 30, 30])
```

### Manual Convolution Implementation

```python
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
x = torch.randn(2, 3, 8, 8)  # Batch of 2 RGB images
kernel = torch.randn(16, 3, 3, 3)  # 16 filters, 3x3
bias = torch.randn(16)

output_manual = manual_conv2d(x, kernel, bias)
print(f"Manual conv output shape: {output_manual.shape}")  # [2, 16, 6, 6]

# Verify against PyTorch's implementation
conv = nn.Conv2d(3, 16, 3, bias=True)
conv.weight.data = kernel
conv.bias.data = bias
output_pytorch = conv(x)

print(f"Max difference: {(output_manual - output_pytorch).abs().max().item():.2e}")
# Should be very small (numerical precision)
```

### Visualizing Convolution with Edge Detection

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Create a simple test image
def create_test_image():
    """Create a simple image with clear edges."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 1.0  # White square on black background
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

# Define Sobel kernels
sobel_x = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=torch.float32).view(1, 1, 3, 3)

sobel_y = torch.tensor([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=torch.float32).view(1, 1, 3, 3)

# Apply convolution
image = create_test_image()
edge_x = F.conv2d(image, sobel_x, padding=1)
edge_y = F.conv2d(image, sobel_y, padding=1)

# Compute edge magnitude
edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(image.squeeze(), cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(edge_x.squeeze(), cmap='gray')
axes[1].set_title('Horizontal Edges (Sobel X)')
axes[1].axis('off')

axes[2].imshow(edge_y.squeeze(), cmap='gray')
axes[2].set_title('Vertical Edges (Sobel Y)')
axes[2].axis('off')

axes[3].imshow(edge_magnitude.squeeze(), cmap='gray')
axes[3].set_title('Edge Magnitude')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('convolution_edge_detection.png', dpi=150)
plt.show()
```

### Comparing Convolution and Cross-Correlation

```python
def true_convolution(x, kernel):
    """
    Perform true mathematical convolution (with kernel flip).
    """
    # Flip the kernel both horizontally and vertically
    flipped_kernel = torch.flip(kernel, dims=[2, 3])
    return F.conv2d(x, flipped_kernel)

def cross_correlation(x, kernel):
    """
    Perform cross-correlation (what PyTorch calls conv2d).
    """
    return F.conv2d(x, kernel)

# Asymmetric kernel to see the difference
kernel = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.float32).view(1, 1, 3, 3)

x = torch.randn(1, 1, 5, 5)

conv_result = true_convolution(x, kernel)
xcorr_result = cross_correlation(x, kernel)

print("For asymmetric kernels, convolution ≠ cross-correlation")
print(f"Difference: {(conv_result - xcorr_result).abs().max().item():.4f}")

# For symmetric kernels, they are the same
symmetric_kernel = torch.tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=torch.float32).view(1, 1, 3, 3)

conv_symmetric = true_convolution(x, symmetric_kernel)
xcorr_symmetric = cross_correlation(x, symmetric_kernel)

print("\nFor symmetric kernels, convolution = cross-correlation")
print(f"Difference: {(conv_symmetric - xcorr_symmetric).abs().max().item():.2e}")
```

## Computational Complexity

### Direct Convolution

For input size $H \times W$ with $C_{in}$ channels, kernel size $K \times K$, and $C_{out}$ output channels:

$$O(H \times W \times C_{in} \times C_{out} \times K^2)$$

### FFT-Based Convolution

For large kernels, FFT-based convolution can be more efficient:

$$O(H \times W \times \log(HW) \times C_{in} \times C_{out})$$

In practice, for typical CNN kernel sizes ($3 \times 3$, $5 \times 5$), direct convolution is faster due to better memory access patterns and optimized implementations (cuDNN).

## Summary

Key takeaways from this section:

1. **Convolution** is a mathematical operation that combines an input and a kernel to produce an output, enabling feature extraction.

2. **Cross-correlation** (used in deep learning as "convolution") differs from true convolution by not flipping the kernel.

3. **Multi-channel convolution** naturally handles images with multiple channels and produces multiple output feature maps.

4. **Key properties** include translation equivariance, locality, parameter sharing, and sparse connectivity.

5. **PyTorch's `nn.Conv2d`** implements cross-correlation, which is standard in deep learning frameworks.

## Exercises

1. **Manual Implementation**: Implement 2D convolution without using any library functions. Verify your implementation against PyTorch's `F.conv2d`.

2. **Edge Detection**: Apply Sobel operators to real images and visualize the detected edges. Experiment with other edge detection kernels (Prewitt, Laplacian).

3. **Parameter Counting**: Calculate the number of parameters in the first three convolutional layers of VGG-16. Verify by examining `model.parameters()`.

4. **Convolution vs. Cross-Correlation**: Create examples where convolution and cross-correlation produce different results. Under what conditions are they equivalent?

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Chapter 9: Convolutional Networks.

3. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.
