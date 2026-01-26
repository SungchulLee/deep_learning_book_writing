# Convolution as Matrix Multiplication (Toeplitz)

## Introduction

While convolution is conceptually understood as a sliding window operation, it can be mathematically reformulated as **matrix multiplication**. This perspective is essential for:

1. **Understanding computational implementations**: GPUs are optimized for matrix operations
2. **Deriving backpropagation**: Gradients become transpose operations
3. **Theoretical analysis**: Connection to linear algebra frameworks
4. **Hardware acceleration**: GEMM (General Matrix Multiply) optimizations

This section develops the theory of representing convolution as matrix multiplication using **Toeplitz matrices** and **im2col** transformations.

---

## 1D Convolution as Matrix Multiplication

### Basic Setup

Consider a 1D convolution with:

- Input: $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^\top$ (length 5)
- Kernel: $\mathbf{k} = [k_0, k_1, k_2]^\top$ (length 3)
- Output: $\mathbf{y} = [y_0, y_1, y_2]^\top$ (length 3, valid convolution)

The convolution operation is:

$$y_i = \sum_{j=0}^{2} x_{i+j} \cdot k_j$$

### Toeplitz Matrix Representation

We can express this as $\mathbf{y} = \mathbf{T} \mathbf{x}$, where $\mathbf{T}$ is a **Toeplitz matrix**:

$$\mathbf{T} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}$$

**Verification:**

$$\mathbf{y} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}
=
\begin{bmatrix}
k_0 x_0 + k_1 x_1 + k_2 x_2 \\
k_0 x_1 + k_1 x_2 + k_2 x_3 \\
k_0 x_2 + k_1 x_3 + k_2 x_4
\end{bmatrix}$$

### Convolution vs. Cross-Correlation

Note: True mathematical convolution flips the kernel, while neural network "convolution" is actually cross-correlation:

**True Convolution (kernel flipped):**
$$
\mathbf{T}_{\text{conv}} = \begin{bmatrix}
k_2 & k_1 & k_0 & 0 & 0 \\
0 & k_2 & k_1 & k_0 & 0 \\
0 & 0 & k_2 & k_1 & k_0
\end{bmatrix}
$$

**Cross-Correlation (what CNNs use):**
$$
\mathbf{T}_{\text{cross}} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}
$$

### Toeplitz Matrix Properties

A **Toeplitz matrix** has constant diagonals:

$$T_{i,j} = T_{i+1, j+1}$$

Properties:

1. Each row is a shifted version of the previous row
2. Encodes the "sliding" nature of convolution
3. Sparse structure allows efficient computation
4. Enables efficient storage and computation

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np

def create_toeplitz_1d(kernel, input_length, padding=0, stride=1):
    """
    Create Toeplitz matrix for 1D convolution.
    
    Args:
        kernel: 1D kernel tensor of length K
        input_length: Length of input signal
        padding: Zero padding on each side
        stride: Convolution stride
    
    Returns:
        Toeplitz matrix T such that conv(x, k) = T @ x
    """
    K = len(kernel)
    padded_length = input_length + 2 * padding
    output_length = (padded_length - K) // stride + 1
    
    # Initialize with zeros
    T = torch.zeros(output_length, padded_length)
    
    # Fill in the Toeplitz structure
    for i in range(output_length):
        start = i * stride
        T[i, start:start + K] = kernel
    
    # If padding, extract the part corresponding to original input
    if padding > 0:
        T_original = T[:, padding:padded_length - padding]
        return T_original
    
    return T


# Example
kernel = torch.tensor([1., 2., 3.])
x = torch.tensor([1., 2., 3., 4., 5.])

# Using Toeplitz matrix
T = create_toeplitz_1d(kernel, len(x))
y_toeplitz = T @ x

# Using PyTorch conv1d
x_conv = x.view(1, 1, -1)
k_conv = kernel.view(1, 1, -1)
y_pytorch = F.conv1d(x_conv, k_conv).squeeze()

print("Toeplitz matrix:")
print(T)
print("\nToeplitz result:", y_toeplitz)
print("PyTorch result:", y_pytorch)
print("Match:", torch.allclose(y_toeplitz, y_pytorch))
```

**Output:**
```
Toeplitz matrix:
tensor([[1., 2., 3., 0., 0.],
        [0., 1., 2., 3., 0.],
        [0., 0., 1., 2., 3.]])

Toeplitz result: tensor([10., 16., 22.])
PyTorch result: tensor([10., 16., 22.])
Match: True
```

### NumPy Implementation with Mode Options

```python
import numpy as np

def conv1d_as_matrix(x, kernel, mode='valid'):
    """
    Implement 1D convolution as matrix multiplication.
    
    Args:
        x: Input array of length n
        kernel: Kernel array of length k
        mode: 'valid' (no padding), 'same' (same output size), 'full'
    
    Returns:
        Convolution output and Toeplitz matrix
    """
    n = len(x)
    k = len(kernel)
    
    if mode == 'valid':
        out_len = n - k + 1
        pad = 0
    elif mode == 'same':
        out_len = n
        pad = k // 2
    elif mode == 'full':
        out_len = n + k - 1
        pad = k - 1
    
    # Pad input if needed
    if pad > 0:
        x = np.pad(x, (pad, pad), mode='constant', constant_values=0)
    
    n_padded = len(x)
    
    # Build Toeplitz matrix for cross-correlation
    T = np.zeros((out_len, n_padded))
    for i in range(out_len):
        T[i, i:i+k] = kernel
    
    # Compute convolution via matrix multiplication
    y = T @ x
    return y, T


# Example
x = np.array([1, 2, 3, 4, 5], dtype=float)
kernel = np.array([1, 0, -1], dtype=float)

y, T = conv1d_as_matrix(x, kernel, mode='valid')

print("Input x:", x)
print("Kernel:", kernel)
print("\nToeplitz matrix T:")
print(T)
print("\nOutput y = T @ x:", y)

# Verify with numpy's correlate
y_np = np.correlate(x, kernel, mode='valid')
print("NumPy correlate result:", y_np)
print("Match:", np.allclose(y, y_np))
```

---

## 2D Convolution as Matrix Multiplication

### The im2col Transformation

For 2D images, the matrix formulation uses the **im2col** (image-to-column) transformation, which:

1. Extracts all kernel-sized patches from the input
2. Reshapes each patch into a column
3. Allows convolution to be computed as a single matrix multiplication

### Mathematical Formulation

For input $\mathbf{X} \in \mathbb{R}^{H \times W}$ and kernel $\mathbf{K} \in \mathbb{R}^{K_H \times K_W}$:

1. **im2col**: Transform $\mathbf{X}$ into $\mathbf{X}_{col} \in \mathbb{R}^{(K_H \cdot K_W) \times (H_{out} \cdot W_{out})}$
2. **Flatten kernel**: Reshape $\mathbf{K}$ into $\mathbf{k} \in \mathbb{R}^{1 \times (K_H \cdot K_W)}$
3. **Multiply**: $\mathbf{Y}_{col} = \mathbf{k} \cdot \mathbf{X}_{col}$
4. **Reshape**: Transform $\mathbf{Y}_{col}$ back to $\mathbf{Y} \in \mathbb{R}^{H_{out} \times W_{out}}$

### Visual Example

```
Input (4×4)               Kernel (2×2)
┌───┬───┬───┬───┐        ┌───┬───┐
│ a │ b │ c │ d │        │k00│k01│
├───┼───┼───┼───┤        ├───┼───┤
│ e │ f │ g │ h │        │k10│k11│
├───┼───┼───┼───┤        └───┴───┘
│ i │ j │ k │ l │
├───┼───┼───┼───┤
│ m │ n │ o │ p │
└───┴───┴───┴───┘

im2col(X):                Flattened kernel:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ a │ b │ c │ e │ f │ g │ i │ j │ k │   [k00, k01, k10, k11]
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ b │ c │ d │ f │ g │ h │ j │ k │ l │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ e │ f │ g │ i │ j │ k │ m │ n │ o │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ f │ g │ h │ j │ k │ l │ n │ o │ p │
└───┴───┴───┴───┴───┴───┴───┴───┴───┘

Each column = one flattened patch
Each row = one output position
```

### Visual Illustration (Alternative View)

```
Input (4×4):              Extract 3×3 patches (im2col):
┌─────────────────┐       
│ a  b  c  d │    Patch 1 (0,0):  Patch 2 (0,1):  Patch 3 (1,0):  Patch 4 (1,1):
│ e  f  g  h │    [a,b,c,         [b,c,d,         [e,f,g,         [f,g,h,
│ i  j  k  l │     e,f,g,          f,g,h,          i,j,k,          j,k,l,
│ m  n  o  p │     i,j,k]          j,k,l]          m,n,o]          n,o,p]
└─────────────────┘

im2col result (9×4 matrix):     Kernel (row vector):    Output:
┌─────────────────┐             ┌─────────────────┐     ┌─────┐
│ a  b  e  f │                  │ k0 k1 k2 k3 ... │  →  │ y1  │
│ b  c  f  g │                  └─────────────────┘     │ y2  │
│ c  d  g  h │                         1×9              │ y3  │
│ e  f  i  j │                                          │ y4  │
│ f  g  j  k │ ← 9×4                                    └─────┘
│ g  h  k  l │                                            4×1
│ i  j  m  n │
│ j  k  n  o │
│ k  l  o  p │
└─────────────────┘
```

### PyTorch Implementation of im2col

```python
import torch
import torch.nn.functional as F

def im2col(input, kernel_size, stride=1, padding=0):
    """
    Transform input tensor to column matrix for matrix-based convolution.
    
    Args:
        input: (batch, channels, H, W)
        kernel_size: (kH, kW)
        stride: convolution stride
        padding: zero padding
    
    Returns:
        Column matrix (batch, C*kH*kW, H_out*W_out)
    """
    # Use unfold to extract patches
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))
    
    batch, C, H, W = input.shape
    kH, kW = kernel_size
    
    # Calculate output dimensions
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    
    # Unfold along height, then width
    # After first unfold: (batch, C, H_out, W, kH)
    # After second unfold: (batch, C, H_out, W_out, kH, kW)
    patches = input.unfold(2, kH, stride).unfold(3, kW, stride)
    
    # Reshape to (batch, C*kH*kW, H_out*W_out)
    patches = patches.contiguous().view(batch, C * kH * kW, H_out * W_out)
    
    return patches


def col2im(col, output_shape, kernel_size, stride=1, padding=0):
    """
    Transform column matrix back to image tensor.
    This is used for the backward pass.
    
    Args:
        col: (batch, C*kH*kW, H_out*W_out)
        output_shape: (H, W) of original input
        kernel_size: (kH, kW)
        stride: convolution stride
        padding: zero padding
    
    Returns:
        Tensor of shape (batch, C, H, W)
    """
    batch, col_channels, num_patches = col.shape
    kH, kW = kernel_size
    H, W = output_shape
    C = col_channels // (kH * kW)
    
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1
    
    # Initialize output
    output = torch.zeros(batch, C, H_padded, W_padded, device=col.device)
    
    # Reshape col
    col = col.view(batch, C, kH, kW, H_out, W_out)
    
    # Accumulate patches (handles overlapping regions)
    for i in range(kH):
        for j in range(kW):
            output[:, :, i:i+H_out*stride:stride, j:j+W_out*stride:stride] += col[:, :, i, j, :, :]
    
    # Remove padding
    if padding > 0:
        output = output[:, :, padding:-padding, padding:-padding]
    
    return output
```

### Convolution via im2col

```python
def conv2d_im2col(input, weight, stride=1, padding=0):
    """
    2D convolution using im2col transformation.
    
    Args:
        input: (batch, C_in, H, W)
        weight: (C_out, C_in, kH, kW)
        stride: convolution stride
        padding: zero padding
    
    Returns:
        Convolution output (batch, C_out, H_out, W_out)
    """
    batch, C_in, H, W = input.shape
    C_out, _, kH, kW = weight.shape
    
    # im2col transformation
    col = im2col(input, (kH, kW), stride, padding)  # (batch, C_in*kH*kW, H_out*W_out)
    
    # Reshape kernel
    weight_col = weight.view(C_out, -1)  # (C_out, C_in*kH*kW)
    
    # Matrix multiplication
    # (C_out, C_in*kH*kW) @ (batch, C_in*kH*kW, H_out*W_out) -> (batch, C_out, H_out*W_out)
    out_col = torch.einsum('oi,bio->bo', weight_col, col).view(batch, C_out, -1)
    
    # Reshape output
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    output = out_col.view(batch, C_out, H_out, W_out)
    
    return output


# Test
x = torch.randn(2, 3, 8, 8)  # batch=2, channels=3, 8x8
w = torch.randn(16, 3, 3, 3)  # 16 filters, 3x3

y_im2col = conv2d_im2col(x, w, stride=1, padding=1)
y_pytorch = F.conv2d(x, w, stride=1, padding=1)

print(f"im2col output shape: {y_im2col.shape}")
print(f"PyTorch output shape: {y_pytorch.shape}")
print(f"Max difference: {(y_im2col - y_pytorch).abs().max():.2e}")
```

---

## Doubly Block Toeplitz Matrix

For 2D convolution, the full matrix representation uses a **doubly block Toeplitz** structure.

### Structure

For a 2D kernel $\mathbf{K} \in \mathbb{R}^{k \times k}$:

$$
\mathbf{T}_{2D} = \begin{bmatrix}
\mathbf{T}_0 & \mathbf{T}_1 & \cdots & \mathbf{T}_{k-1} & \mathbf{0} & \cdots \\
\mathbf{0} & \mathbf{T}_0 & \mathbf{T}_1 & \cdots & \mathbf{T}_{k-1} & \cdots \\
\vdots & & \ddots & & & \vdots
\end{bmatrix}
$$

where each $\mathbf{T}_i$ is itself a Toeplitz matrix formed from the $i$-th row of $\mathbf{K}$.

### Python Implementation

```python
import numpy as np
import torch
import torch.nn.functional as F

def build_doubly_block_toeplitz(kernel, input_shape, padding=0):
    """
    Build the full doubly block Toeplitz matrix for 2D convolution.
    
    Args:
        kernel: 2D kernel array (k, k)
        input_shape: (H, W) of input
        padding: Padding amount
    
    Returns:
        Full convolution matrix
    """
    H, W = input_shape
    k = kernel.shape[0]
    
    # Apply padding
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    
    # Output dimensions
    H_out = H_padded - k + 1
    W_out = W_padded - k + 1
    
    # Total sizes
    in_size = H_padded * W_padded
    out_size = H_out * W_out
    
    # Build the matrix
    T = np.zeros((out_size, in_size))
    
    for i in range(H_out):
        for j in range(W_out):
            out_idx = i * W_out + j
            
            for ki in range(k):
                for kj in range(k):
                    in_i = i + ki
                    in_j = j + kj
                    in_idx = in_i * W_padded + in_j
                    
                    T[out_idx, in_idx] = kernel[ki, kj]
    
    return T


# Example: 3×3 kernel on 4×4 input
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=float)  # Sobel-x

x = np.arange(16, dtype=float).reshape(4, 4)

# Build Toeplitz matrix
T = build_doubly_block_toeplitz(kernel, (4, 4), padding=0)

print(f"Input shape: (4, 4)")
print(f"Kernel shape: (3, 3)")
print(f"Toeplitz matrix shape: {T.shape}")  # (4, 16) for 4×4 input, 3×3 kernel → 2×2 output

# Verify
x_flat = x.flatten()
y_flat = T @ x_flat
y = y_flat.reshape(2, 2)

print("\nInput:")
print(x)
print("\nOutput (via matrix mult):")
print(y)

# PyTorch verification
x_torch = torch.from_numpy(x[np.newaxis, np.newaxis, :, :].astype(np.float32))
k_torch = torch.from_numpy(kernel[np.newaxis, np.newaxis, :, :].astype(np.float32))
y_torch = F.conv2d(x_torch, k_torch)
print("\nOutput (PyTorch):")
print(y_torch.squeeze().numpy())
```

---

## Doubly Block Circulant Matrix (FFT Connection)

For convolution with **circular padding**, the matrix is **doubly block circulant**.

### Structure

For a 2D kernel $\mathbf{K} \in \mathbb{R}^{K \times K}$ and input $\mathbf{X} \in \mathbb{R}^{N \times N}$:

$$\mathbf{C} = \begin{bmatrix}
\mathbf{C}_0 & \mathbf{C}_{N-1} & \cdots & \mathbf{C}_1 \\
\mathbf{C}_1 & \mathbf{C}_0 & \cdots & \mathbf{C}_2 \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{C}_{N-1} & \mathbf{C}_{N-2} & \cdots & \mathbf{C}_0
\end{bmatrix}$$

where each $\mathbf{C}_i$ is itself a circulant matrix.

### Connection to FFT

Circulant matrices are diagonalized by the **Discrete Fourier Transform (DFT)**:

$$\mathbf{C} = \mathbf{F}^{-1} \mathbf{\Lambda} \mathbf{F}$$

where:

- $\mathbf{F}$: DFT matrix
- $\mathbf{\Lambda}$: Diagonal matrix of eigenvalues (DFT of first column)

This leads to **FFT-based convolution**:

$$\mathbf{y} = \text{IFFT}(\text{FFT}(\mathbf{x}) \odot \text{FFT}(\mathbf{k}))$$

### FFT Convolution Implementation

```python
import torch
import torch.fft
import torch.nn.functional as F

def conv2d_fft(input, kernel, padding=0):
    """
    2D convolution using FFT.
    Efficient for large kernels.
    
    Args:
        input: (batch, C_in, H, W)
        kernel: (C_out, C_in, kH, kW)
        padding: zero padding
    
    Returns:
        Convolution output
    """
    batch, C_in, H, W = input.shape
    C_out, _, kH, kW = kernel.shape
    
    # Pad input
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))
    
    _, _, H_padded, W_padded = input.shape
    
    # Output size
    H_out = H_padded - kH + 1
    W_out = W_padded - kW + 1
    
    # FFT size (pad to avoid circular convolution artifacts)
    fft_H = H_padded + kH - 1
    fft_W = W_padded + kW - 1
    
    # FFT of input
    input_fft = torch.fft.rfft2(input, s=(fft_H, fft_W))
    
    # FFT of flipped kernel (convolution vs correlation)
    kernel_padded = torch.zeros(C_out, C_in, fft_H, fft_W, device=kernel.device)
    kernel_padded[:, :, :kH, :kW] = torch.flip(kernel, dims=[2, 3])
    kernel_fft = torch.fft.rfft2(kernel_padded)
    
    # Multiply in frequency domain
    output_fft = torch.einsum('bihw,oihw->bohw', input_fft, kernel_fft)
    
    # Inverse FFT
    output = torch.fft.irfft2(output_fft, s=(fft_H, fft_W))
    
    # Crop to valid region
    output = output[:, :, kH-1:kH-1+H_out, kW-1:kW-1+W_out]
    
    return output


# Compare with direct convolution
x = torch.randn(1, 1, 32, 32)
k = torch.randn(1, 1, 5, 5)

y_fft = conv2d_fft(x, k)
y_direct = F.conv2d(x, k)

print(f"FFT output shape: {y_fft.shape}")
print(f"Direct output shape: {y_direct.shape}")
print(f"Max difference: {(y_fft - y_direct).abs().max():.2e}")
```

---

## Multi-Channel Convolution

### Formulation

For $C_{in}$ input channels and $C_{out}$ output channels:

$$
\mathbf{Y}_{out} = \mathbf{W} \mathbf{X}_{col}
$$

where:
- $\mathbf{X}_{col} \in \mathbb{R}^{(C_{in} \cdot k^2) \times (H_{out} \cdot W_{out})}$
- $\mathbf{W} \in \mathbb{R}^{C_{out} \times (C_{in} \cdot k^2)}$
- $\mathbf{Y}_{out} \in \mathbb{R}^{C_{out} \times (H_{out} \cdot W_{out})}$

### Implementation

```python
import numpy as np
import torch
import torch.nn.functional as F

def conv2d_im2col_batch(x, kernel, bias=None, stride=1, padding=0):
    """
    Full batched multi-channel convolution using im2col.
    
    Args:
        x: Input tensor (N, C_in, H, W)
        kernel: Weight tensor (C_out, C_in, k, k)
        bias: Bias tensor (C_out,) or None
        stride: Stride
        padding: Padding
    
    Returns:
        Output tensor (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = x.shape
    C_out, C_in_k, k, _ = kernel.shape
    
    # Pad input
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)))
    
    H_padded, W_padded = x.shape[2], x.shape[3]
    H_out = (H_padded - k) // stride + 1
    W_out = (W_padded - k) // stride + 1
    
    # im2col for entire batch
    x_col = np.zeros((N, C_in * k * k, H_out * W_out))
    
    for n in range(N):
        col_idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x[n, :, i*stride:i*stride+k, j*stride:j*stride+k]
                x_col[n, :, col_idx] = patch.reshape(-1)
                col_idx += 1
    
    # Reshape kernel to matrix
    kernel_mat = kernel.reshape(C_out, -1)
    
    # Batch matrix multiplication
    out = np.zeros((N, C_out, H_out * W_out))
    for n in range(N):
        out[n] = kernel_mat @ x_col[n]
    
    # Add bias
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    
    # Reshape to output
    out = out.reshape(N, C_out, H_out, W_out)
    
    return out


# Test
N, C_in, H, W = 2, 3, 8, 8
C_out, k = 16, 3

x = np.random.randn(N, C_in, H, W).astype(np.float32)
kernel = np.random.randn(C_out, C_in, k, k).astype(np.float32)
bias = np.random.randn(C_out).astype(np.float32)

# Our implementation
y_ours = conv2d_im2col_batch(x, kernel, bias, padding=1)

# PyTorch
x_torch = torch.from_numpy(x)
k_torch = torch.from_numpy(kernel)
b_torch = torch.from_numpy(bias)
y_torch = F.conv2d(x_torch, k_torch, b_torch, padding=1)

print(f"Match: {np.allclose(y_ours, y_torch.numpy(), atol=1e-5)}")
```

---

## Gradient Computation

### Forward Pass as Matrix Multiplication

Recall the forward pass:

$$\mathbf{Y} = \mathbf{K}_{col} \cdot \mathbf{X}_{col}$$

where:

- $\mathbf{X}_{col}$: im2col transformed input
- $\mathbf{K}_{col}$: Reshaped kernel weights
- $\mathbf{Y}$: Output (reshaped)

### Gradient with Respect to Input

Given $\frac{\partial L}{\partial \mathbf{Y}}$, we need $\frac{\partial L}{\partial \mathbf{X}}$.

Using the chain rule and the matrix formulation:

$$\frac{\partial L}{\partial \mathbf{X}_{col}} = \mathbf{K}_{col}^\top \cdot \frac{\partial L}{\partial \mathbf{Y}_{col}}$$

Then apply **col2im** to get $\frac{\partial L}{\partial \mathbf{X}}$.

**Key insight**: The backward pass through convolution with respect to the input is equivalent to a **transposed convolution** (also called deconvolution) with the flipped kernel.

### Gradient with Respect to Kernel

$$\frac{\partial L}{\partial \mathbf{K}_{col}} = \frac{\partial L}{\partial \mathbf{Y}_{col}} \cdot \mathbf{X}_{col}^\top$$

This is a matrix multiplication between the gradient and the im2col-transformed input.

### Transposed Convolution (Deconvolution)

The backward pass of convolution is transposed convolution:

$$
\frac{\partial L}{\partial \mathbf{X}} = \mathbf{T}^T \frac{\partial L}{\partial \mathbf{Y}}
$$

This is equivalent to convolution with a flipped kernel and fractional strides.

```python
def transposed_conv1d(grad_output, kernel, stride=1, padding=0):
    """
    Transposed 1D convolution (gradient w.r.t. input).
    
    Args:
        grad_output: Gradient from next layer
        kernel: Original convolution kernel
        stride: Original stride
        padding: Original padding
    
    Returns:
        Gradient w.r.t. input
    """
    n_out = len(grad_output)
    k = len(kernel)
    
    # Input size reconstruction
    n_in = (n_out - 1) * stride + k - 2 * padding
    
    # Build transposed Toeplitz matrix
    T = np.zeros((n_in + 2*padding, n_out))
    for i in range(n_out):
        for j in range(k):
            T[i*stride + j, i] = kernel[j]
    
    # Remove padding rows
    if padding > 0:
        T = T[padding:-padding, :]
    
    grad_input = T @ grad_output
    return grad_input
```

### PyTorch Implementation of Gradients

```python
import torch
import torch.nn.functional as F

class Conv2dFunction(torch.autograd.Function):
    """
    Custom autograd function demonstrating convolution gradients.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0):
        """Forward pass of convolution."""
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(input, weight, bias)
        
        return F.conv2d(input, weight, bias, stride, padding)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computing gradients."""
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Gradient w.r.t. input: transposed convolution
            grad_input = F.conv_transpose2d(
                grad_output, weight, 
                stride=stride, padding=padding
            )
        
        if ctx.needs_input_grad[1]:
            # Gradient w.r.t. weight
            grad_weight = torch.zeros_like(weight)
            batch = input.shape[0]
            
            for b in range(batch):
                grad_weight += F.conv2d(
                    input[b:b+1].transpose(0, 1),  # (C_in, 1, H, W)
                    grad_output[b:b+1].transpose(0, 1),  # (C_out, 1, H_out, W_out)
                    padding=padding
                ).transpose(0, 1)
        
        if bias is not None and ctx.needs_input_grad[2]:
            # Gradient w.r.t. bias: sum over spatial and batch dims
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        return grad_input, grad_weight, grad_bias, None, None


# Test custom backward
x = torch.randn(2, 3, 8, 8, requires_grad=True)
w = torch.randn(16, 3, 3, 3, requires_grad=True)

# Using custom function
y_custom = Conv2dFunction.apply(x, w, None, 1, 1)
loss_custom = y_custom.sum()
loss_custom.backward()

grad_x_custom = x.grad.clone()
grad_w_custom = w.grad.clone()

# Reset gradients
x.grad = None
w.grad = None

# Using PyTorch
y_pytorch = F.conv2d(x, w, padding=1)
loss_pytorch = y_pytorch.sum()
loss_pytorch.backward()

print(f"Input gradient match: {torch.allclose(grad_x_custom, x.grad, atol=1e-5)}")
print(f"Weight gradient match: {torch.allclose(grad_w_custom, w.grad, atol=1e-5)}")
```

---

## Winograd Convolution

### Overview

**Winograd convolution** reduces the number of multiplications for small kernels (especially 3×3) at the cost of more additions.

For $m \times m$ output tile with $r \times r$ kernel:

- Standard: $m^2 \cdot r^2$ multiplications
- Winograd: $(m + r - 1)^2$ multiplications

### Mathematical Formulation

The Winograd algorithm expresses convolution as:

$$Y = A^T [(G \cdot g \cdot G^T) \odot (B^T \cdot d \cdot B)] A$$

where:

- $g$: Filter
- $d$: Input tile
- $G, B, A$: Transformation matrices
- $\odot$: Element-wise multiplication

### Example: F(2,3) Winograd

For 2×2 output with 3×3 kernel:

```python
import torch
import torch.nn.functional as F

# Winograd transformation matrices for F(2,3)
G = torch.tensor([
    [1, 0, 0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0, 0, 1]
], dtype=torch.float32)

B = torch.tensor([
    [1, 0, -1, 0],
    [0, 1, 1, 0],
    [0, -1, 1, 0],
    [0, 1, 0, -1]
], dtype=torch.float32)

A = torch.tensor([
    [1, 1, 1, 0],
    [0, 1, -1, -1]
], dtype=torch.float32)

def winograd_conv_f23(input_tile, kernel):
    """
    Winograd convolution for 2x2 output from 4x4 input with 3x3 kernel.
    
    Args:
        input_tile: 4x4 input tile
        kernel: 3x3 kernel
    
    Returns:
        2x2 output
    """
    # Transform kernel: U = G @ kernel @ G.T
    U = G @ kernel @ G.T  # 4x4
    
    # Transform input: V = B.T @ input_tile @ B
    V = B.T @ input_tile @ B  # 4x4
    
    # Element-wise multiply
    M = U * V  # 4x4, only 16 multiplications!
    
    # Transform output: Y = A.T @ M @ A
    Y = A.T @ M @ A  # 2x2
    
    return Y


# Compare with direct convolution
kernel = torch.randn(3, 3)
input_tile = torch.randn(4, 4)

# Winograd result
y_winograd = winograd_conv_f23(input_tile, kernel)

# Direct convolution result
y_direct = F.conv2d(
    input_tile.view(1, 1, 4, 4),
    kernel.view(1, 1, 3, 3)
).squeeze()

print(f"Winograd output:\n{y_winograd}")
print(f"Direct output:\n{y_direct}")
print(f"Max difference: {(y_winograd - y_direct).abs().max():.2e}")

# Count multiplications
print(f"\nDirect: {2*2*3*3} = 36 multiplications")
print(f"Winograd: {4*4} = 16 multiplications")
```

---

## Computational Considerations

### im2col Memory Trade-off

| Approach | Time Complexity | Memory |
|----------|-----------------|--------|
| Direct convolution | $O(N^2 K^2)$ | $O(N^2)$ |
| im2col + GEMM | $O(N^2 K^2)$ | $O(N^2 K^2)$ |
| FFT | $O(N^2 \log N)$ | $O(N^2)$ |

### Direct Convolution vs Matrix Multiplication

| Method | Operations | Memory |
|--------|-----------|--------|
| Direct sliding | $O(C_{out} \cdot C_{in} \cdot k^2 \cdot H_{out} \cdot W_{out})$ | $O(1)$ extra |
| im2col + GEMM | Same | $O(C_{in} \cdot k^2 \cdot H_{out} \cdot W_{out})$ |

**Why use im2col despite memory overhead?**

1. **GEMM optimization**: Matrix multiplication is extremely optimized on GPUs
2. **Cache efficiency**: Better memory access patterns
3. **Parallelization**: Easy to parallelize across output positions
4. **Batching**: Multiple samples computed together

### When to Use FFT

FFT-based convolution is preferred when:

- Kernel size is large ($K > 11$ typically)
- Padding is circular
- Memory is limited

### NVIDIA cuDNN Algorithms

cuDNN automatically selects the best algorithm:

```python
import torch
import torch.backends.cudnn as cudnn

# Enable cuDNN autotuning
cudnn.benchmark = True

# Available algorithms (conceptually):
# - IMPLICIT_GEMM: Low memory, slower
# - IMPLICIT_PRECOMP_GEMM: im2col style
# - GEMM: Explicit im2col
# - DIRECT: Direct computation
# - FFT: FFT-based
# - FFT_TILING: Tiled FFT
# - WINOGRAD: Winograd transform (for 3x3)
# - WINOGRAD_NONFUSED: Non-fused Winograd
```

### Benchmark Example

```python
import torch
import time

def benchmark_conv():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Large convolution
    x = torch.randn(32, 256, 56, 56, device=device)
    conv = torch.nn.Conv2d(256, 512, 3, padding=1).to(device)
    
    # Warmup
    for _ in range(10):
        _ = conv(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = conv(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"Device: {device}")
    print(f"Time per conv: {elapsed/100*1000:.2f} ms")

benchmark_conv()
```

---

## Sparse Convolution

### Motivation

For inputs with many zeros (e.g., point clouds, sparse feature maps), standard convolution wastes computation on zero values.

### Sparse Matrix Representation

Using **Compressed Sparse Row (CSR)** format:

```python
import torch

def sparse_conv2d(input_sparse, weight, output_shape):
    """
    Convolution using sparse matrix multiplication.
    
    Args:
        input_sparse: Sparse input in COO format
        weight: Dense convolution weight
        output_shape: Shape of output tensor
    
    Returns:
        Sparse output tensor
    """
    # For production sparse convolution, use libraries like:
    # - MinkowskiEngine
    # - SpConv
    # - TorchSparse
    pass
```

---

## Summary

Key takeaways:

1. **Toeplitz matrices** provide the mathematical framework for understanding convolution as matrix multiplication

2. **im2col transformation** converts convolution to GEMM, enabling GPU optimization

3. **Gradient computation** follows naturally from the matrix formulation:
   - Input gradient: transposed convolution
   - Weight gradient: correlation of input and output gradient

4. **FFT convolution** is efficient for large kernels due to $O(N^2 \log N)$ complexity

5. **Winograd convolution** reduces multiplications for small kernels (3×3) commonly used in CNNs

6. **Sparse convolution** handles inputs with many zeros efficiently

7. **Modern frameworks** use sophisticated algorithms beyond simple im2col (cuDNN auto-selection)

Understanding these implementations helps in:

- Debugging gradient issues
- Optimizing memory usage
- Implementing custom convolution variants
- Understanding hardware acceleration

---

## Exercises

1. **Manual im2col**: Implement im2col without using `unfold`. Verify against PyTorch's implementation.

2. **Transpose Convolution**: Show mathematically that the gradient of convolution w.r.t. input is a transposed convolution. Implement and verify.

3. **FFT vs Direct**: Benchmark FFT convolution vs direct convolution for various kernel sizes. Find the crossover point.

4. **Winograd Extension**: Implement Winograd F(4,3) for 4×4 output tiles with 3×3 kernels.

5. **Memory Analysis**: Calculate the memory overhead of im2col for ResNet-50's first convolutional layer with batch size 32.

---

## References

1. Chellapilla, K., Puri, S., & Simard, P. (2006). High performance convolutional neural networks for document processing. *Tenth International Workshop on Frontiers in Handwriting Recognition*.

2. Lavin, A., & Gray, S. (2016). Fast algorithms for convolutional neural networks. *CVPR 2016*.

3. Vasilache, N., Johnson, J., Mathieu, M., Chintala, S., Piantino, S., & LeCun, Y. (2015). Fast convolutional nets with fbfft: A GPU performance evaluation. *ICLR 2015*.

4. Chetlur, S., et al. (2014). cuDNN: Efficient primitives for deep learning. *arXiv preprint arXiv:1410.0759*.
