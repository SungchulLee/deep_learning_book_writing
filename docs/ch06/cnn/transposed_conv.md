# Transposed Convolutions

## Introduction

**Transposed convolution** (also called **fractionally strided convolution** or, informally, **deconvolution**) is the gradient operation of regular convolution with respect to its input. While standard convolution typically reduces spatial dimensions, transposed convolution **increases** them, making it the primary learnable upsampling method in neural networks.

Transposed convolutions are essential components of:

- **Encoder-decoder architectures** (U-Net, SegNet) for semantic segmentation
- **Generative models** (GANs, VAEs) for image synthesis
- **Super-resolution networks** for upscaling images
- **Feature pyramid networks** for multi-scale object detection

> **Terminology Note**: The name "deconvolution" is technically incorrect (deconvolution is a specific signal processing operation that inverts convolution). "Transposed convolution" is the mathematically precise term, referring to multiplication by the transpose of the convolution's Toeplitz matrix.

---

## Mathematical Foundation

### Convolution as Matrix Multiplication

To understand transposed convolution, we first express standard convolution as matrix multiplication. For a 1D input $\mathbf{x} \in \mathbb{R}^5$ and kernel $\mathbf{k} = [k_0, k_1, k_2]$, the valid convolution $\mathbf{y} = \mathbf{C}\mathbf{x}$ uses:

$$\mathbf{C} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}$$

This maps $\mathbb{R}^5 \to \mathbb{R}^3$ (downsampling).

### The Transposed Operation

The **transposed convolution** uses $\mathbf{C}^\top$:

$$\mathbf{C}^\top = \begin{bmatrix}
k_0 & 0 & 0 \\
k_1 & k_0 & 0 \\
k_2 & k_1 & k_0 \\
0 & k_2 & k_1 \\
0 & 0 & k_2
\end{bmatrix}$$

This maps $\mathbb{R}^3 \to \mathbb{R}^5$ (upsampling). The same kernel weights are used, but the connectivity pattern is reversed.

### Key Insight: Gradient of Convolution

The backward pass of convolution w.r.t. its input is exactly a transposed convolution:

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{C}^\top \frac{\partial L}{\partial \mathbf{y}}$$

This is why transposed convolution naturally arises in backpropagation:

```python
import torch
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

## How Transposed Convolution Works

### The Upsampling Mechanism

Transposed convolution can be understood as:

1. **Insert zeros** between input elements (if stride > 1)
2. **Pad** the input
3. Apply a **standard convolution** with the same kernel

For stride-2 transposed convolution with a 3×3 kernel:

```
Input (2×2):       Insert zeros (3×3):      Pad (5×5):           Convolve (4×4 output):
┌───┬───┐          ┌───┬───┬───┐           ┌───┬───┬───┬───┬───┐
│ a │ b │    →      │ a │ 0 │ b │     →     │ 0 │ 0 │ 0 │ 0 │ 0 │  →  4×4 output
├───┼───┤          ├───┼───┼───┤           ├───┼───┼───┼───┼───┤
│ c │ d │          │ 0 │ 0 │ 0 │           │ 0 │ a │ 0 │ b │ 0 │
└───┴───┘          ├───┼───┼───┤           ├───┼───┼───┼───┼───┤
                   │ c │ 0 │ d │           │ 0 │ 0 │ 0 │ 0 │ 0 │
                   └───┴───┴───┘           ├───┼───┼───┼───┼───┤
                                           │ 0 │ c │ 0 │ d │ 0 │
                                           ├───┼───┼───┼───┼───┤
                                           │ 0 │ 0 │ 0 │ 0 │ 0 │
                                           └───┴───┴───┴───┴───┘
```

### Output Size Formula

For transposed convolution:

$$H_{out} = (H_{in} - 1) \times s - 2p + d(K - 1) + p_{out} + 1$$

where:
- $H_{in}$: Input height
- $s$: Stride
- $p$: Padding
- $d$: Dilation
- $K$: Kernel size
- $p_{out}$: Output padding (resolves ambiguity)

For the common case of stride-2 upsampling with 3×3 kernel:
$$H_{out} = (H_{in} - 1) \times 2 - 2 \times 1 + 3 + 1 = 2 \times H_{in}$$

---

## PyTorch Implementation

### Basic Usage

```python
import torch
import torch.nn as nn

# Regular convolution (downsample)
conv = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)

# Transposed convolution (upsample)
conv_transpose = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, 
                                     padding=1, output_padding=1)

x = torch.randn(1, 64, 32, 32)

# Downsample
y = conv(x)
print(f"Conv: {x.shape} → {y.shape}")  # [1, 64, 32, 32] → [1, 32, 16, 16]

# Upsample
z = conv_transpose(y)
print(f"ConvT: {y.shape} → {z.shape}")  # [1, 32, 16, 16] → [1, 64, 32, 32]
```

### The `output_padding` Parameter

When stride > 1, multiple input sizes can produce the same output size under regular convolution. For example, both 31×31 and 32×32 inputs with stride 2 produce 16×16 output. `output_padding` resolves this ambiguity:

```python
# Without output_padding: might get 31×31 instead of 32×32
conv_t_no_op = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
# With output_padding=1: guarantees 32×32
conv_t_with_op = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1)

y = torch.randn(1, 32, 16, 16)
print(f"Without output_padding: {conv_t_no_op(y).shape}")     # [1, 64, 31, 31]
print(f"With output_padding=1: {conv_t_with_op(y).shape}")    # [1, 64, 32, 32]
```

### Parameter Count

Transposed convolution has the same parameter count as regular convolution with swapped input/output channels:

```python
# Regular: maps 64 → 32 channels
conv = nn.Conv2d(64, 32, 3, bias=False)
print(f"Conv2d params: {sum(p.numel() for p in conv.parameters()):,}")
# 32 × 64 × 3 × 3 = 18,432

# Transposed: maps 32 → 64 channels  
conv_t = nn.ConvTranspose2d(32, 64, 3, bias=False)
print(f"ConvTranspose2d params: {sum(p.numel() for p in conv_t.parameters()):,}")
# 32 × 64 × 3 × 3 = 18,432 (same!)
```

---

## The Checkerboard Artifact Problem

### The Issue

Transposed convolutions with stride > 1 are notorious for producing **checkerboard artifacts**—a grid-like pattern in the output caused by uneven overlap of the kernel:

```
Stride-2 ConvTranspose with 3×3 kernel:

Contribution count at each output position:
┌───┬───┬───┬───┬───┬───┐
│ 1 │ 1 │ 2 │ 1 │ 2 │ 1 │    Uneven overlap creates
├───┼───┼───┼───┼───┼───┤    a checkerboard pattern
│ 1 │ 1 │ 2 │ 1 │ 2 │ 1 │    where some positions
├───┼───┼───┼───┼───┼───┤    receive more contributions
│ 2 │ 2 │ 4 │ 2 │ 4 │ 2 │    than others
├───┼───┼───┼───┼───┼───┤
│ 1 │ 1 │ 2 │ 1 │ 2 │ 1 │
└───┴───┴───┴───┴───┴───┘
```

### Solution 1: Kernel Size Divisible by Stride

Use kernel sizes that are evenly divisible by the stride:

```python
# Bad: stride=2, kernel=3 → uneven overlap
bad = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

# Better: stride=2, kernel=4 → even overlap
better = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

# Also good: stride=2, kernel=2 → no overlap
good = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
```

### Solution 2: Resize + Convolution (Recommended)

A cleaner alternative avoids transposed convolution entirely by separating upsampling from convolution:

```python
class UpsampleConv(nn.Module):
    """
    Upsample via interpolation + convolution.
    Avoids checkerboard artifacts from transposed convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 scale_factor=2, mode='bilinear'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode,
                                     align_corners=False if mode != 'nearest' else None)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


# Comparison
x = torch.randn(1, 64, 16, 16)

# Transposed convolution
conv_t = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)

# Resize + conv (artifact-free)
resize_conv = UpsampleConv(64, 32, scale_factor=2)

print(f"ConvTranspose: {conv_t(x).shape}")     # [1, 32, 32, 32]
print(f"Resize+Conv:   {resize_conv(x).shape}") # [1, 32, 32, 32]
```

### Solution 3: Sub-Pixel Convolution (PixelShuffle)

Used in super-resolution, this rearranges channels into spatial dimensions:

```python
class SubPixelUpsample(nn.Module):
    """
    Sub-pixel convolution (PixelShuffle) for efficient upsampling.
    
    Produces r² channels via regular conv, then rearranges to 
    increase spatial resolution by factor r.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor**2, 
                              kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.conv(x)
        return self.shuffle(x)


# Example
sub_pixel = SubPixelUpsample(64, 32, upscale_factor=2)
x = torch.randn(1, 64, 16, 16)
out = sub_pixel(x)
print(f"Sub-pixel: {x.shape} → {out.shape}")  # [1, 64, 16, 16] → [1, 32, 32, 32]
```

---

## Encoder-Decoder Architectures

### Simple Autoencoder

```python
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder using transposed convolutions for decoding.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder: progressively downsample
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),    # 224 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 112 → 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 56 → 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: progressively upsample
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 28 → 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 56 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 112 → 224
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model = ConvAutoencoder()
x = torch.randn(2, 3, 224, 224)
reconstruction = model(x)
print(f"Input: {x.shape}")
print(f"Reconstruction: {reconstruction.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### U-Net Style (with Skip Connections)

```python
class UNetDecoder(nn.Module):
    """
    U-Net decoder block with skip connections from encoder.
    
    Upsamples the feature map and concatenates with the corresponding
    encoder features, then applies convolutions.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Upsample (either transposed conv or resize+conv)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                      kernel_size=4, stride=2, padding=1)
        
        # After concatenation: (in_channels//2 + skip_channels) → out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatches (can occur with odd dimensions)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:])
        
        x = torch.cat([x, skip], dim=1)  # Concatenate along channels
        return self.conv(x)
```

---

## Comparison of Upsampling Methods

| Method | Learnable | Artifacts | Parameters | Speed |
|--------|-----------|-----------|------------|-------|
| **ConvTranspose2d** | Yes | Checkerboard (if K % s ≠ 0) | $C_{in} \times C_{out} \times K^2$ | Fast |
| **Bilinear + Conv** | Partially | Clean | $C_{in} \times C_{out} \times K^2$ | Medium |
| **Nearest + Conv** | Partially | Block-like | $C_{in} \times C_{out} \times K^2$ | Medium |
| **PixelShuffle** | Yes | Clean | $C_{in} \times C_{out} \times r^2 \times K^2$ | Fast |
| **Bilinear only** | No | Smooth (blurry) | 0 | Very fast |

```python
import torch
import torch.nn as nn

# All methods: 64 channels, 16×16 → 32×32

x = torch.randn(1, 64, 16, 16)

methods = {
    'ConvTranspose (K=4, s=2)': nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
    'ConvTranspose (K=2, s=2)': nn.ConvTranspose2d(64, 32, 2, stride=2),
    'PixelShuffle': nn.Sequential(
        nn.Conv2d(64, 32 * 4, 3, padding=1),
        nn.PixelShuffle(2)
    ),
    'Bilinear + Conv': nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, 3, padding=1)
    ),
}

for name, module in methods.items():
    out = module(x)
    params = sum(p.numel() for p in module.parameters())
    print(f"{name:<30}: {x.shape} → {out.shape}, params: {params:,}")
```

---

## 1D Transposed Convolution

Transposed convolutions also work in 1D for temporal upsampling:

```python
# 1D transposed convolution for temporal upsampling
conv_t1d = nn.ConvTranspose1d(
    in_channels=64,
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding=1
)

x = torch.randn(1, 64, 50)  # 50 time steps
out = conv_t1d(x)
print(f"1D ConvTranspose: {x.shape} → {out.shape}")  # [1, 32, 100]
```

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Operation** | Transpose of the convolution matrix—maps low-res to high-res |
| **Relationship to conv** | Gradient of conv2d w.r.t. input |
| **Output size** | $(H_{in}-1) \times s - 2p + d(K-1) + p_{out} + 1$ |
| **Common use** | Decoder networks, GANs, segmentation, super-resolution |
| **Main pitfall** | Checkerboard artifacts when $K \% s \neq 0$ |
| **Best practice** | Use $K$ divisible by $s$, or prefer resize + conv |

## Key Takeaways

1. **Transposed convolution is the transpose of the convolution matrix**, not its inverse—it does not undo convolution
2. **It arises naturally as the gradient** of regular convolution w.r.t. the input during backpropagation
3. **Checkerboard artifacts** are caused by uneven overlap when kernel size is not divisible by stride—use $K = 2s$ or $K = s$ to avoid them
4. **Resize + convolution** (bilinear interpolation followed by regular conv) is often preferred for artifact-free upsampling
5. **PixelShuffle** (sub-pixel convolution) provides efficient, artifact-free upsampling by rearranging channels
6. **output_padding** resolves the ambiguity when multiple input sizes map to the same output size under regular convolution

## References

1. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.

2. Long, J., Shelhamer, E., & Darrell, T. (2015). "Fully Convolutional Networks for Semantic Segmentation." *CVPR*.

3. Odena, A., Dumoulin, V., & Olah, C. (2016). "Deconvolution and Checkerboard Artifacts." *Distill*. https://distill.pub/2016/deconv-checkerboard/

4. Shi, W., et al. (2016). "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network." *CVPR*.

5. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.
