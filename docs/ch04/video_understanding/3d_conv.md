# 3D Convolutions for Video Understanding

## Learning Objectives

By the end of this section, you will be able to:

- Understand the mathematical formulation of 3D convolutions
- Implement 3D convolutional layers in PyTorch
- Build complete 3D CNN architectures (C3D, I3D, R3D)
- Analyze the computational trade-offs of 3D vs 2D approaches
- Design efficient 3D networks using factorization techniques

## Mathematical Foundation

### 3D Convolution Operation

For a video input $V \in \mathbb{R}^{T \times C \times H \times W}$ and a 3D kernel $K \in \mathbb{R}^{t_k \times C \times h_k \times w_k}$:

$$\text{Output}(\tau, i, j) = \sum_{t'=0}^{t_k-1} \sum_{c=0}^{C-1} \sum_{h'=0}^{h_k-1} \sum_{w'=0}^{w_k-1} V(\tau+t', c, i+h', j+w') \cdot K(t', c, h', w')$$

### Key Insight: Spatiotemporal Feature Learning

The critical difference from 2D convolution:

| Aspect | 2D Convolution | 3D Convolution |
|--------|---------------|----------------|
| Kernel | $(h_k, w_k)$ | $(t_k, h_k, w_k)$ |
| Output | Spatial features | Spatiotemporal features |
| Motion | Not captured | Directly learned |
| Parameters | $C_{in} \times C_{out} \times h_k \times w_k$ | $C_{in} \times C_{out} \times t_k \times h_k \times w_k$ |

For a $3 \times 3 \times 3$ kernel, 3D conv has **3× more parameters** than 2D.

### Output Dimension Calculation

For input $(T_{in}, H_{in}, W_{in})$ with kernel $(t_k, h_k, w_k)$, stride $(s_t, s_h, s_w)$, and padding $(p_t, p_h, p_w)$:

$$T_{out} = \frac{T_{in} + 2p_t - t_k}{s_t} + 1$$

$$H_{out} = \frac{H_{in} + 2p_h - h_k}{s_h} + 1$$

$$W_{out} = \frac{W_{in} + 2p_w - w_k}{s_w} + 1$$

## PyTorch Implementation

### Basic 3D Convolution

```python
import torch
import torch.nn as nn

# PyTorch Conv3d expects: (B, C, T, H, W)
# Note: Channels before time dimension!

conv3d = nn.Conv3d(
    in_channels=3,        # RGB input
    out_channels=64,      # Output feature maps
    kernel_size=(3, 3, 3),  # (temporal, height, width)
    stride=(1, 1, 1),
    padding=(1, 1, 1)     # Maintain spatial dimensions
)

# Example forward pass
video = torch.randn(2, 3, 16, 112, 112)  # (B, C, T, H, W)
output = conv3d(video)  # (2, 64, 16, 112, 112)
```

### Comparing 2D and 3D Convolutions

```python
class ConvolutionComparison:
    """
    Demonstrates the key differences between 2D and 3D convolutions.
    """
    
    @staticmethod
    def apply_2d_framewise(video: torch.Tensor, conv2d: nn.Conv2d):
        """
        Apply 2D conv to each frame independently.
        
        Args:
            video: (B, T, C, H, W)
            conv2d: 2D convolution layer
        
        Returns:
            Features with no temporal modeling
        """
        B, T, C, H, W = video.shape
        
        # Process all frames in parallel
        video_flat = video.view(B * T, C, H, W)
        features = conv2d(video_flat)  # (B*T, C_out, H', W')
        
        # Reshape back
        _, C_out, H_out, W_out = features.shape
        return features.view(B, T, C_out, H_out, W_out)
    
    @staticmethod
    def apply_3d(video: torch.Tensor, conv3d: nn.Conv3d):
        """
        Apply 3D conv to video volume.
        
        Args:
            video: (B, C, T, H, W) - Note: different format!
            conv3d: 3D convolution layer
        
        Returns:
            Spatiotemporal features
        """
        return conv3d(video)

# Demonstrate
conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv3d = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

video_2d_format = torch.randn(2, 16, 3, 112, 112)  # (B, T, C, H, W)
video_3d_format = torch.randn(2, 3, 16, 112, 112)  # (B, C, T, H, W)

out_2d = ConvolutionComparison.apply_2d_framewise(video_2d_format, conv2d)
out_3d = ConvolutionComparison.apply_3d(video_3d_format, conv3d)

print(f"2D per-frame output: {out_2d.shape}")  # (2, 16, 64, 112, 112)
print(f"3D spatiotemporal output: {out_3d.shape}")  # (2, 64, 16, 112, 112)
print(f"2D parameters: {sum(p.numel() for p in conv2d.parameters()):,}")
print(f"3D parameters: {sum(p.numel() for p in conv3d.parameters()):,}")
```

## Building Blocks

### 3D Convolutional Block

```python
class Conv3DBlock(nn.Module):
    """
    Standard 3D convolutional block with BatchNorm and ReLU.
    
    Architecture: Conv3D → BatchNorm3D → ReLU → (Optional) MaxPool3D
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple = (3, 3, 3),
                 stride: tuple = (1, 1, 1),
                 padding: tuple = (1, 1, 1),
                 use_pooling: bool = False,
                 pool_kernel: tuple = (2, 2, 2)):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # BatchNorm handles bias
        )
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.pool = None
        if use_pooling:
            self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.pool is not None:
            x = self.pool(x)
        
        return x
```

### Residual 3D Block

```python
class Residual3DBlock(nn.Module):
    """
    3D Residual block with skip connection.
    
    Enables training of very deep 3D networks by allowing
    gradient flow through skip connections.
    
    Mathematical formulation:
        y = F(x) + x
        
    where F(x) is the residual function.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 downsample_temporal: bool = False):
        super().__init__()
        
        stride = (2, 1, 1) if downsample_temporal else (1, 1, 1)
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels or downsample_temporal:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out
```

## Complete Architectures

### C3D (Convolutional 3D)

The foundational 3D CNN architecture:

```python
class C3D(nn.Module):
    """
    C3D: Learning Spatiotemporal Features with 3D Convolutional Networks
    (Tran et al., 2015)
    
    Key findings from the paper:
    - 3x3x3 kernels work best (empirically determined)
    - 8 conv layers with increasing depth
    - Standard input: 16 frames at 112×112 resolution
    
    Architecture:
        Conv1 → Pool1 → Conv2 → Pool2 → Conv3a → Conv3b → Pool3 →
        Conv4a → Conv4b → Pool4 → Conv5a → Conv5b → Pool5 →
        FC6 → FC7 → FC8 (output)
    """
    
    def __init__(self, num_classes: int = 101):
        super().__init__()
        
        # Block 1: 3 → 64 channels
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Block 2: 64 → 128 channels
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 3: 128 → 256 channels (2 conv layers)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 4: 256 → 512 channels (2 conv layers)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 5: 512 → 512 channels (2 conv layers)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), 
                                   padding=(0, 1, 1))
        
        # Fully connected layers
        # After all pooling: 16 frames → 1 temporal, 112x112 → 4x4
        self.fc6 = nn.Linear(512 * 1 * 4 * 4, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input video (B, 3, 16, 112, 112)
        Returns:
            Class logits (B, num_classes)
        """
        # Conv blocks with pooling
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        
        # Flatten and FC layers
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)
        
        return x
```

### R3D (ResNet 3D)

3D ResNet with residual connections:

```python
class R3D(nn.Module):
    """
    R3D: 3D ResNet for video classification.
    
    Applies ResNet architecture to 3D convolutions.
    Benefits:
    - Deeper networks with residual learning
    - Better gradient flow
    - State-of-the-art performance
    """
    
    def __init__(self, 
                 num_classes: int = 400,
                 layers: list = [2, 2, 2, 2]):  # ResNet-18 config
        super().__init__()
        
        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), 
                     stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Residual stages
        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1], downsample_temporal=True)
        self.layer3 = self._make_layer(128, 256, layers[2], downsample_temporal=True)
        self.layer4 = self._make_layer(256, 512, layers[3], downsample_temporal=True)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, 
                    downsample_temporal=False):
        layers = []
        
        # First block may downsample
        layers.append(Residual3DBlock(
            in_channels, out_channels, 
            downsample_temporal=downsample_temporal
        ))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(Residual3DBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x
```

### (2+1)D Factorized Convolution

Efficient factorization of 3D convolutions:

```python
class R2Plus1D(nn.Module):
    """
    R(2+1)D: A Closer Look at Spatiotemporal Convolutions
    (Tran et al., 2018)
    
    Key insight: Factor 3D conv into spatial 2D + temporal 1D:
        3D conv: (t, d, d) → 2D conv: (1, d, d) + 1D conv: (t, 1, 1)
    
    Benefits:
    - Same or fewer parameters
    - Doubles the number of nonlinearities
    - Easier to optimize
    - Better performance empirically
    """
    
    def __init__(self, num_classes: int = 400):
        super().__init__()
        
        self.stem = nn.Sequential(
            SpatioTemporalConv(3, 45, spatial_kernel=7, temporal_kernel=1,
                              stride=(1, 2, 2), padding=(0, 3, 3)),
            SpatioTemporalConv(45, 64, spatial_kernel=1, temporal_kernel=3,
                              stride=(1, 1, 1), padding=(1, 0, 0))
        )
        
        # Build ResNet-style network with (2+1)D convolutions
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_planes, planes, blocks, stride=1):
        layers = [R2Plus1DBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(R2Plus1DBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


class SpatioTemporalConv(nn.Module):
    """Factorized spatiotemporal convolution."""
    
    def __init__(self, in_channels, out_channels, 
                 spatial_kernel=3, temporal_kernel=3,
                 stride=(1, 1, 1), padding=(0, 0, 0)):
        super().__init__()
        
        # Intermediate channels for equivalent parameter count
        mid_channels = (temporal_kernel * spatial_kernel ** 2 * in_channels * out_channels) // \
                       (spatial_kernel ** 2 * in_channels + temporal_kernel * out_channels)
        
        # Spatial convolution
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels,
                     kernel_size=(1, spatial_kernel, spatial_kernel),
                     stride=(1, stride[1], stride[2]),
                     padding=(0, padding[1], padding[2]), bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Temporal convolution
        self.temporal = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels,
                     kernel_size=(temporal_kernel, 1, 1),
                     stride=(stride[0], 1, 1),
                     padding=(padding[0], 0, 0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        return x
```

## Computational Analysis

### Parameter Comparison

```python
def compare_conv_parameters():
    """Compare parameter counts of different approaches."""
    
    in_c, out_c = 64, 128
    
    # Full 3D conv: 3x3x3
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
    params_3d = sum(p.numel() for p in conv3d.parameters())
    
    # 2D conv: 3x3 (per frame)
    conv2d = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    params_2d = sum(p.numel() for p in conv2d.parameters())
    
    # (2+1)D factorized
    mid_c = (3 * 9 * in_c * out_c) // (9 * in_c + 3 * out_c)
    spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, 3, 3), padding=(0, 1, 1))
    temporal = nn.Conv3d(mid_c, out_c, kernel_size=(3, 1, 1), padding=(1, 0, 0))
    params_2p1d = sum(p.numel() for p in spatial.parameters()) + \
                  sum(p.numel() for p in temporal.parameters())
    
    print(f"2D Conv parameters: {params_2d:,}")
    print(f"3D Conv parameters: {params_3d:,} ({params_3d/params_2d:.1f}x 2D)")
    print(f"(2+1)D parameters: {params_2p1d:,} ({params_2p1d/params_3d:.2f}x 3D)")

compare_conv_parameters()
```

### FLOPs Analysis

```python
def compute_flops(in_shape, conv_layer):
    """Estimate FLOPs for convolution."""
    
    if isinstance(conv_layer, nn.Conv3d):
        B, C_in, T, H, W = in_shape
        C_out = conv_layer.out_channels
        kt, kh, kw = conv_layer.kernel_size
        
        # Output dimensions (assuming padding maintains size)
        T_out, H_out, W_out = T, H, W
        
        # FLOPs = 2 * K_t * K_h * K_w * C_in * C_out * T_out * H_out * W_out
        flops = 2 * kt * kh * kw * C_in * C_out * T_out * H_out * W_out
        return flops * B
    
    return 0
```

## Training Considerations

### Memory Management

3D CNNs require significant GPU memory:

```python
# Strategies for memory-efficient training

# 1. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientC3D(C3D):
    def forward(self, x):
        x = checkpoint(self._forward_block1, x)
        x = checkpoint(self._forward_block2, x)
        # ... etc
        return x

# 2. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(video)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 3. Reduced batch size + gradient accumulation
accumulation_steps = 4
for i, (videos, labels) in enumerate(dataloader):
    output = model(videos)
    loss = criterion(output, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Data Augmentation

```python
class Video3DAugmentation:
    """Augmentation pipeline for 3D CNN training."""
    
    def __init__(self, crop_size=(16, 112, 112)):
        self.crop_size = crop_size
    
    def __call__(self, video):
        # Random temporal crop
        T = video.shape[0]
        if T > self.crop_size[0]:
            start = torch.randint(0, T - self.crop_size[0] + 1, (1,)).item()
            video = video[start:start + self.crop_size[0]]
        
        # Random spatial crop
        _, _, H, W = video.shape
        top = torch.randint(0, H - self.crop_size[1] + 1, (1,)).item()
        left = torch.randint(0, W - self.crop_size[2] + 1, (1,)).item()
        video = video[:, :, top:top+self.crop_size[1], left:left+self.crop_size[2]]
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            video = video.flip(dims=[-1])
        
        return video
```

## Summary

| Architecture | Year | Parameters | Key Innovation |
|-------------|------|------------|----------------|
| C3D | 2015 | ~78M | First successful deep 3D CNN |
| I3D | 2017 | ~25M | Inflated 2D weights from ImageNet |
| R3D | 2017 | ~33M | ResNet + 3D convolutions |
| R(2+1)D | 2018 | ~33M | Factorized spatiotemporal convolutions |
| SlowFast | 2019 | ~34M | Dual-pathway different temporal rates |

### Key Takeaways

1. **3D convolutions** directly learn spatiotemporal features from raw video
2. **Trade-off**: More parameters and computation than 2D approaches
3. **Factorization** (2+1)D reduces parameters while improving performance
4. **Residual connections** enable training of very deep 3D networks
5. **Memory** is the primary bottleneck; use mixed precision and gradient checkpointing

## Next Steps

- **Two-Stream Networks**: Combining RGB and optical flow
- **Video Transformers**: Attention-based spatiotemporal modeling
- **Efficient Video Models**: MobileVideo, X3D for real-time applications
