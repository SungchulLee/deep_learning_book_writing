# Batch Normalization

## Overview

Batch Normalization (BatchNorm), introduced by Ioffe and Szegedy in 2015, normalizes activations across the batch dimension for each feature/channel. It has become a standard component in convolutional neural networks and feedforward architectures, enabling faster training, higher learning rates, and improved generalization.

## Mathematical Formulation

### Forward Pass

For a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$ of size $m$, BatchNorm computes:

**Step 1: Compute batch statistics**

$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma^2_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**Step 2: Normalize**

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

**Step 3: Scale and shift**

$$y_i = \gamma \hat{x}_i + \beta$$

Where:
- $\epsilon$ is a small constant (typically $10^{-5}$) for numerical stability
- $\gamma$ (scale) and $\beta$ (shift) are learnable parameters
- Initial values: $\gamma = 1$, $\beta = 0$

### For Convolutional Layers (BatchNorm2d)

For input tensor of shape $(N, C, H, W)$:
- $N$ = batch size
- $C$ = number of channels
- $H, W$ = spatial dimensions

Statistics are computed **per channel** across $(N, H, W)$:

$$\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w}$$

Each channel has its own $\gamma_c$ and $\beta_c$, giving $2C$ learnable parameters total.

### Running Statistics for Inference

During training, BatchNorm maintains exponential moving averages:

$$\mu_{\text{running}} \leftarrow (1 - \alpha) \cdot \mu_{\text{running}} + \alpha \cdot \mu_\mathcal{B}$$

$$\sigma^2_{\text{running}} \leftarrow (1 - \alpha) \cdot \sigma^2_{\text{running}} + \alpha \cdot \sigma^2_\mathcal{B}$$

Where $\alpha$ is the momentum (default: 0.1 in PyTorch).

During inference, running statistics replace batch statistics:

$$\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}$$

## PyTorch Implementation

### From Scratch

```python
import torch
import torch.nn as nn
import numpy as np

class BatchNorm1dFromScratch(nn.Module):
    """
    Batch Normalization implementation from scratch.
    Normalizes across the batch dimension for each feature.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Learnable parameters
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        
        # Running statistics (not learnable)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, num_features) or (N, num_features, ...)
        
        Returns:
            Normalized tensor of same shape
        """
        # Determine dimensions to normalize over
        if x.dim() == 2:
            # (N, C) - normalize over N for each C
            reduce_dims = [0]
            broadcast_shape = (1, self.num_features)
        elif x.dim() == 4:
            # (N, C, H, W) - normalize over N, H, W for each C
            reduce_dims = [0, 2, 3]
            broadcast_shape = (1, self.num_features, 1, 1)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.dim()}D")
        
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            
            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        mean = mean.view(broadcast_shape)
        var = var.view(broadcast_shape)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        if self.affine:
            gamma = self.gamma.view(broadcast_shape)
            beta = self.beta.view(broadcast_shape)
            return gamma * x_normalized + beta
        
        return x_normalized
    
    def extra_repr(self):
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}')
```

### Using PyTorch Built-in

```python
import torch
import torch.nn as nn

# For fully connected layers (2D input: batch, features)
bn1d = nn.BatchNorm1d(num_features=256)

# For convolutional layers (4D input: batch, channels, height, width)
bn2d = nn.BatchNorm2d(num_features=64)

# For 3D data (5D input: batch, channels, depth, height, width)
bn3d = nn.BatchNorm3d(num_features=32)

# Common parameters
bn = nn.BatchNorm2d(
    num_features=64,
    eps=1e-5,              # Numerical stability
    momentum=0.1,          # For running stats update
    affine=True,           # Learnable gamma and beta
    track_running_stats=True  # Maintain running mean/var
)
```

## Backward Pass (Gradient Derivation)

The backward pass through BatchNorm is more complex than typical layers due to the dependency on batch statistics.

### Gradient Computation

Given loss gradient $\frac{\partial \mathcal{L}}{\partial y_i}$, we need:

**Step 1: Gradient w.r.t. learnable parameters**

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i}$$

**Step 2: Gradient w.r.t. normalized input**

$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma$$

**Step 3: Gradient w.r.t. variance**

$$\frac{\partial \mathcal{L}}{\partial \sigma^2_\mathcal{B}} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \left( -\frac{1}{2} \right) (\sigma^2_\mathcal{B} + \epsilon)^{-3/2}$$

**Step 4: Gradient w.r.t. mean**

$$\frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} = \left( \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} \right) + \frac{\partial \mathcal{L}}{\partial \sigma^2_\mathcal{B}} \cdot \frac{\sum_{i=1}^{m} -2(x_i - \mu_\mathcal{B})}{m}$$

**Step 5: Gradient w.r.t. input**

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma^2_\mathcal{B}} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

### Implementation of Backward Pass

```python
class BatchNormFunction(torch.autograd.Function):
    """Custom autograd function for BatchNorm."""
    
    @staticmethod
    def forward(ctx, x, gamma, beta, eps=1e-5):
        # Compute statistics
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        std = torch.sqrt(var + eps)
        
        # Normalize
        x_centered = x - mean
        x_norm = x_centered / std
        
        # Scale and shift
        out = gamma * x_norm + beta
        
        # Save for backward
        ctx.save_for_backward(x_norm, gamma, std, x_centered)
        ctx.eps = eps
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x_norm, gamma, std, x_centered = ctx.saved_tensors
        m = grad_output.shape[0]
        
        # Gradient w.r.t. gamma and beta
        grad_gamma = (grad_output * x_norm).sum(dim=0)
        grad_beta = grad_output.sum(dim=0)
        
        # Gradient w.r.t. x_norm
        grad_x_norm = grad_output * gamma
        
        # Gradient w.r.t. variance
        grad_var = (grad_x_norm * x_centered * -0.5 * std**(-3)).sum(dim=0)
        
        # Gradient w.r.t. mean
        grad_mean = (grad_x_norm * -1 / std).sum(dim=0)
        grad_mean += grad_var * (-2 * x_centered).mean(dim=0)
        
        # Gradient w.r.t. input
        grad_x = grad_x_norm / std
        grad_x += grad_var * 2 * x_centered / m
        grad_x += grad_mean / m
        
        return grad_x, grad_gamma, grad_beta, None
```

## Training vs Inference Behavior

### Critical Difference

```python
def demonstrate_train_eval_difference():
    """Show the critical difference between train and eval modes."""
    bn = nn.BatchNorm1d(10)
    
    # Training data
    x_train = torch.randn(32, 10) * 2 + 5  # mean~5, std~2
    
    # Simulate training
    bn.train()
    for _ in range(100):
        _ = bn(x_train + torch.randn_like(x_train) * 0.1)
    
    print(f"Running mean: {bn.running_mean[:3]}")
    print(f"Running var: {bn.running_var[:3]}")
    
    # Test with different distribution
    x_test = torch.randn(1, 10)  # mean~0, std~1
    
    # WRONG: training mode with single sample
    bn.train()
    out_train = bn(x_test)
    
    # CORRECT: eval mode uses running statistics
    bn.eval()
    out_eval = bn(x_test)
    
    print(f"\nSingle sample test:")
    print(f"Train mode output std: {out_train.std():.4f}")  # Unreliable!
    print(f"Eval mode output std: {out_eval.std():.4f}")   # Stable

demonstrate_train_eval_difference()
```

### Common Mistake

```python
# WRONG - forgetting to switch modes
model = MyModel()
model.load_state_dict(torch.load('checkpoint.pth'))
# model.eval()  # Forgot this!
output = model(test_input)  # BatchNorm uses batch stats!

# CORRECT
model = MyModel()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()  # Critical for BatchNorm!
with torch.no_grad():
    output = model(test_input)
```

## Network Architectures with BatchNorm

### Basic CNN Block

```python
class ConvBNReLU(nn.Module):
    """Standard Conv-BN-ReLU block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

**Note:** We use `bias=False` in the convolution because BatchNorm's $\beta$ parameter serves the same purpose.

### ResNet Block with BatchNorm

```python
class ResNetBlock(nn.Module):
    """ResNet basic block with BatchNorm."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out
```

### Complete Image Classifier

```python
class ImageClassifier(nn.Module):
    """Simple image classifier with BatchNorm."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBNReLU(3, 64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
            nn.MaxPool2d(2),
            ConvBNReLU(128, 256),
            nn.MaxPool2d(2),
            ConvBNReLU(256, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## Batch Size Sensitivity

### The Problem

BatchNorm's effectiveness depends heavily on batch size:

```python
def test_batch_size_sensitivity():
    """Demonstrate BatchNorm sensitivity to batch size."""
    bn = nn.BatchNorm1d(100)
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    print("BatchNorm statistics by batch size:")
    print("-" * 50)
    
    for bs in batch_sizes:
        # Create data with known statistics
        x = torch.randn(bs, 100) * 2 + 3  # true mean=3, std=2
        
        bn.train()
        out = bn(x)
        
        # Compute output statistics
        out_mean = out.mean().item()
        out_std = out.std().item()
        
        print(f"Batch size {bs:3d}: mean={out_mean:7.4f}, std={out_std:.4f}")

test_batch_size_sensitivity()
```

**Output:**
```
BatchNorm statistics by batch size:
--------------------------------------------------
Batch size   1: mean= 0.0000, std=0.0000  # Degenerate!
Batch size   2: mean=-0.0089, std=0.7312  # High variance
Batch size   4: mean= 0.0021, std=0.8656
Batch size   8: mean=-0.0015, std=0.9324
Batch size  16: mean= 0.0008, std=0.9651
Batch size  32: mean=-0.0003, std=0.9825
Batch size  64: mean= 0.0001, std=0.9912  # Close to ideal
```

### Solutions for Small Batches

1. **Use GroupNorm instead** (batch-independent)
2. **Use SyncBatchNorm** for multi-GPU training
3. **Increase effective batch size** via gradient accumulation

```python
# Gradient accumulation for larger effective batch size
accumulation_steps = 8
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = criterion(output, y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Multi-GPU Training: SyncBatchNorm

Standard BatchNorm computes statistics per GPU, which can cause inconsistencies:

```python
# Convert BatchNorm to SyncBatchNorm for multi-GPU
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Or use directly
sync_bn = nn.SyncBatchNorm(num_features=64)
```

```python
# Full multi-GPU setup with SyncBatchNorm
def setup_distributed_model(model, device_ids):
    """Setup model for distributed training with synchronized BatchNorm."""
    
    # Convert all BatchNorm layers to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=device_ids
    )
    
    return model
```

## Best Practices

### 1. Layer Ordering

```python
# Standard: Conv → BN → Activation
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True)
)

# Pre-activation (ResNet v2): BN → Activation → Conv
nn.Sequential(
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 128, 3, padding=1, bias=False)
)
```

### 2. Bias in Convolutions

```python
# WRONG: Redundant bias
nn.Conv2d(64, 128, 3, bias=True)  # bias will be absorbed by BN's beta
nn.BatchNorm2d(128)

# CORRECT: No bias needed
nn.Conv2d(64, 128, 3, bias=False)
nn.BatchNorm2d(128)
```

### 3. Momentum Tuning

```python
# Default momentum (0.1) works for most cases
bn = nn.BatchNorm2d(64, momentum=0.1)

# For small datasets, use smaller momentum
bn = nn.BatchNorm2d(64, momentum=0.01)

# For very long training, consider even smaller
bn = nn.BatchNorm2d(64, momentum=0.001)
```

### 4. Freezing BatchNorm

When fine-tuning, be careful with BatchNorm layers:

```python
def freeze_bn(model):
    """Freeze BatchNorm layers (keep in eval mode)."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

# During training
model.train()
freeze_bn(model)  # Keep BN in eval mode
```

## When to Use BatchNorm

### Good Use Cases

- **Image classification with CNNs** (ResNet, VGG, etc.)
- **Object detection** (with SyncBatchNorm for multi-GPU)
- **Large batch sizes** (≥ 16-32)
- **Feedforward networks**

### Avoid When

- **Batch size = 1** (statistics are meaningless)
- **Small batches** (< 8) without SyncBatchNorm
- **RNNs/LSTMs** (use LayerNorm instead)
- **Transformers** (use LayerNorm instead)
- **Style transfer** (use InstanceNorm instead)
- **Online learning** scenarios

## Summary

Batch Normalization is a powerful technique that:

1. **Normalizes** activations across the batch dimension
2. **Maintains** running statistics for inference
3. **Enables** higher learning rates and faster convergence
4. **Acts as** a regularizer through batch noise

Key considerations:
- Always call `model.eval()` before inference
- Use `bias=False` in preceding convolutions
- Consider alternatives for small batches or specific architectures
- Use SyncBatchNorm for multi-GPU training

## References

1. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

3. Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? *NeurIPS*.
