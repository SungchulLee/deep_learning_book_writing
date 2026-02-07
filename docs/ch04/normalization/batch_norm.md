# Batch Normalization

Batch Normalization (Ioffe & Szegedy, 2015) normalizes activations across the batch dimension for each feature or channel. It stabilises and accelerates training, enables higher learning rates, and provides implicit regularisation through mini-batch noise. BatchNorm has become a standard component in convolutional and feedforward architectures.

## Mathematical Formulation

### Forward Pass

For a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$ of activations at a given layer, BatchNorm operates independently on each feature dimension $j$:

**Step 1 — Mini-batch statistics:**

$$\mu_{\mathcal{B},j} = \frac{1}{m} \sum_{i=1}^{m} x_{i,j}, \qquad \sigma_{\mathcal{B},j}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_{i,j} - \mu_{\mathcal{B},j})^2$$

**Step 2 — Normalise:**

$$\hat{x}_{i,j} = \frac{x_{i,j} - \mu_{\mathcal{B},j}}{\sqrt{\sigma_{\mathcal{B},j}^2 + \epsilon}}$$

where $\epsilon > 0$ (typically $10^{-5}$) prevents division by zero.

**Step 3 — Scale and shift:**

$$y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j$$

The learnable parameters $\gamma_j$ (scale) and $\beta_j$ (shift) allow the network to recover the identity transform if optimal: setting $\gamma_j = \sigma_{\mathcal{B},j}$ and $\beta_j = \mu_{\mathcal{B},j}$ recovers the original activation.

### For Convolutional Layers (BatchNorm2d)

For input tensor of shape $(N, C, H, W)$, statistics are computed **per channel** across $(N, H, W)$:

$$\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n,h,w} x_{n,c,h,w}$$

Each channel has its own $\gamma_c$ and $\beta_c$, giving $2C$ learnable parameters total.

## Training vs Inference

### Training Mode

During training, BatchNorm uses mini-batch statistics and simultaneously maintains **running (exponential moving average) statistics**:

$$\hat{\mu}_{\text{running}} \leftarrow (1 - \alpha)\,\hat{\mu}_{\text{running}} + \alpha\,\mu_{\mathcal{B}}$$

$$\hat{\sigma}^2_{\text{running}} \leftarrow (1 - \alpha)\,\hat{\sigma}^2_{\text{running}} + \alpha\,\sigma_{\mathcal{B}}^2$$

where $\alpha$ is the momentum parameter (default 0.1 in PyTorch).

### Inference Mode

At inference the mini-batch may have a different size or contain a single example. BatchNorm uses the accumulated running statistics:

$$y_{i,j} = \gamma_j \cdot \frac{x_{i,j} - \hat{\mu}_{\text{running},j}}{\sqrt{\hat{\sigma}^2_{\text{running},j} + \epsilon}} + \beta_j$$

This makes inference deterministic and independent of batch composition.

!!! warning "Mode Switching"
    Always call `model.train()` before training and `model.eval()` before inference. Forgetting to switch modes is one of the most common BatchNorm bugs.

## Gradient Computation

The backward pass must differentiate through the mean and variance computations. Given loss gradient $\frac{\partial \mathcal{L}}{\partial y_i}$:

**Gradients w.r.t. learnable parameters:**

$$\frac{\partial \mathcal{L}}{\partial \gamma_j} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_{i,j}} \cdot \hat{x}_{i,j}, \qquad \frac{\partial \mathcal{L}}{\partial \beta_j} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_{i,j}}$$

**Gradient w.r.t. input** (simplified form):

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \frac{\partial \mathcal{L}}{\partial y_i} - \frac{1}{m} \sum_k \frac{\partial \mathcal{L}}{\partial y_k} - \frac{\hat{x}_i}{m} \sum_k \frac{\partial \mathcal{L}}{\partial y_k} \hat{x}_k \right)$$

The gradient is centred and scaled, which contributes to training stability.

## Why BatchNorm Regularises

BatchNorm was designed for training stability, but it has a notable regularisation effect:

1. **Noise from mini-batch statistics**: each example's normalised value depends on which other examples are in the mini-batch, injecting multiplicative and additive noise
2. **Stochasticity between iterations**: the same input produces slightly different outputs across mini-batches
3. **Smoother loss landscape**: BatchNorm has been shown to make the optimisation landscape smoother, reducing sensitivity to initialisation

The regularisation effect **decreases** as batch size increases (because mini-batch statistics become more stable).

## PyTorch Implementation

### Using Built-in Layers

```python
import torch.nn as nn

# For fully connected layers (N, C)
bn1d = nn.BatchNorm1d(num_features=256)

# For convolutional layers (N, C, H, W)
bn2d = nn.BatchNorm2d(num_features=64)

# Common parameters
bn = nn.BatchNorm2d(
    num_features=64,
    eps=1e-5,                 # numerical stability
    momentum=0.1,             # running stats update rate
    affine=True,              # learnable gamma and beta
    track_running_stats=True  # maintain running mean/var
)
```

### From Scratch

```python
import torch
import torch.nn as nn


class BatchNorm1dManual(nn.Module):
    """Manual BatchNorm1d for pedagogical clarity."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
```

### CNN with BatchNorm

```python
class ConvBNReLU(nn.Sequential):
    """Conv → BatchNorm → ReLU block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
            nn.MaxPool2d(2),
            ConvBNReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

## Batch Size Sensitivity

BatchNorm's effectiveness depends on batch size. With very small batches, the mini-batch statistics are noisy:

| Batch size | Behaviour |
|------------|-----------|
| 1 | Degenerate (variance = 0) |
| 2–4 | High variance in statistics |
| 8–16 | Acceptable for many tasks |
| 32+ | Stable statistics |

### Solutions for Small Batches

1. **Use Group Normalization** (batch-independent)
2. **Use SyncBatchNorm** for multi-GPU training
3. **Gradient accumulation** to increase effective batch size

```python
# Gradient accumulation
accumulation_steps = 8
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## SyncBatchNorm for Multi-GPU

Standard BatchNorm computes statistics per GPU. For consistency across devices, use synchronised BatchNorm:

```python
# Convert existing model
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# With DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank])
```

## Best Practices

### Layer Ordering

```python
# Standard: Conv → BN → Activation
nn.Conv2d(64, 128, 3, padding=1, bias=False)
nn.BatchNorm2d(128)
nn.ReLU(inplace=True)

# Pre-activation (ResNet v2): BN → Activation → Conv
nn.BatchNorm2d(64)
nn.ReLU(inplace=True)
nn.Conv2d(64, 128, 3, padding=1, bias=False)
```

### Remove Redundant Bias

The bias in a linear/conv layer before BatchNorm is absorbed by $\beta$:

```python
# Correct: no bias
nn.Conv2d(64, 128, 3, bias=False)
nn.BatchNorm2d(128)
```

### Combining with Dropout

When using both, apply dropout **after** BatchNorm and activation:

```python
nn.Linear(256, 128, bias=False)
nn.BatchNorm1d(128)
nn.ReLU()
nn.Dropout(0.1)  # lower rate than usual (BN already regularises)
```

### Freezing for Fine-Tuning

When fine-tuning a pretrained model, keep BatchNorm in eval mode:

```python
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

model.train()
freeze_bn(model)  # BN stays in eval mode
```

### Momentum Tuning

| Momentum | Use case |
|----------|----------|
| 0.1 (default) | Standard training |
| 0.01 | Small datasets, noisy batches |
| 0.001 | Very long training |

## When to Use BatchNorm

**Good use cases:**

- CNNs for image classification, detection, segmentation
- Feedforward networks with moderate-to-large batches
- ResNets, VGGs, and similar architectures

**Avoid when:**

- Batch size = 1 (use Instance Normalization or Layer Normalization)
- Small batches < 8 without SyncBatchNorm
- RNNs/LSTMs (use Layer Normalization)
- Transformers (use Layer Normalization)
- Style transfer (use Instance Normalization)

For alternatives, see [Layer Normalization](layer_norm.md), [Instance Normalization](instance_norm.md), and [Group Normalization](group_norm.md).

## Summary

| Aspect | Description |
|--------|-------------|
| **Normalisation** | Per-feature, across batch dimension |
| **Learnable parameters** | $\gamma$ (scale) and $\beta$ (shift) per feature |
| **Training** | Uses mini-batch statistics; updates running stats |
| **Inference** | Uses running statistics (deterministic) |
| **Regularisation** | Implicit, via mini-batch noise |
| **Key requirement** | Moderate batch size (≥ 8–16) |

## References

1. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML*.
2. Santurkar, S., et al. (2018). "How Does Batch Normalization Help Optimization?" *NeurIPS*.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
4. Wu, Y., & He, K. (2018). "Group Normalization." *ECCV*.
