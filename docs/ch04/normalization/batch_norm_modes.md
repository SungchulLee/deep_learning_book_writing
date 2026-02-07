# Batch Normalization — Training vs Inference

Batch Normalization behaves differently during training and inference. This distinction is a frequent source of bugs and the single most important operational detail to understand when using BatchNorm. This page covers the mechanics, the running-statistics accumulation, common pitfalls, and best practices for mode switching.

## The Core Difference

### Training Mode

During training, BatchNorm computes **fresh statistics from the current mini-batch**:

$$\mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^m x_i, \qquad \sigma_{\mathcal{B}}^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$

These per-batch statistics serve two purposes: they normalise the activations for the forward/backward pass, and they incrementally update a set of **running statistics** stored in the module's buffers.

### Inference Mode

At inference time the mini-batch may contain a single example or have a different distribution from training. Using batch statistics would make predictions dependent on the batch composition, breaking determinism. Instead, BatchNorm uses the **accumulated running statistics**:

$$y_i = \gamma \cdot \frac{x_i - \hat{\mu}_{\text{run}}}{\sqrt{\hat{\sigma}^2_{\text{run}} + \epsilon}} + \beta$$

The output becomes a fixed affine transformation — deterministic and independent of what else is in the batch.

| Aspect | Training | Inference |
|--------|----------|-----------|
| Statistics source | Current mini-batch | Running buffers |
| Stochastic | Yes (depends on batch) | No (deterministic) |
| Gradient | Flows through $\mu, \sigma^2$ | Fixed transform, no stat gradients |
| Requires `model.train()` / `model.eval()` | `.train()` | `.eval()` |

## Running Statistics: Exponential Moving Average

### Update Rule

At each training step, the running statistics are updated via an exponential moving average (EMA):

$$\hat{\mu}_{\text{run}} \leftarrow (1 - \alpha)\,\hat{\mu}_{\text{run}} + \alpha\,\mu_{\mathcal{B}}$$

$$\hat{\sigma}^2_{\text{run}} \leftarrow (1 - \alpha)\,\hat{\sigma}^2_{\text{run}} + \alpha\,\sigma_{\mathcal{B}}^2$$

where $\alpha$ is the `momentum` parameter (default 0.1 in PyTorch).

!!! note "PyTorch's momentum convention"
    PyTorch uses `momentum` $= \alpha$, so a value of 0.1 means the running average assigns 10 % weight to the current batch and 90 % to the accumulated history. Some frameworks use the complementary convention $(1 - \alpha)$; always check the documentation.

### Bessel Correction for Running Variance

The forward pass uses the biased (population) variance $\frac{1}{m}\sum(x_i - \mu)^2$, but the contribution to the running variance applies Bessel's correction:

$$\hat{\sigma}^2_{\text{run}} \leftarrow (1 - \alpha)\,\hat{\sigma}^2_{\text{run}} + \alpha \cdot \frac{m}{m-1}\,\sigma_{\mathcal{B}}^2$$

This corrects for the downward bias of the sample variance estimator and becomes important when batch sizes are small.

### Initialisation

PyTorch initialises `running_mean` to zero and `running_var` to one. At the start of training the running statistics may not yet be representative, which is why it is common to see slightly different behaviour in the first few evaluation passes.

```python
import torch
import torch.nn as nn

bn = nn.BatchNorm2d(64)
print(f"running_mean: {bn.running_mean[:4]}")  # tensor([0., 0., 0., 0.])
print(f"running_var:  {bn.running_var[:4]}")    # tensor([1., 1., 1., 1.])
print(f"num_batches_tracked: {bn.num_batches_tracked}")  # tensor(0)
```

## Mode Switching in Practice

### The Golden Rule

```python
model.train()   # Before every training loop
model.eval()    # Before every evaluation / inference
```

`model.train()` and `model.eval()` recursively set the `training` flag on every sub-module. BatchNorm inspects this flag to decide which statistics to use.

### Forgetting `model.eval()` — The Most Common Bug

```python
# WRONG — BatchNorm still uses mini-batch statistics at test time
model.load_state_dict(torch.load("checkpoint.pt"))
output = model(test_input)

# CORRECT
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()
with torch.no_grad():
    output = model(test_input)
```

Symptoms of this bug:

- Test accuracy fluctuates between batches.
- Single-sample inference gives different results than the same sample in a larger batch.
- Metrics improve when the test batch size is increased.

### Verifying Mode

```python
def check_bn_mode(model):
    """Check that all BatchNorm layers are in eval mode."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.training:
                print(f"WARNING: {name} is in training mode")
            else:
                print(f"OK: {name} is in eval mode")
```

## Demonstrating the Train / Eval Difference

```python
import torch
import torch.nn as nn


def demonstrate_mode_difference():
    """Show how outputs differ between train and eval modes."""
    torch.manual_seed(42)

    bn = nn.BatchNorm1d(4)

    # Train for a few steps to build up running statistics
    bn.train()
    for _ in range(100):
        x = torch.randn(32, 4) * 2 + 3  # mean ≈ 3, std ≈ 2
        _ = bn(x)

    # Same input, different modes
    x_test = torch.randn(8, 4) * 2 + 3

    bn.train()
    out_train = bn(x_test)

    bn.eval()
    out_eval = bn(x_test)

    print("Output statistics (per feature, averaged over batch):")
    print(f"  Train mode — mean: {out_train.mean(0).tolist()}")
    print(f"  Eval  mode — mean: {out_eval.mean(0).tolist()}")
    print(f"  Max difference:    {(out_train - out_eval).abs().max():.4f}")

    # Single sample
    x_single = torch.randn(1, 4)
    bn.eval()
    out_single = bn(x_single)
    print(f"\nSingle-sample eval output: {out_single[0, :4].tolist()}")

demonstrate_mode_difference()
```

## Freezing BatchNorm for Fine-Tuning

When fine-tuning a pretrained model on a small dataset, the new data distribution may corrupt the pretrained running statistics. The standard practice is to **freeze BatchNorm layers** so they remain in eval mode even while the rest of the network trains.

### Method 1: Freeze Parameters and Mode

```python
def freeze_bn(model):
    """Freeze all BatchNorm layers: eval mode + no gradient."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

# Usage
model.train()
freeze_bn(model)  # BN stays in eval mode
```

!!! warning "Call `freeze_bn` after `model.train()`"
    `model.train()` recursively sets all modules to training mode, undoing per-module `eval()` calls. Always freeze BN **after** the global `model.train()`.

### Method 2: Override `train()` on the Model

```python
class FrozenBNModel(nn.Module):
    """Model that keeps BatchNorm in eval mode during training."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def train(self, mode=True):
        super().train(mode)
        # Override: keep BN in eval
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
        return self
```

### Method 3: Disable Running Stats Entirely

```python
bn = nn.BatchNorm2d(64, track_running_stats=False)
# Now uses batch statistics in both train and eval.
# Useful for meta-learning or when eval batches are large.
```

## Gradient Accumulation and Effective Batch Size

Gradient accumulation increases the effective batch size for the optimiser but **does not** increase BatchNorm's statistical batch size. Each forward pass still computes statistics from its own (small) mini-batch.

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

If the per-step batch size is too small for stable BN statistics, consider:

1. **SyncBatchNorm** across GPUs.
2. **GroupNorm** which is batch-independent.
3. **Ghost BatchNorm** which normalises over virtual sub-batches within a larger batch.

## SyncBatchNorm for Multi-GPU Training

With `DistributedDataParallel`, each GPU computes BN statistics from its own shard of the batch. For a global batch of 256 across 8 GPUs, each BN layer only sees 32 samples. `SyncBatchNorm` aggregates statistics across all GPUs:

```python
# Convert all BN layers to SyncBatchNorm
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Wrap with DDP
model = nn.parallel.DistributedDataParallel(
    model.cuda(), device_ids=[local_rank]
)
```

The synchronisation adds one all-reduce per BN layer per forward pass, which is a modest overhead relative to the gradient all-reduce.

## Momentum Tuning

The `momentum` parameter controls how quickly the running statistics adapt:

| Momentum | Effective window | Use case |
|----------|-----------------|----------|
| 0.1 (default) | ~10 recent batches | Standard training |
| 0.01 | ~100 recent batches | Small/noisy batches |
| 0.001 | ~1000 recent batches | Very long training schedules |

A smaller momentum gives more stable running statistics but adapts more slowly if the data distribution shifts during training (e.g., curriculum learning).

```python
# For small-batch or noisy training
bn = nn.BatchNorm2d(64, momentum=0.01)
```

## Fusing BatchNorm for Deployment

At inference, BatchNorm is a fixed affine transform that can be **fused** into the preceding linear or convolutional layer, eliminating the BN computation entirely.

For a convolution $y = W * x + b$ followed by BN with running statistics $\hat{\mu}, \hat{\sigma}^2$ and parameters $\gamma, \beta$:

$$y_{\text{fused}} = \frac{\gamma}{\sqrt{\hat{\sigma}^2 + \epsilon}}\,W * x + \left(\beta - \frac{\gamma\,\hat{\mu}}{\sqrt{\hat{\sigma}^2 + \epsilon}} + \frac{\gamma\,b}{\sqrt{\hat{\sigma}^2 + \epsilon}}\right)$$

```python
def fuse_conv_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d."""
    assert not bn.training, "BN must be in eval mode"

    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=True
    )

    # Compute fused weights
    bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data = conv.weight * bn_scale.view(-1, 1, 1, 1)

    # Compute fused bias
    if conv.bias is not None:
        fused.bias.data = bn_scale * (conv.bias - bn.running_mean) + bn.bias
    else:
        fused.bias.data = bn.bias - bn_scale * bn.running_mean

    return fused
```

Fusion is standard in production frameworks (TensorRT, ONNX Runtime, Core ML) and typically provides a 5–15 % inference speedup.

## Summary

| Topic | Key Point |
|-------|-----------|
| **Training** | Uses mini-batch $\mu, \sigma^2$; updates running stats via EMA |
| **Inference** | Uses running $\hat{\mu}, \hat{\sigma}^2$; deterministic |
| **Mode switching** | Always call `model.train()` / `model.eval()` |
| **Freezing for fine-tuning** | Keep BN in eval mode; disable `requires_grad` |
| **Gradient accumulation** | Does not increase BN's effective batch size |
| **Multi-GPU** | Use `SyncBatchNorm` for consistent cross-GPU statistics |
| **Deployment** | Fuse BN into preceding Conv/Linear for speed |

## References

1. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML*.
2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
3. Li, Y., et al. (2019). "Rethinking Batch Normalization in Meta-Learning." *NeurIPS*.
