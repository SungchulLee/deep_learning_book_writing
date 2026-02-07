# Mixed Precision Training

## Overview

Mixed precision training uses lower-precision floating-point formats (FP16 or BF16) for computation while maintaining FP32 master copies of parameters for numerical stability. This reduces memory usage and accelerates training on modern GPUs with hardware support for half-precision arithmetic.

## Why Mixed Precision?

**Memory savings**: FP16 tensors use half the memory of FP32, enabling larger batch sizes or models.

**Speed**: Modern GPUs (NVIDIA Volta and later) have Tensor Cores optimized for FP16/BF16 matrix operations, achieving 2–8× higher throughput.

**Accuracy**: With proper loss scaling, mixed precision training matches FP32 accuracy for most tasks.

## PyTorch AMP (Automatic Mixed Precision)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast('cuda'):
        pred = model(x)
        loss = loss_fn(pred, y)

    # Backward pass with loss scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### How It Works

1. **`autocast`**: Automatically casts operations to FP16 where safe, keeping certain operations (e.g., loss computation, softmax, layer norm) in FP32 for numerical stability.

2. **`GradScaler`**: Scales the loss before backward to prevent FP16 gradient underflow. After backward, it unscales gradients before the optimizer step. If any gradients contain `inf` or `NaN` (due to overflow), the step is skipped.

## Gradient Clipping with AMP

When combining gradient clipping with mixed precision, unscale gradients first:

```python
scaler.scale(loss).backward()

# Unscale before clipping
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

## BFloat16

BFloat16 (BF16) has the same exponent range as FP32 but reduced mantissa precision. This eliminates the need for loss scaling:

```python
with autocast('cuda', dtype=torch.bfloat16):
    pred = model(x)
    loss = loss_fn(pred, y)

# No GradScaler needed with BF16
loss.backward()
optimizer.step()
```

BF16 is supported on NVIDIA Ampere (A100) and later GPUs, and on recent AMD and Intel hardware.

## Validation with Mixed Precision

Mixed precision can also accelerate validation:

```python
@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast('cuda'):
            pred = model(x)
            loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)
```

## When to Use Mixed Precision

**Recommended**: Large models, large batch sizes, models with many matrix multiplications (transformers, large CNNs).

**Caution**: Models with operations sensitive to precision (e.g., small-valued weights, high dynamic range activations). Always validate that mixed precision training matches FP32 performance on a small run.

## Key Takeaways

- Mixed precision uses FP16/BF16 for speed and memory while maintaining FP32 parameter copies for stability.
- `autocast` automatically selects precision per operation; `GradScaler` prevents gradient underflow.
- BFloat16 eliminates the need for loss scaling but requires newer hardware.
- Always validate that mixed precision matches FP32 accuracy for your specific model.
