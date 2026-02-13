# Mixed Precision

## Overview

Neural network training is dominated by matrix multiplications and element-wise operations on floating-point tensors. The **numerical precision** of these tensors—how many bits represent each number—directly affects both computation speed and numerical accuracy. **Mixed precision training** strategically uses lower-precision formats (float16, bfloat16) for bulk computation while retaining float32 for numerically sensitive operations, achieving significant speedups with minimal impact on model quality.

---

## Floating-Point Formats in PyTorch

PyTorch supports several floating-point formats, each with different bit widths, dynamic ranges, and precision characteristics:

| Format | Bits | Exponent | Mantissa | Dynamic Range | Decimal Digits |
|---|---|---|---|---|---|
| `torch.float64` (double) | 64 | 11 | 52 | $\sim 10^{\pm 308}$ | ~15–16 |
| `torch.float32` (float) | 32 | 8 | 23 | $\sim 10^{\pm 38}$ | ~7 |
| `torch.bfloat16` (brain float) | 16 | 8 | 7 | $\sim 10^{\pm 38}$ | ~2–3 |
| `torch.float16` (half) | 16 | 5 | 10 | $\sim 10^{\pm 4.8}$ | ~3–4 |

The key distinctions:

**float32** is the default precision in PyTorch and the baseline for neural network training. It provides sufficient dynamic range and precision for virtually all deep learning workloads.

**float16** halves memory usage and enables hardware-accelerated arithmetic on NVIDIA GPUs (Tensor Cores). However, its narrow dynamic range ($\sim 6 \times 10^{-8}$ to $6.5 \times 10^{4}$) means that small gradients can **underflow** to zero and large activations can **overflow** to infinity.

**bfloat16** retains the same exponent width (and hence the same dynamic range) as float32 while reducing mantissa precision. This makes it far less susceptible to overflow and underflow than float16, at the cost of slightly lower precision. It is natively supported on NVIDIA Ampere (A100) and later GPUs, as well as Google TPUs.

```python
import torch

x32 = torch.tensor(1.0, dtype=torch.float32)
x16 = torch.tensor(1.0, dtype=torch.float16)
xbf = torch.tensor(1.0, dtype=torch.bfloat16)

print(x32.element_size())  # 4 bytes
print(x16.element_size())  # 2 bytes
print(xbf.element_size())  # 2 bytes
```

---

## Why Mixed Precision Works

Using 16-bit formats throughout training is problematic: small gradient values underflow, loss values overflow, and accumulated rounding errors degrade convergence. Pure float32 training avoids these issues but leaves performance on the table.

Mixed precision resolves this tension by recognising that different parts of the computation have different precision requirements:

- **Matrix multiplications** (the dominant cost) are tolerant of reduced precision because errors in individual products are averaged over thousands of accumulations. These run in float16/bfloat16.
- **Reductions** (loss computation, gradient accumulation, batch normalisation statistics) require float32 to avoid catastrophic cancellation and overflow.
- **Weight updates** (optimizer step: $w \leftarrow w - \eta \cdot g$) require float32 because the update $\eta \cdot g$ may be orders of magnitude smaller than $w$, and float16 cannot represent the difference.

The standard mixed precision recipe maintains a **float32 master copy** of model weights. During each training step:

1. Cast weights to float16/bfloat16 for the forward and backward passes.
2. Compute gradients in float16/bfloat16.
3. Apply the optimizer update to the float32 master weights.

This achieves near-float16 speed with float32-level convergence.

---

## Automatic Mixed Precision in PyTorch

PyTorch provides `torch.cuda.amp` (**Automatic Mixed Precision**) to handle precision casting automatically:

### `autocast`: Automatic Dtype Selection

The `autocast` context manager intercepts operations and selects the appropriate precision:

```python
from torch.cuda.amp import autocast

model = MyModel().to("cuda")
x = torch.randn(32, 784, device="cuda")

with autocast():
    output = model(x)    # matmuls run in float16, reductions in float32
    loss = criterion(output, target)
```

Inside an `autocast` region, PyTorch applies per-operation precision rules:

- **float16:** `torch.mm`, `torch.matmul`, `torch.conv2d`, `torch.linear` — the compute-heavy operations.
- **float32:** `torch.sum`, `torch.mean`, `torch.softmax`, `torch.log`, `torch.batch_norm`, `torch.layer_norm`, `torch.cross_entropy` — operations where precision matters.

Tensors are cast automatically at each operation boundary. The programmer does not manually insert `.half()` or `.float()` calls.

### `GradScaler`: Preventing Gradient Underflow

When using float16 (not bfloat16), small gradient values can underflow to zero, stalling learning. **Gradient scaling** addresses this by:

1. Multiplying the loss by a large scale factor before `.backward()`, which scales all gradients proportionally into the representable float16 range.
2. Dividing gradients by the same factor before the optimizer step, restoring their true magnitude in float32.
3. Dynamically adjusting the scale factor: increasing it when no overflow is detected, decreasing it when `inf` or `NaN` gradients appear.

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

for batch_x, batch_y in dataloader:
    batch_x = batch_x.to("cuda")
    batch_y = batch_y.to("cuda")
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast():
        output = model(batch_x)
        loss = criterion(output, batch_y)

    # Scaled backward pass
    scaler.scale(loss).backward()

    # Unscale gradients, then step (skips step if inf/NaN detected)
    scaler.step(optimizer)

    # Update scale factor for next iteration
    scaler.update()
```

With **bfloat16**, gradient scaling is typically unnecessary because its dynamic range matches float32. The training loop simplifies to:

```python
with autocast(dtype=torch.bfloat16):
    output = model(batch_x)
    loss = criterion(output, batch_y)

loss.backward()
optimizer.step()
```

---

## Performance Gains

Mixed precision provides two sources of speedup:

**Faster arithmetic.** NVIDIA Tensor Cores execute float16/bfloat16 matrix multiplies at 2–8× the throughput of float32, depending on the GPU generation. On an A100, float16 Tensor Core throughput is 312 TFLOPS versus 19.5 TFLOPS for float32 (a 16× theoretical ratio, with practical gains of 2–3× in end-to-end training).

**Reduced memory.** Activations stored in float16 consume half the memory of float32. This permits either larger batch sizes (improving GPU utilisation and statistical efficiency) or larger models within the same GPU memory budget.

```python
# Memory comparison
batch_size, seq_len, hidden = 64, 512, 1024

fp32_activations = batch_size * seq_len * hidden * 4  # bytes
fp16_activations = batch_size * seq_len * hidden * 2  # bytes

print(f"float32: {fp32_activations / 1e6:.1f} MB")  # 134.2 MB
print(f"float16: {fp16_activations / 1e6:.1f} MB")   #  67.1 MB
```

---

## Numerical Stability Considerations

Mixed precision is not free of risk. Certain computations require care:

**Softmax and cross-entropy.** Computing $\text{softmax}(x_i) = \exp(x_i) / \sum_j \exp(x_j)$ in float16 is dangerous because $\exp(x)$ overflows for $x > 11$ in float16. PyTorch's `autocast` automatically runs these in float32.

**Accumulation.** Summing many small float16 values accumulates rounding error. Reduction operations (means, variances, norms) are kept in float32 by `autocast`.

**Small weight updates.** If a learning rate is $10^{-4}$ and a gradient is $10^{-3}$, the update magnitude is $10^{-7}$—below the representable precision of float16 relative to a weight of magnitude $1$. The float32 master copy ensures these updates are applied correctly.

### Precision in Quantitative Finance

Financial applications demand particular attention to numerical precision:

- **Pricing derivatives** often involves computing small differences between large numbers (e.g., discounted cashflows). Float16 precision (~3 decimal digits) is insufficient for this; critical pricing calculations should remain in float32 or float64.
- **Risk sensitivities** (Greeks) are computed via finite differences or automatic differentiation of pricing functions. Rounding error in the pricing function amplifies in the derivative, making precision especially important.
- **Mixed precision is safe** for the neural network components of a hybrid system (e.g., a neural SDE model) but the final pricing and risk outputs should be computed in full precision.

---

## Choosing Between float16 and bfloat16

| Criterion | float16 | bfloat16 |
|---|---|---|
| Hardware support | All CUDA GPUs with Tensor Cores (Volta+) | Ampere+ (A100, H100, RTX 30/40 series) |
| Dynamic range | Narrow ($10^{\pm 4.8}$) | Wide ($10^{\pm 38}$, same as float32) |
| Precision | ~3.3 decimal digits | ~2.4 decimal digits |
| Gradient scaling needed | Yes | Usually not |
| Recommended for | Inference; training with `GradScaler` | Training (simpler, more robust) |

If the hardware supports bfloat16, it is generally the better choice for training: the wider dynamic range eliminates most overflow/underflow issues, and the slight loss of mantissa precision rarely affects convergence.

---

## Key Takeaways

- Mixed precision uses float16 or bfloat16 for compute-heavy operations (matrix multiplies) and float32 for numerically sensitive operations (reductions, weight updates).
- PyTorch's `autocast` context manager handles per-operation precision selection automatically.
- `GradScaler` prevents gradient underflow when using float16; it is usually unnecessary with bfloat16.
- Practical speedups of 2–3× and 50% memory reduction are typical with minimal code changes.
- Financial applications should retain float32/float64 precision for final pricing and risk outputs, using mixed precision only within the neural network training components.
