# Performance Optimization

## Overview

Moving tensors to a GPU is a necessary first step, but it is not sufficient for achieving peak performance. Poorly structured training code can leave a GPU idle for the majority of each iteration—waiting for data, synchronising unnecessarily, or executing Python overhead. This section covers practical techniques for identifying and eliminating performance bottlenecks in PyTorch training pipelines.

---

## The Typical Training Bottleneck

A training iteration consists of three phases: **data loading**, **forward/backward computation**, and **optimizer update**. On a modern GPU, the compute phase may take only a few milliseconds for a moderately sized model, while data loading—reading files, decoding images, applying transforms, transferring to GPU—can take tens of milliseconds. If these phases run sequentially, the GPU sits idle during data loading:

```
CPU: [===load===]                    [===load===]
GPU:              [==compute==]                    [==compute==]
```

The goal of performance optimisation is to **overlap** these phases so the GPU is never starved:

```
CPU: [===load===][===load===][===load===]
GPU:              [==compute==][==compute==][==compute==]
```

---

## DataLoader Optimization

PyTorch's `DataLoader` is the primary mechanism for feeding data to the training loop. Its configuration has a large impact on throughput.

### Multi-Worker Loading

By default, `DataLoader` loads data in the main process, blocking computation. Setting `num_workers > 0` spawns separate processes for data loading:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,     # parallel data loading processes
    pin_memory=True,   # explained below
    prefetch_factor=2, # batches pre-loaded per worker
)
```

A reasonable starting point is `num_workers = min(4, os.cpu_count())`. Too many workers can cause CPU contention and actually slow things down.

### Pinned (Page-Locked) Memory

Setting `pin_memory=True` allocates CPU tensors in **page-locked memory**, which enables faster (and asynchronous) DMA transfers to GPU:

```python
dataloader = DataLoader(dataset, batch_size=64, pin_memory=True)

for batch_x, batch_y in dataloader:
    # non_blocking=True allows overlap with computation
    batch_x = batch_x.to("cuda", non_blocking=True)
    batch_y = batch_y.to("cuda", non_blocking=True)
```

The `non_blocking=True` argument makes the transfer asynchronous: the CPU immediately returns control to Python while the DMA engine transfers data in the background. This only works correctly with pinned memory.

### Persistent Workers

When `persistent_workers=True`, worker processes are kept alive between epochs, avoiding the overhead of process creation and dataset initialisation:

```python
dataloader = DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True,  # keep workers alive between epochs
)
```

This matters most when dataset initialisation is expensive (e.g., opening database connections, loading large index files).

---

## Avoiding Python Overhead

Python's interpreter introduces overhead that can dominate training time if not managed.

### Vectorise Operations

Replace Python loops with vectorised tensor operations:

```python
# Slow: Python loop over elements
result = torch.zeros(1000)
for i in range(1000):
    result[i] = x[i] * w[i] + b[i]

# Fast: vectorised operation
result = x * w + b
```

The vectorised version dispatches a single CUDA kernel; the loop version dispatches 1,000 separate operations with Python interpreter overhead between each.

### Avoid Unnecessary Tensor-to-Python Conversions

Calling `.item()`, `.numpy()`, or `print()` on a GPU tensor forces a **device synchronisation**: the CPU waits for all pending GPU operations to complete before transferring the value. In a training loop, this serialises execution:

```python
# Slow: synchronises GPU every iteration
for batch_x, batch_y in dataloader:
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print(f"Loss: {loss.item()}")  # forces GPU sync
    loss.backward()
    optimizer.step()

# Faster: log periodically
for i, (batch_x, batch_y) in enumerate(dataloader):
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item()}")
```

### `torch.no_grad()` for Inference

During evaluation, disabling gradient tracking avoids building the computational graph, saving both memory and computation:

```python
model.eval()
with torch.no_grad():
    for batch_x, batch_y in val_dataloader:
        batch_x = batch_x.to("cuda")
        output = model(batch_x)
        # no graph built, no gradient buffers allocated
```

---

## Batching Strategy

Batch size directly affects GPU utilisation. Small batches under-utilise the GPU's parallel capacity; large batches improve throughput up to the memory limit.

### Gradient Accumulation

When the desired effective batch size exceeds GPU memory, **gradient accumulation** simulates a larger batch by accumulating gradients over multiple forward-backward passes:

```python
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()
for i, (batch_x, batch_y) in enumerate(dataloader):
    batch_x = batch_x.to("cuda")
    batch_y = batch_y.to("cuda")

    output = model(batch_x)
    loss = criterion(output, batch_y) / accumulation_steps  # normalise

    loss.backward()  # gradients accumulate in .grad buffers

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

This produces mathematically equivalent gradients to a single large batch (assuming the loss is normalised) while keeping per-step memory bounded.

### Finding Optimal Batch Size

A practical approach: start with a small batch size and double it until either GPU memory is exhausted or throughput (samples/second) plateaus. PyTorch's memory management makes it easy to detect the limit:

```python
import torch

torch.cuda.empty_cache()
try:
    x = torch.randn(large_batch_size, input_dim, device="cuda")
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
except torch.cuda.OutOfMemoryError:
    print("Batch size too large — reduce and retry")
```

---

## Memory Optimization

GPU memory is a finite resource. Efficient memory use permits larger batches and larger models.

### `torch.cuda.empty_cache()`

PyTorch's CUDA memory allocator caches freed blocks for reuse. Calling `empty_cache()` releases these cached blocks back to the CUDA driver, making them available to other processes (but not necessarily increasing the memory available to the current process):

```python
torch.cuda.empty_cache()
```

This is useful between training phases (e.g., after validation) but should not be called every iteration—it defeats the purpose of the caching allocator.

### Gradient Checkpointing

For very deep models, activation memory (tensors saved for the backward pass) can dominate GPU memory. **Gradient checkpointing** trades compute for memory by recomputing activations during the backward pass instead of storing them:

```python
from torch.utils.checkpoint import checkpoint

class DeepModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = HeavyBlock()
        self.block2 = HeavyBlock()
        self.block3 = HeavyBlock()

    def forward(self, x):
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        x = checkpoint(self.block3, x, use_reentrant=False)
        return x
```

Checkpointing typically reduces activation memory by $O(\sqrt{L})$ for $L$ layers, at the cost of one additional forward pass through each checkpointed segment.

### In-Place Operations

Some operations can be performed in-place to avoid allocating new tensors:

```python
# Out-of-place: allocates new tensor
y = x.relu()

# In-place: modifies x directly
x.relu_()
```

Use in-place operations cautiously—they can interfere with autograd if the modified tensor is needed for gradient computation. PyTorch will raise an error if an in-place operation invalidates the computational graph.

---

## Profiling

Optimisation without measurement is guesswork. PyTorch provides built-in profiling tools to identify bottlenecks.

### `torch.utils.benchmark`

For microbenchmarking individual operations:

```python
import torch.utils.benchmark as benchmark

t = benchmark.Timer(
    stmt="x @ y",
    setup="x = torch.randn(1000, 1000, device='cuda'); y = torch.randn(1000, 1000, device='cuda')",
)
print(t.timeit(100))
```

### `torch.profiler`

For profiling entire training loops with detailed kernel-level timing:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to("cuda")
        batch_y = batch_y.to("cuda")

        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prof.step()
        if step >= 5:
            break
```

The output integrates with TensorBoard and shows per-kernel execution times, memory allocations, and CPU-GPU synchronisation points.

### `torch.cuda.Event` for Timing

CPU-side `time.time()` does not correctly measure GPU execution time because GPU operations are asynchronous. Use CUDA events instead:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(x)
end.record()

torch.cuda.synchronize()  # wait for GPU to finish
print(f"Forward pass: {start.elapsed_time(end):.2f} ms")
```

---

## `torch.compile` (PyTorch 2.0+)

`torch.compile` applies **graph-level optimisation** by capturing the computation graph and compiling it with the TorchDynamo/Inductor backend. This can fuse kernels, eliminate unnecessary memory allocations, and generate optimised GPU code:

```python
model = MyModel().to("cuda")
compiled_model = torch.compile(model)

# First call triggers compilation (slow); subsequent calls use compiled code
output = compiled_model(x)
```

Typical speedups range from 1.3–2× for standard architectures, with no changes to model code. Compilation works best with static shapes and standard operations; dynamic control flow and custom CUDA extensions may prevent compilation.

---

## Performance Checklist

A practical checklist for PyTorch training performance:

1. **Device placement:** All tensors and model parameters on GPU.
2. **DataLoader:** `num_workers > 0`, `pin_memory=True`, `non_blocking=True` transfers.
3. **Batch size:** As large as GPU memory allows (use gradient accumulation if needed).
4. **Mixed precision:** `autocast` + `GradScaler` (or bfloat16 on Ampere+).
5. **Avoid syncs:** Minimise `.item()`, `.numpy()`, `print()` calls inside training loops.
6. **Vectorise:** Replace Python loops with tensor operations.
7. **Inference mode:** `torch.no_grad()` during evaluation.
8. **Compile:** `torch.compile(model)` for static-graph models on PyTorch 2.0+.
9. **Profile:** Use `torch.profiler` to verify that the GPU is the bottleneck, not the CPU or data pipeline.

---

## Key Takeaways

- GPU utilisation depends as much on data pipeline efficiency and code structure as on the GPU itself.
- Overlapping data loading with computation (`num_workers`, `pin_memory`, `non_blocking`) is the single most impactful optimisation for many workloads.
- Python-level overhead (loops, synchronisation points, unnecessary `.item()` calls) can serialise GPU execution and must be eliminated.
- Gradient accumulation enables effective batch sizes larger than GPU memory.
- Profiling with `torch.profiler` and CUDA events provides the data needed to optimise systematically rather than by guesswork.
