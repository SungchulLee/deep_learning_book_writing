# Memory Pinning

## Overview

Memory pinning (page-locking) is an optimization that accelerates CPU-to-GPU data transfer. By allocating tensors in pinned (non-pageable) memory, the transfer to GPU can use DMA (Direct Memory Access) without involving the CPU, enabling asynchronous and overlapped transfers.

## How Pinned Memory Works

Normal (pageable) memory can be swapped to disk by the OS. Before transferring pageable data to the GPU, CUDA must first copy it to a temporary pinned buffer—a synchronous operation. Pinned memory skips this step:

```
Pageable memory → [copy to pinned buffer] → DMA to GPU  (slower, synchronous)
Pinned memory → DMA to GPU                               (faster, async-capable)
```

## Enabling in DataLoader

```python
loader = DataLoader(dataset, batch_size=64, pin_memory=True, num_workers=4)

for data, target in loader:
    # Transfer to GPU with non_blocking=True to overlap with computation
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
```

When `pin_memory=True`, the `DataLoader` automatically pins the output tensors in page-locked memory after collation. Combined with `non_blocking=True` in `.to()`, the CPU-GPU transfer can overlap with the next batch's preparation.

## Manual Pinning

For tensors outside the DataLoader pipeline:

```python
x = torch.randn(1000, 100).pin_memory()
assert x.is_pinned()

# Async transfer
x_gpu = x.to('cuda', non_blocking=True)
```

## Custom Pin Memory for Complex Batches

When the collate function returns non-standard types (e.g., nested dictionaries), implement a `pin_memory` method on the batch object:

```python
class CustomBatch:
    def __init__(self, data):
        self.features = data['features']
        self.targets = data['targets']
        self.mask = data['mask']

    def pin_memory(self):
        self.features = self.features.pin_memory()
        self.targets = self.targets.pin_memory()
        self.mask = self.mask.pin_memory()
        return self

def custom_collate(batch):
    # ... collation logic ...
    return CustomBatch(collated_data)

loader = DataLoader(dataset, collate_fn=custom_collate, pin_memory=True)
```

## When to Use Pinned Memory

**Use when**: Training on GPU with `num_workers > 0`. The benefit is most pronounced for large batch sizes and when data transfer is a bottleneck.

**Avoid when**: Training on CPU only, or when system RAM is limited (pinned memory cannot be swapped to disk).

## Performance Impact

The speedup from pinned memory is typically 2–3× for the data transfer step itself. The overall training speedup depends on how much time is spent on data transfer relative to computation:

- **Compute-bound** models (large models, small data): Minimal benefit.
- **I/O-bound** pipelines (small models, large images): Significant benefit.

## Key Takeaways

- `pin_memory=True` allocates DataLoader outputs in page-locked memory for faster GPU transfer.
- Combine with `non_blocking=True` in `.to()` to overlap transfer with computation.
- Pinned memory consumes physical RAM that cannot be swapped—use judiciously on memory-constrained systems.
