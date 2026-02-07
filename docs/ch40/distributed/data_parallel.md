# Data Parallel

## Overview

Data Parallelism distributes training across multiple GPUs by replicating the model on each device and splitting the data batch. PyTorch's `DataParallel` (DP) provides the simplest multi-GPU training with minimal code changes, though `DistributedDataParallel` is preferred for production workloads.

## `torch.nn.DataParallel`

```python
import torch
import torch.nn as nn

model = MyModel()

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.cuda()

# Training loop remains unchanged
for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## How DataParallel Works

```
1. Replicate model to all GPUs
2. Split input batch across GPUs
3. Forward pass on each GPU (parallel)
4. Gather outputs on GPU 0
5. Compute loss on GPU 0
6. Scatter gradients to all GPUs
7. Backward pass on each GPU (parallel)
8. Reduce gradients to GPU 0
9. Update parameters on GPU 0
10. Broadcast updated parameters
```

## Limitations

- **GPU 0 bottleneck**: All gathering/scattering happens on one GPU
- **GIL limitation**: Python's GIL limits true parallelism
- **Memory imbalance**: GPU 0 uses more memory than others
- **Single-machine only**: Cannot scale across nodes

For these reasons, `DistributedDataParallel` is recommended for production.

## References

1. PyTorch DataParallel: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
