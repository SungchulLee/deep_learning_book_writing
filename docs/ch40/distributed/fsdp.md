# Fully Sharded Data Parallel (FSDP)

## Overview

FSDP shards model parameters, gradients, and optimizer states across data-parallel workers, dramatically reducing per-GPU memory usage. This enables training models that would not fit on a single GPU even with model parallelism.

## How FSDP Works

Unlike DDP (which replicates the full model):

```
DDP:  Each GPU holds full model copy
FSDP: Each GPU holds 1/N of parameters, gathers on demand
```

## Basic FSDP Usage

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Define wrapping policy
auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1_000_000)

# Wrap model
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float32,
    ),
)
```

## Memory Comparison

| Strategy | Memory per GPU (7B model, 8 GPUs) |
|----------|-----------------------------------|
| DDP | ~56 GB (full copy) |
| FSDP (params only) | ~7 GB params + optimizer |
| FSDP (full shard) | ~7 GB total |

## References

1. PyTorch FSDP Tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
2. FSDP API: https://pytorch.org/docs/stable/fsdp.html
