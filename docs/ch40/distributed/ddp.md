# Distributed Data Parallel

## Overview

`DistributedDataParallel` (DDP) is PyTorch's recommended approach for multi-GPU and multi-node training. Unlike `DataParallel`, DDP uses separate processes per GPU, avoiding the GIL bottleneck and enabling efficient scaling across machines.

## Key Differences from DataParallel

| Feature | DataParallel | DistributedDataParallel |
|---------|-------------|------------------------|
| Process model | Single process, multi-thread | Multi-process |
| Communication | Gather on GPU 0 | AllReduce (NCCL) |
| GPU memory | Imbalanced | Balanced |
| Multi-node | No | Yes |
| Performance | Good for 2-4 GPUs | Scales to hundreds |

## Basic DDP Setup

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    optimizer = torch.optim.Adam(ddp_model.parameters())
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch
mp.spawn(train, args=(world_size,), nprocs=world_size)
```

## Launch with `torchrun`

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node (2 nodes, 4 GPUs each)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=node0 --master_port=29500 train.py
```

## Best Practices

- **Use `torchrun`** for launching (handles environment variables)
- **Set `sampler.set_epoch(epoch)`** for proper shuffling across epochs
- **Use `DistributedSampler`** to ensure non-overlapping data partitions
- **Save checkpoints from rank 0 only** to avoid file conflicts
- **Use NCCL backend** for GPU training (Gloo for CPU)
- **Scale learning rate** linearly with effective batch size

## References

1. PyTorch DDP Tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
2. PyTorch Distributed Overview: https://pytorch.org/docs/stable/distributed.html
