# Model Parallel

## Overview

Model parallelism splits a single model across multiple devices when the model is too large to fit in one GPU's memory. Unlike data parallelism (which replicates the model), model parallelism partitions the model's layers or tensors across devices.

## Pipeline vs. Tensor Parallelism

- **Pipeline Parallelism**: Split layers across GPUs sequentially
- **Tensor Parallelism**: Split individual layers across GPUs

## Basic Model Parallelism

```python
import torch
import torch.nn as nn

class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First half on GPU 0
        self.encoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
        ).to('cuda:0')
        
        # Second half on GPU 1
        self.decoder = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to('cuda:1')
    
    def forward(self, x):
        x = self.encoder(x.to('cuda:0'))
        x = self.decoder(x.to('cuda:1'))  # Transfer between GPUs
        return x
```

## Limitations

- **Sequential execution**: GPUs idle while waiting for upstream layers
- **Communication overhead**: Data transfers between GPUs add latency
- **Load imbalance**: Layers have different compute requirements

Pipeline parallelism addresses the sequential execution problem.

## References

1. PyTorch Model Parallel Tutorial: https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
