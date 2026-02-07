# Pipeline Parallel

## Overview

Pipeline parallelism overlaps computation across GPUs by splitting micro-batches, reducing GPU idle time compared to naive model parallelism. Each GPU processes a different micro-batch at each time step, creating a pipeline of computation.

## GPipe-Style Pipeline

```
Time →  t1    t2    t3    t4    t5    t6
GPU 0: [F1]  [F2]  [F3]  [F4]  [B4]  [B3]  [B2]  [B1]
GPU 1:       [F1]  [F2]  [F3]  [B3]  [B2]  [B1]
GPU 2:             [F1]  [F2]  [B2]  [B1]
GPU 3:                   [F1]  [B1]

F = Forward, B = Backward, number = micro-batch
```

## PyTorch Pipeline API

```python
from torch.distributed.pipeline.sync import Pipe

# Create sequential model
model = nn.Sequential(
    nn.Linear(1024, 2048).to('cuda:0'),
    nn.ReLU().to('cuda:0'),
    nn.Linear(2048, 2048).to('cuda:1'),
    nn.ReLU().to('cuda:1'),
    nn.Linear(2048, 1024).to('cuda:1'),
)

# Wrap with Pipe (splits into micro-batches)
model = Pipe(model, chunks=8)

# Training
output = model(input_tensor)
loss = criterion(output.local_value(), target)
loss.backward()
optimizer.step()
```

## References

1. GPipe Paper: https://arxiv.org/abs/1811.06965
2. PyTorch Pipeline Tutorial: https://pytorch.org/docs/stable/pipeline.html
