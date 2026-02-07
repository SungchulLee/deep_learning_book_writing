# Batch Inference

## Overview

Batch inference processes multiple inputs simultaneously, maximizing hardware utilization and throughput for offline prediction tasks. In quantitative finance, batch inference is essential for daily portfolio rebalancing, end-of-day risk calculations, and backtesting across large historical datasets.

## Throughput vs. Latency Trade-off

Larger batch sizes improve throughput but increase per-request latency:

$$\text{Throughput} = \frac{\text{batch\_size} \times \text{num\_batches}}{\text{total\_time}}$$

$$\text{Latency}_{\text{per\_sample}} = \frac{\text{batch\_latency}}{\text{batch\_size}}$$

## PyTorch Batch Inference

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import List, Dict

def batch_inference(model: nn.Module,
                   dataloader: DataLoader,
                   device: str = 'cpu') -> torch.Tensor:
    """Run batch inference over a DataLoader."""
    model.eval()
    model.to(device)
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())
    
    return torch.cat(all_predictions, dim=0)


def find_optimal_batch_size(model: nn.Module,
                            input_shape: tuple,
                            device: str = 'cuda',
                            max_batch_size: int = 1024) -> Dict:
    """Find optimal batch size by profiling throughput."""
    model.eval().to(device)
    results = {}
    
    batch_size = 1
    while batch_size <= max_batch_size:
        try:
            x = torch.randn(batch_size, *input_shape, device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            num_runs = 50
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            throughput = (batch_size * num_runs) / elapsed
            latency_ms = (elapsed / num_runs) * 1000
            
            results[batch_size] = {
                'throughput': throughput,
                'latency_ms': latency_ms,
                'samples_per_sec': throughput
            }
            
            batch_size *= 2
            
        except RuntimeError:  # OOM
            break
    
    # Find batch size with highest throughput
    best = max(results.items(), key=lambda x: x[1]['throughput'])
    print(f"Optimal batch size: {best[0]} "
          f"({best[1]['throughput']:.0f} samples/sec)")
    
    return results
```

## Dynamic Batching

Accumulate requests into batches for serving systems:

```python
import asyncio
import time
from collections import deque

class DynamicBatcher:
    """Accumulate requests into optimal batches."""
    
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
    
    async def predict(self, input_tensor):
        """Add request to batch queue and wait for result."""
        future = asyncio.Future()
        self.queue.append((input_tensor, future))
        
        if len(self.queue) >= self.max_batch_size:
            self._process_batch()
        else:
            await asyncio.sleep(self.max_wait_ms / 1000)
            if self.queue:
                self._process_batch()
        
        return await future
    
    def _process_batch(self):
        batch_items = []
        while self.queue and len(batch_items) < self.max_batch_size:
            batch_items.append(self.queue.popleft())
        
        inputs = torch.stack([item[0] for item in batch_items])
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        for i, (_, future) in enumerate(batch_items):
            future.set_result(outputs[i])
```

## Best Practices

- **Profile batch sizes** on target hardware to find throughput sweet spot
- **Use DataLoader** with `num_workers > 0` and `pin_memory=True` for GPU inference
- **Pre-allocate output tensors** for large batch jobs to avoid memory fragmentation
- **Consider memory limits** — maximum batch size is bounded by GPU/CPU memory
- **Use mixed precision** (`torch.cuda.amp.autocast()`) for GPU batch inference

## References

1. PyTorch Performance Tuning Guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
