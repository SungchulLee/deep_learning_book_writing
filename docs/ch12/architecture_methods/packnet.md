# PackNet

PackNet (Mallya & Lazebnik, 2018) packs multiple tasks into a single network by iteratively pruning and re-training. After learning each task, unimportant weights are freed for future tasks while important weights are frozen.

## Algorithm

1. **Train** on task $t$ using all available (non-frozen) parameters
2. **Prune** the least important parameters (e.g., bottom 75% by magnitude)
3. **Re-train** with only the remaining parameters
4. **Freeze** the remaining parameters
5. Free the pruned parameters for the next task

## Implementation

```python
import torch
import torch.nn as nn


class PackNet:
    """PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning."""
    
    def __init__(self, model, prune_ratio=0.75):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks = {}  # task_id -> parameter masks
        self.frozen_mask = {}  # Parameters frozen across all tasks
        
        # Initialise: all parameters available
        for n, p in model.named_parameters():
            self.frozen_mask[n] = torch.zeros_like(p, dtype=torch.bool)
    
    def train_task(self, task_id, train_loader, epochs, device='cuda'):
        """Train on a new task using available parameters."""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                
                # Zero gradients for frozen parameters
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        p.grad[self.frozen_mask[n]] = 0
                
                optimizer.step()
    
    def prune_and_freeze(self, task_id):
        """Prune unimportant weights and freeze the rest."""
        task_mask = {}
        
        for n, p in self.model.named_parameters():
            available = ~self.frozen_mask[n]
            available_weights = p.data.abs()[available]
            
            if available_weights.numel() == 0:
                task_mask[n] = torch.zeros_like(p, dtype=torch.bool)
                continue
            
            # Keep top (1 - prune_ratio) by magnitude
            threshold = torch.quantile(available_weights, self.prune_ratio)
            keep_mask = (p.data.abs() >= threshold) & available
            
            task_mask[n] = keep_mask
            self.frozen_mask[n] = self.frozen_mask[n] | keep_mask
            
            # Zero out pruned weights
            p.data[available & ~keep_mask] = 0
        
        self.masks[task_id] = task_mask
```

## Capacity Analysis

With 75% pruning per task, available capacity decreases geometrically:

| Task | Available params | Used params |
|------|-----------------|-------------|
| 1 | 100% | 25% |
| 2 | 75% | 18.75% |
| 3 | 56.25% | 14.06% |
| 4 | 42.19% | 10.55% |

## References

1. Mallya, A., & Lazebnik, S. (2018). "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning." *CVPR*.
