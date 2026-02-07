# Online Continual Learning

Online continual learning is the most restrictive setting: data arrives as a stream and each sample is seen only once (single pass). The model must learn incrementally without revisiting past data.

## Problem Setting

Data arrives as a stream $(x_1, y_1), (x_2, y_2), ...$:

- Each sample is processed once and discarded
- No epoch-based training
- Task boundaries may be unknown (blurry)
- Memory buffer is limited (if available at all)

## Online Learning with Memory

```python
import torch
import torch.nn as nn
import random


class ReservoirBuffer:
    """
    Reservoir sampling buffer for online continual learning.
    
    Maintains a fixed-size buffer that uniformly samples from
    the entire stream seen so far.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.num_seen = 0
    
    def add(self, x, y):
        """Add a sample using reservoir sampling."""
        self.num_seen += 1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((x.clone(), y.clone()))
        else:
            # Replace with probability capacity / num_seen
            idx = random.randint(0, self.num_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (x.clone(), y.clone())
    
    def sample(self, batch_size):
        """Sample a batch from the buffer."""
        if len(self.buffer) == 0:
            return None, None
        
        indices = random.sample(range(len(self.buffer)), 
                               min(batch_size, len(self.buffer)))
        xs = torch.stack([self.buffer[i][0] for i in indices])
        ys = torch.stack([self.buffer[i][1] for i in indices])
        return xs, ys


def online_continual_training(model, data_stream, buffer_size=500,
                               replay_batch_size=32, lr=0.01, device='cuda'):
    """
    Online continual learning with experience replay.
    
    Each sample is seen exactly once. A replay buffer provides
    pseudo-rehearsal of past examples.
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    buffer = ReservoirBuffer(buffer_size)
    
    for step, (x, y) in enumerate(data_stream):
        x, y = x.to(device), y.to(device)
        
        model.train()
        optimizer.zero_grad()
        
        # Current sample loss
        logits = model(x.unsqueeze(0))
        loss = criterion(logits, y.unsqueeze(0))
        
        # Replay loss
        replay_x, replay_y = buffer.sample(replay_batch_size)
        if replay_x is not None:
            replay_x = replay_x.to(device)
            replay_y = replay_y.to(device)
            replay_logits = model(replay_x)
            replay_loss = criterion(replay_logits, replay_y)
            loss = 0.5 * loss + 0.5 * replay_loss
        
        loss.backward()
        optimizer.step()
        
        # Add to buffer
        buffer.add(x.cpu(), y.cpu())
        
        if step % 1000 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    return model
```

## Challenges Unique to Online CL

| Challenge | Description |
|-----------|-------------|
| Single pass | No revisiting data |
| Small batches | Often single-sample updates |
| Non-stationary | Distribution shifts over time |
| Blurry boundaries | Tasks may not have clear transitions |
| Catastrophic forgetting | Amplified by single-pass constraint |

## References

1. Aljundi, R., et al. (2019). "Online Continual Learning with Maximally Interfered Retrieval." *NeurIPS*.
2. Buzzega, P., et al. (2020). "Dark Experience for General Continual Learning." *NeurIPS*.
