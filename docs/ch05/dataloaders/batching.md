# Batching Strategies

## Overview

The choice of batch size and batching strategy directly affects training dynamics, convergence speed, and generalization. This section covers the tradeoffs involved and advanced batching techniques.

## Batch Size Effects

**Small batches** (8–32): Higher gradient noise acts as implicit regularization, often leading to better generalization. However, training is slower due to less parallelism and more frequent parameter updates.

**Large batches** (256–4096+): More accurate gradient estimates enable larger learning rates and faster wall-clock training. However, large batches can converge to sharp minima that generalize poorly without careful learning rate scaling.

The relationship between batch size $B$ and learning rate $\eta$ follows the **linear scaling rule**: when multiplying the batch size by $k$, multiply the learning rate by $k$ as well:

$$\eta_{\text{new}} = k \cdot \eta_{\text{base}}, \quad B_{\text{new}} = k \cdot B_{\text{base}}$$

This heuristic works well up to a critical batch size, beyond which diminishing returns set in.

## Fixed Batching

The default behavior: every batch contains exactly `batch_size` samples (except possibly the last batch):

```python
loader = DataLoader(dataset, batch_size=64, drop_last=False)
# Last batch may have fewer than 64 samples

loader = DataLoader(dataset, batch_size=64, drop_last=True)
# Last batch is dropped if incomplete — ensures uniform batch size
```

`drop_last=True` is useful when batch normalization requires consistent batch statistics or when the training loop assumes a fixed batch size.

## Gradient Accumulation

When GPU memory is insufficient for the desired effective batch size, gradient accumulation simulates larger batches:

```python
accumulation_steps = 4  # Effective batch = 64 * 4 = 256
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = loss_fn(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

The loss is divided by `accumulation_steps` so the accumulated gradient matches what a single large batch would produce.

## Dynamic Batching for Variable-Length Sequences

When samples have varying lengths (e.g., text, time series), fixed-size batches waste computation on padding. Dynamic batching groups similar-length samples together:

```python
from torch.utils.data import Sampler

class SortedBatchSampler(Sampler):
    """Group samples by length to minimize padding."""
    def __init__(self, lengths, batch_size):
        self.sorted_indices = sorted(range(len(lengths)),
                                     key=lambda i: lengths[i])
        self.batch_size = batch_size

    def __iter__(self):
        batches = [self.sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(self.sorted_indices), self.batch_size)]
        random.shuffle(batches)  # Shuffle batch order, not within batches
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size
```

## Quantitative Finance Considerations

For sequential financial data, batching must respect temporal structure. Time-series batches are typically formed from rolling windows, and care must be taken to avoid data leakage between overlapping windows within the same batch.

## Key Takeaways

- Batch size affects the bias-variance tradeoff of gradient estimates and generalization.
- The linear scaling rule adjusts learning rate proportionally to batch size changes.
- Gradient accumulation enables large effective batch sizes on memory-constrained hardware.
- Dynamic batching reduces wasted computation for variable-length sequences.
