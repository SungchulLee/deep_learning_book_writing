# DataLoader Fundamentals

## Learning Objectives

By the end of this section, you will be able to:

- Understand the Dataset → DataLoader → Batch pipeline
- Create DataLoaders with various configurations
- Master the iteration protocol for batch processing
- Analyze how batch size affects training dynamics

## Introduction

The DataLoader is one of PyTorch's most important utilities for training neural networks efficiently. It serves as the bridge between your raw data (stored in a Dataset) and the mini-batches that flow through your model during training.

### The Data Pipeline

The data flow in PyTorch follows a clear three-stage pipeline:

$$
\text{Dataset} \xrightarrow{\text{DataLoader}} \text{Batches} \xrightarrow{\text{Training Loop}} \text{Model}
$$

Each component has a specific responsibility:

| Component | Responsibility | Key Methods |
|-----------|---------------|-------------|
| **Dataset** | Storage and retrieval of individual samples | `__getitem__(idx)`, `__len__()` |
| **DataLoader** | Batching, shuffling, parallel loading | `__iter__()` |
| **Training Loop** | Model forward/backward passes | N/A |

### Why DataLoaders Matter

Without DataLoader, you would need to manually:

1. Sample indices from your dataset
2. Gather individual samples into batches
3. Handle shuffling between epochs
4. Manage parallel data loading
5. Transfer data to GPU efficiently

DataLoader handles all of this automatically, allowing you to focus on model architecture and training logic.

## Creating Your First DataLoader

### Minimal Dataset Example

Before using DataLoader, you need a Dataset. Here's a minimal example:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    """
    A minimal custom dataset for demonstration.
    
    Dataset Requirements:
    - __len__(): Returns total number of samples
    - __getitem__(idx): Returns the sample at index 'idx'
    """
    
    def __init__(self, num_samples=12, seed=0):
        gen = torch.Generator().manual_seed(seed)
        
        # Features: [num_samples, 4] tensor
        self.X = torch.randn(num_samples, 4, generator=gen)
        
        # Labels: binary classification based on feature sum
        self.y = (self.X.sum(dim=1) > 0).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### Basic DataLoader Creation

With a Dataset defined, creating a DataLoader is straightforward:

```python
dataset = SimpleDataset(num_samples=10, seed=42)

dataloader = DataLoader(
    dataset,
    batch_size=4,    # Group 4 samples per batch
    shuffle=False    # Keep original order
)
```

### Iterating Through Batches

The DataLoader is an iterable that yields batches:

```python
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Features shape: {features.shape}")  # [batch_size, 4]
    print(f"  Labels shape: {labels.shape}")      # [batch_size]
```

**Output:**
```
Batch 0:
  Features shape: torch.Size([4, 4])
  Labels shape: torch.Size([4])
Batch 1:
  Features shape: torch.Size([4, 4])
  Labels shape: torch.Size([4])
Batch 2:
  Features shape: torch.Size([2, 4])  # Last batch has remainder
  Labels shape: torch.Size([2])
```

!!! note "Last Batch Size"
    When the dataset size doesn't divide evenly by batch_size, the last batch contains the remainder. With 10 samples and batch_size=4: batches have sizes [4, 4, 2].

## The Iteration Protocol

Understanding how DataLoader iteration works is essential for advanced usage.

### For-Loop vs Manual Iteration

The for-loop automatically calls `iter()` on the DataLoader:

```python
# These are equivalent:

# Method 1: For-loop (automatic)
for batch in dataloader:
    process(batch)

# Method 2: Manual iteration
iterator = iter(dataloader)
while True:
    try:
        batch = next(iterator)
        process(batch)
    except StopIteration:
        break
```

### Manual Iteration Use Cases

Manual iteration is useful when you need fine-grained control:

```python
# Create an explicit iterator
batch_iterator = iter(dataloader)

# Fetch specific batches
batch1 = next(batch_iterator)  # First batch
batch2 = next(batch_iterator)  # Second batch

# Process remaining batches with for-loop
for batch in batch_iterator:
    process(batch)
```

### Fresh Iterators Per Epoch

Each call to `iter(dataloader)` creates a fresh iterator:

```python
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Epoch 1
iterator1 = iter(dataloader)
epoch1_batches = list(iterator1)

# Epoch 2 - new iterator, potentially different order
iterator2 = iter(dataloader)
epoch2_batches = list(iterator2)

# With shuffle=True, epoch1_batches != epoch2_batches
```

!!! important "Key Insight"
    Each `iter(dataloader)` call creates a **fresh** iterator. This is why for-loops work correctly across multiple epochs—each epoch gets a new iterator that starts from the beginning.

## Batch Size Analysis

The batch_size parameter has profound implications for training.

### Batch Count Calculation

For a dataset of size $N$ and batch size $B$:

$$
\text{Number of batches} = \lceil N / B \rceil
$$

```python
dataset = SimpleDataset(num_samples=10)

batch_sizes = [2, 3, 4, 5, 10]

for bs in batch_sizes:
    loader = DataLoader(dataset, batch_size=bs)
    batch_shapes = [features.shape[0] for features, _ in loader]
    
    print(f"batch_size={bs}: {len(batch_shapes)} batches, shapes={batch_shapes}")
```

**Output:**
```
batch_size=2:  5 batches, shapes=[2, 2, 2, 2, 2]
batch_size=3:  4 batches, shapes=[3, 3, 3, 1]
batch_size=4:  3 batches, shapes=[4, 4, 2]
batch_size=5:  2 batches, shapes=[5, 5]
batch_size=10: 1 batch,  shapes=[10]
```

### Impact on Training

| Aspect | Large Batch Size | Small Batch Size |
|--------|-----------------|------------------|
| **GPU Utilization** | Better (more parallelism) | Worse |
| **Memory Usage** | Higher | Lower |
| **Gradient Noise** | Lower (more stable) | Higher (may help generalization) |
| **Training Speed** | Faster per sample | Slower per sample |
| **Convergence** | May need LR scaling | More updates per epoch |

### Batch Size Selection Guidelines

**Rule of Thumb:** Start with batch_size=32, then adjust:

- **Increase** if GPU memory allows and training is stable
- **Decrease** if running out of memory or gradients are too smooth
- **Consider** the relationship: effective_lr ∝ batch_size

```python
# Common configurations by model type
small_models_batch = 64      # MLPs, small CNNs
medium_models_batch = 32     # ResNets, standard CNNs
large_models_batch = 16      # Large Transformers
very_large_models_batch = 8  # GPT-scale models
```

## Mathematical Foundation

### Stochastic Gradient Descent Connection

The DataLoader implements the batching required for mini-batch SGD. For a loss function $\mathcal{L}$, the gradient estimate over a mini-batch $\mathcal{B}$ is:

$$
\nabla_\theta \mathcal{L}_\mathcal{B}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta \ell(f_\theta(x_i), y_i)
$$

This is an unbiased estimate of the full gradient:

$$
\mathbb{E}_\mathcal{B}[\nabla_\theta \mathcal{L}_\mathcal{B}] = \nabla_\theta \mathcal{L}
$$

### Variance of Gradient Estimates

The variance of the mini-batch gradient decreases with batch size:

$$
\text{Var}[\nabla_\theta \mathcal{L}_\mathcal{B}] = \frac{\sigma^2}{|\mathcal{B}|}
$$

where $\sigma^2$ is the variance of individual sample gradients.

## Complete Example: DataLoader in Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Define Dataset
class RegressionDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=10, seed=42):
        torch.manual_seed(seed)
        self.X = torch.randn(n_samples, n_features)
        self.y = torch.randn(n_samples, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Create Dataset and DataLoader
dataset = RegressionDataset()
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(42)  # Reproducible shuffling
)

# 3. Define Model and Training Components
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Training Loop
num_epochs = 3

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
```

## Summary

| Concept | Key Point |
|---------|-----------|
| **DataLoader Purpose** | Wraps Dataset to produce batches for training |
| **Iteration** | Fresh iterator per epoch via `iter()` |
| **Batch Size** | Trade-off between memory, speed, and gradient stability |
| **Last Batch** | May be smaller if dataset doesn't divide evenly |

## Practice Exercises

1. **Batch Analysis**: Create a dataset with 17 samples. Calculate how many batches you'll get with batch_size=5, then verify with code.

2. **Manual Iteration**: Write code that processes only the first 3 batches using manual iteration, then skips to the last batch.

3. **Reproducibility**: Create two DataLoaders with the same generator seed and verify they produce identical batch sequences.

## What's Next

In the following sections, we'll explore:

- **Batching and Shuffling**: Deep dive into data randomization
- **DataLoader Parameters**: Complete configuration guide
- **Samplers**: Custom sampling strategies for imbalanced data
- **Collate Functions**: Handling variable-length sequences
