# Samplers

## Learning Objectives

By the end of this section, you will be able to:

- Understand the relationship between samplers and shuffle
- Use WeightedRandomSampler for imbalanced datasets
- Implement custom samplers for specialized sampling strategies
- Choose appropriate sampling strategies for different scenarios

## What Are Samplers?

Samplers control **which samples** are drawn from a dataset and in **what order**. They provide the indices that the DataLoader uses to fetch samples.

### Sampler vs Shuffle

The `shuffle` parameter is actually just a convenience wrapper around samplers:

```python
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# These two are equivalent:
loader1 = DataLoader(dataset, shuffle=True)
loader2 = DataLoader(dataset, sampler=RandomSampler(dataset))

# These two are equivalent:
loader3 = DataLoader(dataset, shuffle=False)
loader4 = DataLoader(dataset, sampler=SequentialSampler(dataset))
```

### Why Use Explicit Samplers?

| Use Case | Solution |
|----------|----------|
| Class imbalance | WeightedRandomSampler |
| Distributed training | DistributedSampler |
| Stratified batches | Custom sampler |
| Curriculum learning | Custom sampler |
| Hard example mining | Custom sampler |

## Built-in Samplers

PyTorch provides several samplers in `torch.utils.data`:

### SequentialSampler

Returns indices in order: 0, 1, 2, ..., N-1

```python
from torch.utils.data import SequentialSampler

sampler = SequentialSampler(dataset)
indices = list(sampler)
# [0, 1, 2, 3, 4, 5, ..., N-1]
```

### RandomSampler

Returns indices in random order.

```python
from torch.utils.data import RandomSampler

# Basic random sampling
sampler = RandomSampler(dataset)

# With replacement (allows repeated samples)
sampler = RandomSampler(
    dataset,
    replacement=True,
    num_samples=1000  # Draw 1000 samples (can repeat)
)

# Reproducible random sampling
gen = torch.Generator().manual_seed(42)
sampler = RandomSampler(dataset, generator=gen)
```

### SubsetRandomSampler

Samples from a specific subset of indices.

```python
from torch.utils.data import SubsetRandomSampler

# Only sample from indices [0, 5, 10, 15, ...]
indices = list(range(0, len(dataset), 5))
sampler = SubsetRandomSampler(indices)
```

**Common use case**: Train/validation split

```python
# 80/20 train/val split
indices = list(range(len(dataset)))
split = int(0.8 * len(dataset))

train_sampler = SubsetRandomSampler(indices[:split])
val_sampler = SubsetRandomSampler(indices[split:])

train_loader = DataLoader(dataset, sampler=train_sampler)
val_loader = DataLoader(dataset, sampler=val_sampler)
```

## WeightedRandomSampler

The most important sampler for handling **class imbalance**.

### The Class Imbalance Problem

Consider a fraud detection dataset:

- 99% legitimate transactions (class 0)
- 1% fraudulent transactions (class 1)

With standard sampling, the model rarely sees fraudulent examples, leading to poor fraud detection.

### Solution: Weighted Sampling

Assign higher sampling weights to minority class samples:

```python
from torch.utils.data import WeightedRandomSampler

# Step 1: Get class labels
labels = dataset.targets  # [0, 0, 1, 0, 0, 0, 1, ...]

# Step 2: Compute class counts
class_counts = torch.bincount(torch.tensor(labels))
# tensor([990, 10])  # 990 class-0, 10 class-1

# Step 3: Compute class weights (inverse frequency)
class_weights = 1.0 / class_counts.float()
# tensor([0.0010, 0.1000])

# Step 4: Assign per-sample weights
sample_weights = class_weights[labels]
# Each sample gets its class weight

# Step 5: Create WeightedRandomSampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True  # Required for oversampling minority
)

loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

### Verifying Balance

```python
from collections import Counter

# Count class distribution across one epoch
all_labels = []
for batch_x, batch_y in loader:
    all_labels.extend(batch_y.tolist())

distribution = Counter(all_labels)
print(f"Class 0: {distribution[0]} ({distribution[0]/len(all_labels):.1%})")
print(f"Class 1: {distribution[1]} ({distribution[1]/len(all_labels):.1%})")
# Output: ~50/50 distribution
```

### Custom Weight Schemes

You're not limited to inverse frequency:

```python
# Scheme 1: Square root of inverse frequency (less aggressive)
class_weights = 1.0 / torch.sqrt(class_counts.float())

# Scheme 2: Effective number of samples (for long-tailed distributions)
beta = 0.9999
effective_num = 1.0 - torch.pow(beta, class_counts.float())
class_weights = (1.0 - beta) / effective_num

# Scheme 3: Custom weights based on importance
class_weights = torch.tensor([1.0, 10.0])  # 10x weight for class 1
```

## Custom Samplers

For specialized sampling strategies, implement the `Sampler` interface:

```python
from torch.utils.data import Sampler

class CustomSampler(Sampler):
    """
    Custom sampler must implement:
    - __iter__(): Yield indices one by one
    - __len__(): Return total number of samples
    """
    
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self):
        # Yield indices in your custom order
        for idx in range(len(self.data_source)):
            yield idx
    
    def __len__(self):
        return len(self.data_source)
```

### Example: Balanced Batch Sampler

Guarantee equal class representation in every batch:

```python
class BalancedBatchSampler(Sampler):
    """
    Creates batches with equal samples from each class.
    
    For batch_size=8 with 2 classes:
    Each batch contains 4 samples from each class.
    """
    
    def __init__(self, labels, batch_size, seed=0):
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even")
        
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2
        self.gen = torch.Generator().manual_seed(seed)
        
        # Separate indices by class
        self.class0_indices = (labels == 0).nonzero(as_tuple=True)[0].tolist()
        self.class1_indices = (labels == 1).nonzero(as_tuple=True)[0].tolist()
        
        # Number of complete batches
        min_class_size = min(len(self.class0_indices), len(self.class1_indices))
        self.num_batches = min_class_size // self.samples_per_class
    
    def __iter__(self):
        # Shuffle within each class
        perm0 = torch.randperm(len(self.class0_indices), generator=self.gen)
        perm1 = torch.randperm(len(self.class1_indices), generator=self.gen)
        
        shuffled0 = [self.class0_indices[i] for i in perm0.tolist()]
        shuffled1 = [self.class1_indices[i] for i in perm1.tolist()]
        
        # Yield balanced batches
        for i in range(self.num_batches):
            start = i * self.samples_per_class
            end = start + self.samples_per_class
            
            batch = shuffled0[start:end] + shuffled1[start:end]
            
            # Shuffle within batch
            batch_perm = torch.randperm(len(batch), generator=self.gen)
            batch = [batch[j] for j in batch_perm.tolist()]
            
            for idx in batch:
                yield idx
    
    def __len__(self):
        return self.num_batches * self.batch_size


# Usage
labels = dataset.targets  # Assume this is a tensor
sampler = BalancedBatchSampler(labels, batch_size=8)
loader = DataLoader(dataset, batch_size=8, sampler=sampler)
```

### Example: Curriculum Learning Sampler

Start with easy samples, gradually introduce harder ones:

```python
class CurriculumSampler(Sampler):
    """
    Implements curriculum learning by starting with easy samples.
    
    Difficulty scores should be precomputed (e.g., loss on pretrained model).
    """
    
    def __init__(self, difficulty_scores, current_epoch, total_epochs):
        self.scores = difficulty_scores
        self.epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Sort by difficulty
        self.sorted_indices = torch.argsort(torch.tensor(self.scores))
    
    def __iter__(self):
        # Fraction of data to use this epoch (linear schedule)
        fraction = min(1.0, (self.epoch + 1) / (self.total_epochs * 0.5))
        
        n_samples = int(len(self.sorted_indices) * fraction)
        active_indices = self.sorted_indices[:n_samples].tolist()
        
        # Shuffle active indices
        random.shuffle(active_indices)
        
        for idx in active_indices:
            yield idx
    
    def __len__(self):
        fraction = min(1.0, (self.epoch + 1) / (self.total_epochs * 0.5))
        return int(len(self.sorted_indices) * fraction)
```

## Batch Samplers

For batch-level control, use `BatchSampler`:

```python
from torch.utils.data import BatchSampler

# Create a batch sampler from any base sampler
batch_sampler = BatchSampler(
    sampler=RandomSampler(dataset),
    batch_size=32,
    drop_last=False
)

# BatchSampler yields lists of indices
for batch_indices in batch_sampler:
    print(batch_indices)  # [idx1, idx2, ..., idx32]
```

When using `batch_sampler`, pass it directly to DataLoader:

```python
loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler  # Cannot use batch_size, shuffle, etc.
)
```

## Sampler Selection Guide

| Scenario | Recommended Sampler |
|----------|-------------------|
| Standard training | RandomSampler (via shuffle=True) |
| Class imbalance | WeightedRandomSampler |
| Guaranteed balance | Custom BalancedBatchSampler |
| Train/val split | SubsetRandomSampler |
| Distributed training | DistributedSampler |
| Curriculum learning | Custom CurriculumSampler |
| Sequential evaluation | SequentialSampler |

## Comparison: Sampling Strategies

```python
# Setup: Imbalanced dataset (90% class 0, 10% class 1)
dataset = ImbalancedDataset(num_samples=1000, minority_ratio=0.1)

def measure_distribution(loader):
    """Measure class distribution over one epoch."""
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    
    counter = Counter(all_labels)
    total = sum(counter.values())
    return {k: v/total for k, v in counter.items()}

# Strategy 1: Random sampling
loader1 = DataLoader(dataset, batch_size=32, shuffle=True)
dist1 = measure_distribution(loader1)
print(f"Random:     Class 0: {dist1[0]:.1%}, Class 1: {dist1[1]:.1%}")
# Output: Class 0: 90.0%, Class 1: 10.0%

# Strategy 2: Weighted sampling
class_weights = 1.0 / torch.bincount(dataset.targets)
sample_weights = class_weights[dataset.targets]
sampler2 = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
loader2 = DataLoader(dataset, batch_size=32, sampler=sampler2)
dist2 = measure_distribution(loader2)
print(f"Weighted:   Class 0: {dist2[0]:.1%}, Class 1: {dist2[1]:.1%}")
# Output: Class 0: ~50%, Class 1: ~50%

# Strategy 3: Balanced batch
sampler3 = BalancedBatchSampler(dataset.targets, batch_size=32)
loader3 = DataLoader(dataset, batch_size=32, sampler=sampler3)
dist3 = measure_distribution(loader3)
print(f"Balanced:   Class 0: {dist3[0]:.1%}, Class 1: {dist3[1]:.1%}")
# Output: Class 0: 50.0%, Class 1: 50.0% (exactly)
```

## Mathematical Foundation

### Importance Sampling Connection

Weighted sampling implements importance sampling for the gradient:

$$
\nabla_\theta \mathcal{L} = \mathbb{E}_{x \sim p(x)}[\nabla_\theta \ell(x)]
$$

With weights $w(x)$, we sample from proposal $q(x)$ and reweight:

$$
\nabla_\theta \mathcal{L} = \mathbb{E}_{x \sim q(x)}\left[\frac{p(x)}{q(x)} \nabla_\theta \ell(x)\right]
$$

For class-balanced sampling with inverse frequency weights:

$$
w_c = \frac{N}{N_c} \cdot \frac{1}{C}
$$

where $N$ is total samples, $N_c$ is class $c$ samples, and $C$ is number of classes.

## Summary

| Sampler | Use Case | Replacement | Custom Weights |
|---------|----------|-------------|----------------|
| SequentialSampler | Evaluation | No | No |
| RandomSampler | Standard training | Optional | No |
| SubsetRandomSampler | Data splits | No | No |
| WeightedRandomSampler | Class imbalance | Yes | Yes |
| Custom Sampler | Specialized strategies | Custom | Custom |

## Practice Exercises

1. **Imbalance Handling**: Create a dataset with 95%/5% class split. Compare model performance with RandomSampler vs WeightedRandomSampler.

2. **Custom Sampler**: Implement a "hard example mining" sampler that prioritizes samples with high loss from the previous epoch.

3. **Stratified Validation**: Implement a sampler that maintains class proportions identical to the original dataset (useful for consistent validation).

## What's Next

The next section covers **Custom Collate Functions**, showing how to handle variable-length sequences, multi-modal data, and batch-level transformations.
