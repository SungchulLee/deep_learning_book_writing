# Weighted Sampling

## Overview

Class imbalance is ubiquitous in real-world datasets—fraud detection, medical diagnosis, and rare event prediction in finance all exhibit heavily skewed class distributions. Weighted sampling addresses this by adjusting the probability of selecting each sample.

## WeightedRandomSampler

PyTorch's `WeightedRandomSampler` assigns a weight to each sample and draws samples with probability proportional to their weight:

```python
from torch.utils.data import WeightedRandomSampler

# Compute per-sample weights from class frequencies
class_counts = [9000, 1000]  # 90% class 0, 10% class 1
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# class_weights = [0.000111, 0.001]

sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # Samples per epoch
    replacement=True                   # Must be True for oversampling
)

loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

With `replacement=True`, minority class samples are drawn multiple times per epoch. The resulting batches are approximately class-balanced regardless of the underlying distribution.

## Computing Weights

**Inverse frequency weighting**: The most common approach assigns weight inversely proportional to class frequency:

$$w_c = \frac{N}{C \cdot N_c}$$

where $N$ is the total number of samples, $C$ is the number of classes, and $N_c$ is the count of class $c$.

```python
from collections import Counter

counts = Counter(train_labels)
n_samples = len(train_labels)
n_classes = len(counts)

class_weights = {c: n_samples / (n_classes * count)
                 for c, count in counts.items()}
sample_weights = [class_weights[label] for label in train_labels]
```

**Effective number weighting**: For severely imbalanced datasets, the effective number of samples provides smoother weights:

$$w_c = \frac{1 - \beta}{1 - \beta^{N_c}}, \quad \beta \in [0, 1)$$

```python
beta = 0.999
effective_num = {c: (1 - beta**count) / (1 - beta)
                 for c, count in counts.items()}
class_weights = {c: 1.0 / effective_num[c] for c in counts}
```

## Combining with Loss Weighting

Weighted sampling can be combined with (or substituted by) weighted loss functions:

```python
# Class weights in the loss function
weights = torch.tensor([1.0, 9.0])  # Upweight minority class
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

The two approaches are complementary: weighted sampling affects which samples the model sees, while weighted loss affects how much each sample's error contributes to the gradient.

## Financial Application: Rare Event Sampling

In credit default or market crash prediction, positive events may constitute less than 1% of samples:

```python
# Extreme imbalance: 99.5% non-default, 0.5% default
labels = dataset.labels
default_weight = 1.0 / (labels == 1).sum().item()
non_default_weight = 1.0 / (labels == 0).sum().item()

weights = torch.where(
    torch.tensor(labels) == 1,
    torch.tensor(default_weight),
    torch.tensor(non_default_weight)
)

sampler = WeightedRandomSampler(weights, num_samples=len(labels),
                                replacement=True)
```

## Key Takeaways

- `WeightedRandomSampler` rebalances the effective class distribution seen during training.
- Inverse frequency weighting is the standard approach; effective number weighting handles extreme imbalance.
- Weighted sampling and weighted loss are complementary strategies.
- In finance, rare event prediction (defaults, crashes) requires aggressive rebalancing.
