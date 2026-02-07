# Train-Val-Test Split

## Overview

Proper data splitting is fundamental to honest evaluation. The three-way split serves distinct purposes: training data for parameter optimization, validation data for hyperparameter selection, and test data for final performance estimation.

## Standard Split

```python
from torch.utils.data import random_split

dataset = MyDataset(...)
n = len(dataset)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```

Typical splits: 70/15/15, 80/10/10, or 60/20/20 depending on dataset size.

## Role of Each Split

**Training set**: Used to compute gradients and update model parameters. The model sees this data during backpropagation.

**Validation set**: Used to select hyperparameters (learning rate, model architecture, regularization strength) and for early stopping. The model never trains on this data, but decisions based on validation performance indirectly leak information.

**Test set**: Used exactly once for final performance reporting. Any use of the test set for model selection invalidates its estimate of generalization performance.

## Temporal Splits for Financial Data

For time series, random splitting causes data leakage. Use chronological splits:

```python
def temporal_split(data, train_frac=0.6, val_frac=0.2):
    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test
```

The training set always precedes the validation set, which precedes the test set chronologically.

## Common Mistakes

**Using test set for model selection**: If you evaluate multiple models on the test set and pick the best one, the test set has become a validation set.

**Random splits for time series**: Randomly shuffling time series data and then splitting allows future information to leak into training.

**Not accounting for grouped data**: If the same entity (patient, company, user) appears in both training and test sets, performance estimates are optimistically biased.

## Key Takeaways

- Three distinct data splits serve three distinct purposes.
- The test set must be used exactly once for final reporting.
- Time series data requires chronological splits to prevent look-ahead bias.
- Group-aware splitting prevents entity-level data leakage.
