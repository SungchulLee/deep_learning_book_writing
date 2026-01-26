# Train-Validation-Test Split

## Learning Objectives

By the end of this section, you will be able to:

- Understand why proper data splitting is essential for model evaluation
- Design appropriate splitting strategies for different data types
- Implement stratified, temporal, and grouped splits in PyTorch
- Avoid common pitfalls like data leakage
- Choose appropriate split ratios for different scenarios

## Prerequisites

- Basic understanding of supervised learning
- Familiarity with overfitting concepts
- PyTorch tensor operations

---

## 1. Why Split Data?

### 1.1 The Fundamental Problem

We want to know how well our model will perform on **new, unseen data**. However, we only have access to our current dataset. The solution is to hold out part of the data and pretend we've never seen it.

### 1.2 The Three-Way Split

| Split | Purpose | Used For |
|-------|---------|----------|
| **Training Set** | Learn model parameters | Fitting weights, backpropagation |
| **Validation Set** | Tune hyperparameters | Model selection, early stopping |
| **Test Set** | Final evaluation | Unbiased performance estimate |

```
┌─────────────────────────────────────────────────────────┐
│                    Full Dataset                          │
├───────────────────────┬─────────────┬───────────────────┤
│    Training (60%)     │  Val (20%)  │    Test (20%)     │
│   Learn parameters    │ Tune hyper- │  Final estimate   │
│                       │ parameters  │  (touch once!)    │
└───────────────────────┴─────────────┴───────────────────┘
```

### 1.3 Why Three Sets Instead of Two?

**Without validation set:** You tune hyperparameters on the test set, which means the test set influences model selection. The test error becomes optimistically biased.

**With validation set:** Hyperparameter tuning uses validation data, keeping the test set truly unseen until final evaluation.

!!! warning "Golden Rule"
    The test set should be used **only once** - for final model evaluation. Any repeated use contaminates it.

---

## 2. Splitting Strategies

### 2.1 Random Split (Default)

Randomly assign samples to each set. Appropriate when data is i.i.d.

```python
import torch
from torch.utils.data import random_split, Subset
import numpy as np

def random_train_val_test_split(dataset, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Randomly split dataset into train/val/test sets.
    
    Args:
        dataset: PyTorch Dataset object
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_set, val_set, test_set
    """
    torch.manual_seed(seed)
    
    n = len(dataset)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Split sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    return train_set, val_set, test_set
```

### 2.2 Stratified Split

Preserves class distribution across splits. Essential for imbalanced classification.

```python
from sklearn.model_selection import train_test_split

def stratified_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Stratified split preserving class distribution.
    
    Args:
        X: Features (numpy array or tensor)
        y: Labels (numpy array or tensor)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Convert to numpy if needed
    X_np = X.numpy() if torch.is_tensor(X) else X
    y_np = y.numpy() if torch.is_tensor(y) else y
    
    # First split: separate test set
    test_ratio = 1 - train_ratio - val_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_np, y_np, test_size=test_ratio, stratify=y_np, random_state=seed
    )
    
    # Second split: separate train and val
    val_adjusted_ratio = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted_ratio, stratify=y_temp, random_state=seed
    )
    
    # Verify stratification
    print("Class distribution:")
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = counts / len(labels)
        print(f"  {name}: {dict(zip(unique, dist.round(3)))}")
    
    return (torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test),
            torch.LongTensor(y_train), torch.LongTensor(y_val), torch.LongTensor(y_test))
```

### 2.3 Temporal Split (Time Series)

For time-ordered data, always train on past and test on future.

```python
def temporal_split(X, y, timestamps, train_ratio=0.6, val_ratio=0.2):
    """
    Temporal split respecting time ordering.
    
    CRITICAL: Never use future data to predict past!
    
    Args:
        X: Features
        y: Targets
        timestamps: Time indices or datetime values
        
    Returns:
        Split datasets with temporal ordering preserved
    """
    # Sort by time
    sort_idx = np.argsort(timestamps)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    n = len(X_sorted)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    
    X_train = X_sorted[:train_end]
    y_train = y_sorted[:train_end]
    
    X_val = X_sorted[train_end:val_end]
    y_val = y_sorted[train_end:val_end]
    
    X_test = X_sorted[val_end:]
    y_test = y_sorted[val_end:]
    
    print(f"Temporal split:")
    print(f"  Train: indices 0 to {train_end-1}")
    print(f"  Val: indices {train_end} to {val_end-1}")
    print(f"  Test: indices {val_end} to {n-1}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

### 2.4 Group Split

When samples belong to groups (e.g., patients, users), ensure no group spans multiple splits.

```python
from sklearn.model_selection import GroupShuffleSplit

def group_split(X, y, groups, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Split data ensuring groups don't span across splits.
    
    Example: Medical data where each patient has multiple samples.
    All samples from one patient must be in the same split.
    
    Args:
        groups: Group membership for each sample
    """
    # First split: train+val vs test
    test_ratio = 1 - train_ratio - val_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    temp_idx, test_idx = next(gss1.split(X, y, groups))
    
    # Second split: train vs val
    val_adjusted = val_ratio / (train_ratio + val_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_adjusted, random_state=seed)
    
    X_temp, y_temp, groups_temp = X[temp_idx], y[temp_idx], groups[temp_idx]
    train_idx_temp, val_idx_temp = next(gss2.split(X_temp, y_temp, groups_temp))
    
    train_idx = temp_idx[train_idx_temp]
    val_idx = temp_idx[val_idx_temp]
    
    # Verify no group leakage
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])
    
    assert len(train_groups & val_groups) == 0, "Group leakage: train-val"
    assert len(train_groups & test_groups) == 0, "Group leakage: train-test"
    assert len(val_groups & test_groups) == 0, "Group leakage: val-test"
    
    print(f"Groups: Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")
    
    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]
```

---

## 3. Choosing Split Ratios

### 3.1 Common Ratios

| Dataset Size | Train | Val | Test | Notes |
|--------------|-------|-----|------|-------|
| Small (<1K) | 60% | 20% | 20% | Use cross-validation instead |
| Medium (1K-100K) | 70% | 15% | 15% | Standard ratio |
| Large (100K-1M) | 80% | 10% | 10% | More training data helps |
| Very Large (>1M) | 98% | 1% | 1% | 10K samples suffice for estimation |

### 3.2 Factors to Consider

**Model Complexity:**
- Complex models need more training data
- Simple models can work with smaller training sets

**Number of Hyperparameters:**
- More hyperparameters → larger validation set
- Few hyperparameters → smaller validation set

**Desired Precision:**
- More test samples → more precise performance estimates
- Rule of thumb: 1000+ test samples for ~3% confidence interval

---

## 4. Data Leakage: The Silent Killer

### 4.1 What is Data Leakage?

Data leakage occurs when information from outside the training set influences the model. This makes validation/test performance unrealistically optimistic.

### 4.2 Common Sources of Leakage

**Feature Leakage:**
```python
# WRONG: Normalizing before splitting
X_normalized = (X - X.mean()) / X.std()  # Uses test data stats!
X_train, X_test = split(X_normalized)

# CORRECT: Normalize using only training statistics
X_train, X_test = split(X)
train_mean, train_std = X_train.mean(), X_train.std()
X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std  # Use TRAIN stats
```

**Temporal Leakage:**
```python
# WRONG: Random split on time series
indices = np.random.permutation(n)  # Future data in training!

# CORRECT: Temporal split
train_idx = np.where(timestamps < cutoff_date)[0]
test_idx = np.where(timestamps >= cutoff_date)[0]
```

**Target Leakage:**
```python
# WRONG: Feature that encodes the target
df['will_default'] = df['defaulted']  # Obviously leaks!
df['days_until_default'] = ...  # Subtly leaks!

# CORRECT: Only use features available at prediction time
```

### 4.3 Prevention Strategies

1. **Split first, preprocess after**
2. **Create preprocessing pipelines** that fit only on training data
3. **Be suspicious of perfect results** - often indicates leakage
4. **Audit features** for temporal or target information

---

## 5. PyTorch Dataset Integration

### 5.1 Custom Dataset with Splitting

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TensorDataset(Dataset):
    """Simple dataset from tensors."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                       batch_size=32, num_workers=0):
    """
    Create DataLoaders for train/val/test sets.
    
    Note: Only training loader shuffles data.
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### 5.2 Complete Pipeline Example

```python
class DataPipeline:
    """
    Complete data pipeline with proper splitting and preprocessing.
    """
    def __init__(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.scaler_mean = None
        self.scaler_std = None
    
    def fit_transform(self, X, y, split_type='random', **kwargs):
        """
        Split data and fit preprocessing on training set only.
        """
        # Split data
        if split_type == 'random':
            splits = self._random_split(X, y)
        elif split_type == 'stratified':
            splits = self._stratified_split(X, y)
        elif split_type == 'temporal':
            splits = self._temporal_split(X, y, kwargs.get('timestamps'))
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        # Fit scaler on training data only
        self.scaler_mean = X_train.mean(dim=0)
        self.scaler_std = X_train.std(dim=0) + 1e-8
        
        # Transform all sets using training statistics
        X_train = (X_train - self.scaler_mean) / self.scaler_std
        X_val = (X_val - self.scaler_mean) / self.scaler_std
        X_test = (X_test - self.scaler_mean) / self.scaler_std
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def transform(self, X):
        """Transform new data using fitted scaler."""
        if self.scaler_mean is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        return (X - self.scaler_mean) / self.scaler_std
    
    def _random_split(self, X, y):
        n = len(X)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        
        train_end = int(self.train_ratio * n)
        val_end = int((self.train_ratio + self.val_ratio) * n)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return (X[train_idx], X[val_idx], X[test_idx],
                y[train_idx], y[val_idx], y[test_idx])
    
    def _stratified_split(self, X, y):
        # Implementation as shown in Section 2.2
        pass
    
    def _temporal_split(self, X, y, timestamps):
        # Implementation as shown in Section 2.3
        pass
```

---

## 6. Summary

### Key Principles

1. **Three-way split** separates learning, tuning, and evaluation
2. **Choose strategy** based on data structure (i.i.d., temporal, grouped)
3. **Prevent leakage** by splitting before preprocessing
4. **Respect the test set** - use it only once for final evaluation

### Decision Flowchart

```
Is data time-ordered?
├── Yes → Temporal Split
└── No → Are there natural groups?
         ├── Yes → Group Split
         └── No → Is it classification with imbalanced classes?
                  ├── Yes → Stratified Split
                  └── No → Random Split
```

### Quick Reference

| Situation | Split Type | Key Concern |
|-----------|------------|-------------|
| i.i.d. regression | Random | Sufficient test size |
| Imbalanced classification | Stratified | Class distribution |
| Time series | Temporal | No future leakage |
| Multi-patient medical | Group | Patient isolation |

---

## Exercises

### Exercise 1: Leakage Detection
Given a dataset with 95% test accuracy, design a procedure to test if data leakage might be present.

### Exercise 2: Financial Time Series
Implement a walk-forward validation scheme for stock price prediction that respects temporal ordering.

### Exercise 3: Multi-Site Clinical Trial
Design a splitting strategy for clinical trial data collected from 10 different hospitals, ensuring fair evaluation.

---

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 7.

2. Kaufman, S., et al. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4).

3. Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437-450.
