# Cross-Validation

## Learning Objectives

By the end of this section, you will be able to:

- Understand why cross-validation provides more reliable model evaluation
- Implement k-fold, stratified, and specialized cross-validation schemes
- Use cross-validation for model selection and hyperparameter tuning
- Recognize when different cross-validation strategies are appropriate
- Implement cross-validation efficiently in PyTorch

## Prerequisites

- Understanding of train/validation/test splits
- Familiarity with overfitting and bias-variance concepts
- Basic PyTorch training loops
- Knowledge of model evaluation metrics

---

## 1. Why Cross-Validation?

### 1.1 The Problem with Single Splits

A single train/validation split has limitations:

- **High variance:** Performance estimate depends on which samples end up in validation
- **Wasted data:** Validation samples never contribute to training
- **Unreliable with small datasets:** Random split may not be representative

### 1.2 The Cross-Validation Solution

Cross-validation uses **multiple splits** and averages results:

1. Divide data into k roughly equal parts ("folds")
2. Train k models, each time using a different fold as validation
3. Average the k performance estimates

```
5-Fold Cross-Validation:

Fold 1: [VAL][TRAIN][TRAIN][TRAIN][TRAIN]  → Score₁
Fold 2: [TRAIN][VAL][TRAIN][TRAIN][TRAIN]  → Score₂
Fold 3: [TRAIN][TRAIN][VAL][TRAIN][TRAIN]  → Score₃
Fold 4: [TRAIN][TRAIN][TRAIN][VAL][TRAIN]  → Score₄
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][VAL]  → Score₅

Final Score = mean(Score₁, ..., Score₅) ± std
```

### 1.3 Benefits

- **Lower variance:** Averaging over k estimates reduces variability
- **Better data utilization:** Every sample is used for both training and validation
- **More reliable comparison:** Enables statistical comparison between models

---

## 2. K-Fold Cross-Validation

### 2.1 Standard K-Fold

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

class KFoldCV:
    """
    K-Fold Cross-Validation implementation for PyTorch.
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X):
        """
        Generate train/validation indices for each fold.
        
        Yields:
            train_indices, val_indices for each fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            val_indices = indices[current:current + fold_size]
            train_indices = np.concatenate([
                indices[:current], 
                indices[current + fold_size:]
            ])
            yield train_indices, val_indices
            current += fold_size

def cross_validate(model_fn, X, y, n_splits=5, epochs=100, lr=0.01):
    """
    Perform k-fold cross-validation.
    
    Args:
        model_fn: Function that returns a fresh model instance
        X: Feature tensor
        y: Target tensor
        n_splits: Number of folds
        epochs: Training epochs per fold
        lr: Learning rate
        
    Returns:
        scores: List of validation scores (one per fold)
        mean_score: Average score
        std_score: Standard deviation of scores
    """
    kfold = KFoldCV(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    print(f"\n{'='*60}")
    print(f"Running {n_splits}-Fold Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Get fold data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create fresh model
        model = model_fn()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred.squeeze(), y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred.squeeze(), y_val).item()
        
        scores.append(val_loss)
        print(f"Fold {fold+1}/{n_splits}: Validation Loss = {val_loss:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"{'='*60}")
    print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"{'='*60}")
    
    return scores, mean_score, std_score
```

### 2.2 Choosing K

| K Value | Pros | Cons | When to Use |
|---------|------|------|-------------|
| K=5 | Good balance, fast | Moderate variance | Default choice |
| K=10 | Lower bias | Slower, higher variance | Medium datasets |
| K=N (LOOCV) | Lowest bias | Very slow, high variance | Very small datasets |

**Rule of thumb:** K=5 or K=10 works well for most cases.

### 2.3 Leave-One-Out Cross-Validation (LOOCV)

Special case where K = N (number of samples):

```python
def loocv(model_fn, X, y, epochs=100, lr=0.01):
    """
    Leave-One-Out Cross-Validation.
    
    Each sample serves as validation set once.
    Computationally expensive but lowest bias.
    """
    n = len(X)
    scores = []
    
    for i in range(n):
        # Leave sample i out
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[i] = False
        
        X_train, X_val = X[train_mask], X[i:i+1]
        y_train, y_val = y[train_mask], y[i:i+1]
        
        # Train and evaluate
        model = model_fn()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_train).squeeze(), y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val).squeeze(), y_val).item()
        scores.append(val_loss)
    
    return scores, np.mean(scores), np.std(scores)
```

---

## 3. Stratified Cross-Validation

### 3.1 Why Stratify?

For classification with imbalanced classes, random folds may have different class distributions. Stratification ensures each fold has similar class proportions.

```python
from sklearn.model_selection import StratifiedKFold

class StratifiedKFoldCV:
    """
    Stratified K-Fold that preserves class distribution in each fold.
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, 
                                    random_state=random_state)
    
    def split(self, X, y):
        """
        Generate stratified train/validation indices.
        
        Args:
            X: Features
            y: Class labels (must be provided for stratification)
        """
        # Convert to numpy for sklearn
        y_np = y.numpy() if torch.is_tensor(y) else y
        X_np = np.arange(len(y_np))  # Just need indices
        
        for train_idx, val_idx in self.skf.split(X_np, y_np):
            yield train_idx, val_idx

def stratified_cross_validate(model_fn, X, y, n_splits=5, epochs=100, lr=0.01):
    """
    Stratified k-fold cross-validation for classification.
    """
    skf = StratifiedKFoldCV(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    print(f"\n{'='*60}")
    print(f"Running {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Verify stratification
        train_dist = torch.bincount(y_train) / len(y_train)
        val_dist = torch.bincount(y_val) / len(y_val)
        
        # Train model
        model = model_fn()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate accuracy
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            preds = logits.argmax(dim=1)
            accuracy = (preds == y_val).float().mean().item()
        
        scores.append(accuracy)
        print(f"Fold {fold+1}: Accuracy = {accuracy:.4f}")
    
    print(f"{'='*60}")
    print(f"CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"{'='*60}")
    
    return scores, np.mean(scores), np.std(scores)
```

---

## 4. Specialized Cross-Validation Schemes

### 4.1 Time Series Cross-Validation

For temporal data, we must respect time ordering:

```python
class TimeSeriesSplit:
    """
    Time Series Split with expanding or sliding window.
    
    Always train on past, validate on future.
    """
    def __init__(self, n_splits=5, max_train_size=None):
        self.n_splits = n_splits
        self.max_train_size = max_train_size  # For sliding window
    
    def split(self, X):
        """
        Generate temporal train/validation splits.
        
        Expanding window (max_train_size=None):
        Split 1: [TRAIN][VAL]
        Split 2: [TRAIN TRAIN][VAL]
        Split 3: [TRAIN TRAIN TRAIN][VAL]
        
        Sliding window (max_train_size=k):
        Split 1: [TRAIN×k][VAL]
        Split 2:    [TRAIN×k][VAL]
        Split 3:       [TRAIN×k][VAL]
        """
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            val_start = (i + 1) * test_size
            val_end = val_start + test_size
            
            if self.max_train_size is not None:
                train_start = max(0, val_start - self.max_train_size)
            else:
                train_start = 0
            
            train_indices = np.arange(train_start, val_start)
            val_indices = np.arange(val_start, min(val_end, n_samples))
            
            yield train_indices, val_indices

def time_series_cv(model_fn, X, y, n_splits=5, epochs=100):
    """
    Time series cross-validation respecting temporal ordering.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    print(f"\n{'='*60}")
    print(f"Running Time Series {n_splits}-Split Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold+1}: Train indices {train_idx[0]}-{train_idx[-1]}, "
              f"Val indices {val_idx[0]}-{val_idx[-1]}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and evaluate
        model = model_fn()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_train).squeeze(), y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val).squeeze(), y_val).item()
        
        scores.append(val_loss)
        print(f"  Validation Loss = {val_loss:.4f}")
    
    print(f"{'='*60}")
    print(f"CV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return scores, np.mean(scores), np.std(scores)
```

### 4.2 Group K-Fold

Ensures groups don't span across folds:

```python
from sklearn.model_selection import GroupKFold

def group_cross_validate(model_fn, X, y, groups, n_splits=5, epochs=100):
    """
    Group K-Fold cross-validation.
    
    All samples from a group stay together in either train or val.
    
    Example: Patient data where each patient has multiple samples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    
    # Convert to numpy for sklearn
    X_np = np.arange(len(X))
    y_np = y.numpy() if torch.is_tensor(y) else y
    groups_np = groups.numpy() if torch.is_tensor(groups) else groups
    
    print(f"\n{'='*60}")
    print(f"Running {n_splits}-Fold Group Cross-Validation")
    print(f"Total groups: {len(np.unique(groups_np))}")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_np, y_np, groups_np)):
        train_groups = set(groups_np[train_idx])
        val_groups = set(groups_np[val_idx])
        
        print(f"Fold {fold+1}: {len(train_groups)} train groups, {len(val_groups)} val groups")
        
        # Verify no group leakage
        assert len(train_groups & val_groups) == 0, "Group leakage detected!"
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and evaluate
        model = model_fn()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_train).squeeze(), y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val).squeeze(), y_val).item()
        
        scores.append(val_loss)
    
    return scores, np.mean(scores), np.std(scores)
```

### 4.3 Nested Cross-Validation

For simultaneous model selection and evaluation:

```python
def nested_cross_validation(model_configs, X, y, outer_splits=5, inner_splits=3):
    """
    Nested cross-validation for unbiased model selection AND evaluation.
    
    Outer loop: Evaluates generalization performance
    Inner loop: Selects best hyperparameters
    
    Args:
        model_configs: List of (name, model_fn) tuples to compare
    """
    outer_cv = KFoldCV(n_splits=outer_splits, shuffle=True, random_state=42)
    
    outer_scores = []
    best_configs = []
    
    print(f"\n{'='*60}")
    print(f"Running Nested Cross-Validation")
    print(f"Outer: {outer_splits} folds, Inner: {inner_splits} folds")
    print(f"{'='*60}")
    
    for outer_fold, (outer_train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\nOuter Fold {outer_fold + 1}/{outer_splits}")
        
        X_outer_train, X_test = X[outer_train_idx], X[test_idx]
        y_outer_train, y_test = y[outer_train_idx], y[test_idx]
        
        # Inner CV: Select best model configuration
        inner_cv = KFoldCV(n_splits=inner_splits, shuffle=True, random_state=42)
        
        best_inner_score = float('inf')
        best_config = None
        
        for config_name, model_fn in model_configs:
            inner_scores = []
            
            for inner_train_idx, val_idx in inner_cv.split(X_outer_train):
                X_train = X_outer_train[inner_train_idx]
                X_val = X_outer_train[val_idx]
                y_train = y_outer_train[inner_train_idx]
                y_val = y_outer_train[val_idx]
                
                # Train
                model = model_fn()
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters())
                
                for _ in range(50):  # Fewer epochs for inner loop
                    optimizer.zero_grad()
                    loss = criterion(model(X_train).squeeze(), y_train)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val).squeeze(), y_val).item()
                inner_scores.append(val_loss)
            
            mean_inner = np.mean(inner_scores)
            if mean_inner < best_inner_score:
                best_inner_score = mean_inner
                best_config = (config_name, model_fn)
        
        print(f"  Best inner config: {best_config[0]} (score: {best_inner_score:.4f})")
        best_configs.append(best_config[0])
        
        # Outer evaluation: Train best config on full outer train, test on held-out
        model = best_config[1]()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X_outer_train).squeeze(), y_outer_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test).squeeze(), y_test).item()
        
        outer_scores.append(test_loss)
        print(f"  Outer test loss: {test_loss:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Nested CV Results:")
    print(f"Test Score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    print(f"Best configs per fold: {best_configs}")
    print(f"{'='*60}")
    
    return outer_scores, best_configs
```

---

## 5. Cross-Validation for Model Selection

### 5.1 Comparing Models

```python
def compare_models_cv(models_dict, X, y, n_splits=5, epochs=100):
    """
    Compare multiple models using cross-validation.
    
    Args:
        models_dict: Dict of {name: model_fn}
        
    Returns:
        results: Dict of {name: (mean_score, std_score)}
        best_model: Name of best performing model
    """
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Model Comparison using {n_splits}-Fold Cross-Validation")
    print(f"{'='*70}")
    
    for name, model_fn in models_dict.items():
        scores, mean_score, std_score = cross_validate(
            model_fn, X, y, n_splits=n_splits, epochs=epochs
        )
        results[name] = (mean_score, std_score, scores)
    
    # Summary table
    print(f"\n{'Model':<30} {'Mean Score':<15} {'Std':<10}")
    print("-" * 55)
    for name, (mean, std, _) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"{name:<30} {mean:<15.4f} {std:<10.4f}")
    
    best_model = min(results.items(), key=lambda x: x[1][0])[0]
    print(f"\nBest Model: {best_model}")
    
    return results, best_model
```

### 5.2 Hyperparameter Tuning with CV

```python
def grid_search_cv(model_class, param_grid, X, y, n_splits=5, epochs=100):
    """
    Grid search with cross-validation for hyperparameter tuning.
    
    Args:
        model_class: Model class (not instance)
        param_grid: Dict of {param_name: [values]}
        
    Returns:
        best_params: Best parameter combination
        best_score: Best CV score
        all_results: All tested combinations with scores
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    all_results = []
    best_score = float('inf')
    best_params = None
    
    print(f"\n{'='*70}")
    print(f"Grid Search CV: Testing {len(combinations)} combinations")
    print(f"{'='*70}")
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        # Create model function with these params
        def model_fn(p=params):
            return model_class(**p)
        
        scores, mean_score, std_score = cross_validate(
            model_fn, X, y, n_splits=n_splits, epochs=epochs
        )
        
        all_results.append((params, mean_score, std_score))
        
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            print(f"  New best: {params} -> {mean_score:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score: {best_score:.4f}")
    print(f"{'='*70}")
    
    return best_params, best_score, all_results
```

---

## 6. Statistical Comparison of Models

### 6.1 Paired t-Test

```python
from scipy import stats

def compare_models_statistical(scores_a, scores_b, model_a_name, model_b_name, alpha=0.05):
    """
    Statistically compare two models using paired t-test on CV scores.
    
    Args:
        scores_a, scores_b: Lists of CV scores (must have same length)
        alpha: Significance level
        
    Returns:
        is_significant: Whether difference is statistically significant
    """
    assert len(scores_a) == len(scores_b), "Must have same number of folds"
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    diff_mean = np.mean(scores_a) - np.mean(scores_b)
    
    print(f"\n{'='*60}")
    print(f"Statistical Comparison: {model_a_name} vs {model_b_name}")
    print(f"{'='*60}")
    print(f"{model_a_name} Mean: {np.mean(scores_a):.4f} ± {np.std(scores_a):.4f}")
    print(f"{model_b_name} Mean: {np.mean(scores_b):.4f} ± {np.std(scores_b):.4f}")
    print(f"Mean Difference: {diff_mean:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        winner = model_a_name if diff_mean < 0 else model_b_name
        print(f"\n✓ Difference is significant (p < {alpha})")
        print(f"  {winner} is significantly better")
        return True
    else:
        print(f"\n✗ No significant difference (p >= {alpha})")
        return False
```

---

## 7. Computational Considerations

### 7.1 Efficient Implementation Tips

```python
class EfficientKFoldCV:
    """
    Memory-efficient k-fold CV for large datasets.
    """
    def __init__(self, dataset, n_splits=5, batch_size=32):
        self.dataset = dataset
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        np.random.shuffle(self.indices)
    
    def get_fold_loaders(self, fold_idx):
        """
        Get DataLoaders for a specific fold without copying data.
        
        Uses Subset to avoid memory duplication.
        """
        fold_size = len(self.indices) // self.n_splits
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size
        
        val_indices = self.indices[val_start:val_end]
        train_indices = np.concatenate([
            self.indices[:val_start],
            self.indices[val_end:]
        ])
        
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader
```

### 7.2 Parallel Cross-Validation

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_cross_validate(model_fn, X, y, n_splits=5, n_jobs=4):
    """
    Parallel k-fold cross-validation.
    
    Note: Be careful with GPU memory when parallelizing!
    """
    def train_fold(fold_data):
        fold_idx, train_idx, val_idx = fold_data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_fn()
        # ... training code ...
        return val_loss
    
    # Prepare fold data
    kfold = KFoldCV(n_splits=n_splits)
    fold_data = [(i, train_idx, val_idx) 
                 for i, (train_idx, val_idx) in enumerate(kfold.split(X))]
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        scores = list(executor.map(train_fold, fold_data))
    
    return scores, np.mean(scores), np.std(scores)
```

---

## 8. Summary

### When to Use Each CV Type

| Scenario | CV Type | Key Consideration |
|----------|---------|-------------------|
| Default regression | K-Fold | K=5 or 10 |
| Classification | Stratified K-Fold | Preserves class balance |
| Time series | Time Series Split | Respect temporal order |
| Grouped data | Group K-Fold | No group leakage |
| Very small data | LOOCV | Maximum data use |
| Model selection + evaluation | Nested CV | Unbiased estimates |

### Best Practices

1. **Always use CV** for model selection and hyperparameter tuning
2. **Stratify** for classification problems
3. **Respect temporal ordering** for time series
4. **Keep groups together** when samples are not independent
5. **Report mean ± std** from CV, not just mean
6. **Use nested CV** when both selecting and evaluating

### Quick Reference

```python
# Standard k-fold
scores = cross_validate(model_fn, X, y, n_splits=5)

# Stratified for classification
scores = stratified_cross_validate(model_fn, X, y, n_splits=5)

# Time series
scores = time_series_cv(model_fn, X, y, n_splits=5)

# Group k-fold
scores = group_cross_validate(model_fn, X, y, groups, n_splits=5)

# Model comparison
results, best = compare_models_cv(models_dict, X, y)

# Hyperparameter tuning
best_params = grid_search_cv(ModelClass, param_grid, X, y)
```

---

## Exercises

### Exercise 1: CV Variance Analysis
Implement an experiment that shows how CV variance decreases as K increases. Plot mean ± std for K = 2, 5, 10, 20, N.

### Exercise 2: Stratification Impact
Create an imbalanced classification dataset (95% class 0, 5% class 1) and compare regular K-Fold vs Stratified K-Fold. Show the class distribution in each fold.

### Exercise 3: Walk-Forward Validation
Implement a walk-forward validation scheme for stock price prediction with a fixed window size of 250 days and a prediction horizon of 5 days.

### Exercise 4: Nested CV Experiment
Compare nested CV vs regular CV for model selection. Show that nested CV provides unbiased performance estimates while regular CV is optimistically biased.

---

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 7.

2. Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. *IJCAI*, 14(2), 1137-1145.

3. Varma, S., & Simon, R. (2006). Bias in Error Estimation When Using Cross-Validation for Model Selection. *BMC Bioinformatics*, 7(1), 91.

4. Bergmeir, C., & Benítez, J. M. (2012). On the Use of Cross-Validation for Time Series Predictor Evaluation. *Information Sciences*, 191, 192-213.
