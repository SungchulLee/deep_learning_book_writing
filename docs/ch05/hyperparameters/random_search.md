# Random Search

## Overview

Random search samples hyperparameter configurations uniformly (or from specified distributions) from the search space. Despite its simplicity, it is provably more efficient than grid search for most practical problems.

## Why Random Search Works

The key insight (Bergstra & Bengio, 2012): for most problems, only a few hyperparameters significantly affect performance. Grid search wastes evaluations by systematically varying unimportant dimensions. Random search, by contrast, provides different values for every hyperparameter in every trial, effectively exploring the important dimensions more densely.

## Implementation

```python
import numpy as np

def random_search(n_trials=50):
    best_val_loss = float('inf')
    best_config = None

    for _ in range(n_trials):
        config = {
            'lr': 10 ** np.random.uniform(-5, -1),       # Log-uniform
            'batch_size': int(2 ** np.random.uniform(5, 9)),  # 32–512
            'weight_decay': 10 ** np.random.uniform(-6, -2),
            'dropout': np.random.uniform(0.0, 0.5),
            'hidden_dim': np.random.choice([128, 256, 512, 1024])
        }

        val_loss = train_and_evaluate(config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config

    return best_config
```

## Distribution Selection

Use distributions that match the scale of the hyperparameter:

- **Learning rate**: Log-uniform (e.g., $10^{-5}$ to $10^{-1}$). Equal probability of landing in each order of magnitude.
- **Weight decay**: Log-uniform.
- **Dropout**: Uniform on [0, 0.5].
- **Hidden dimensions**: Categorical or log-uniform integers.
- **Batch size**: Powers of 2.

## Key Takeaways

- Random search is more efficient than grid search for the same budget.
- Use log-uniform distributions for scale-sensitive hyperparameters.
- 50–100 random trials typically suffice for a good configuration.
