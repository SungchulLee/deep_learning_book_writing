# Grid Search

## Overview

Grid search evaluates all combinations of hyperparameter values on a predefined grid. It is exhaustive and simple but scales exponentially with the number of hyperparameters.

## Implementation

```python
from itertools import product

param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'weight_decay': [0, 1e-4, 1e-3]
}

best_val_loss = float('inf')
best_config = None

for lr, bs, wd in product(param_grid['lr'], param_grid['batch_size'],
                           param_grid['weight_decay']):
    config = {'lr': lr, 'batch_size': bs, 'weight_decay': wd}
    val_loss = train_and_evaluate(config)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_config = config
```

## Limitations

Grid search evaluates $\prod_i |V_i|$ configurations, where $|V_i|$ is the number of values for hyperparameter $i$. With 5 hyperparameters and 5 values each, this requires $5^5 = 3125$ training runs.

More critically, grid search allocates equal resolution to all hyperparameters, even those that have little effect on performance. Random search addresses this inefficiency.

## When to Use

Grid search is appropriate for final fine-tuning over 1–2 hyperparameters where you need guaranteed coverage of the search space.

## Key Takeaways

- Grid search is exhaustive but exponentially expensive.
- It wastes budget on unimportant hyperparameters.
- Use only for low-dimensional search spaces or final refinement.
