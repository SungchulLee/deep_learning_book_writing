# Bayesian Optimization

## Overview

Bayesian optimization builds a probabilistic surrogate model of the objective function $f(\lambda)$ and uses it to intelligently select the next configuration to evaluate. It is the method of choice when each evaluation is expensive.

## How It Works

1. **Surrogate model**: A Gaussian Process (or tree-based model) is fitted to the observed (configuration, validation loss) pairs. The GP provides both a mean prediction and uncertainty estimate for any unobserved configuration.

2. **Acquisition function**: Determines where to sample next by trading off exploitation (sampling where the surrogate predicts low loss) and exploration (sampling where uncertainty is high). Common choices include Expected Improvement (EI) and Upper Confidence Bound (UCB).

3. **Iterate**: Evaluate the selected configuration, add it to the observed set, update the surrogate, and repeat.

## Using Optuna

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    n_layers = trial.suggest_int('n_layers', 2, 6)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512, step=64)

    config = {
        'lr': lr, 'batch_size': batch_size,
        'weight_decay': weight_decay,
        'n_layers': n_layers, 'hidden_dim': hidden_dim
    }
    val_loss = train_and_evaluate(config)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best config: {study.best_params}")
print(f"Best val loss: {study.best_value}")
```

## Advantages over Random Search

Bayesian optimization typically finds better configurations in fewer trials because it uses information from previous evaluations to guide the search. The improvement is most significant when evaluations are expensive (long training times) and the search space has exploitable structure.

## Pruning (Early Stopping for Bad Trials)

Optuna supports pruning underperforming trials:

```python
def objective(trial):
    # ... setup ...
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        val_loss = validate(...)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_loss

study.optimize(objective, n_trials=100)
```

## Key Takeaways

- Bayesian optimization uses a surrogate model to guide the search intelligently.
- More sample-efficient than random search, especially for expensive evaluations.
- Optuna provides a practical, easy-to-use framework with pruning support.
