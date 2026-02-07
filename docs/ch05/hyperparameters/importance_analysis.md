# Hyperparameter Importance Analysis

## Overview

Not all hyperparameters matter equally. Importance analysis identifies which hyperparameters have the greatest impact on model performance, guiding where to focus tuning effort.

## fANOVA (Functional ANOVA)

Functional ANOVA decomposes the variance of the objective function across hyperparameter dimensions:

$$f(\lambda) = f_0 + \sum_i f_i(\lambda_i) + \sum_{i<j} f_{ij}(\lambda_i, \lambda_j) + \cdots$$

The variance explained by each term quantifies the importance of individual hyperparameters and their interactions.

## Optuna Importance Analysis

```python
import optuna
from optuna.importance import get_param_importances

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

importances = get_param_importances(study)
for param, importance in importances.items():
    print(f"{param}: {importance:.4f}")
```

## Typical Importance Rankings

Based on extensive empirical studies, hyperparameters typically rank in importance as follows:

1. **Learning rate**: Almost always the most important.
2. **Batch size**: Significant effect on both convergence and generalization.
3. **Number of layers / hidden dimension**: Architecture choices have moderate impact.
4. **Weight decay**: Important for regularization, especially with AdamW.
5. **Dropout rate**: Moderate impact, highly task-dependent.
6. **Momentum / beta parameters**: Usually safe at defaults.

## Sensitivity Analysis

For critical applications, perform sensitivity analysis around the selected configuration:

```python
# Perturb each hyperparameter ±20% and measure validation loss change
base_config = best_config.copy()
for param in base_config:
    for factor in [0.8, 1.0, 1.2]:
        config = base_config.copy()
        config[param] = base_config[param] * factor
        val_loss = train_and_evaluate(config)
        print(f"{param}={config[param]:.4f}: val_loss={val_loss:.4f}")
```

## Key Takeaways

- Learning rate dominates hyperparameter importance for most models.
- Focus tuning budget on the most impactful hyperparameters.
- Use fANOVA or Optuna's importance analysis to identify which hyperparameters to prioritize.
- Sensitivity analysis around the final configuration validates robustness.
