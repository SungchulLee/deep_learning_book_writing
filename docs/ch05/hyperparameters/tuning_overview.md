# Tuning Overview

## Overview

Hyperparameters are configuration choices that are not learned during training: learning rate, batch size, number of layers, weight decay, dropout rate, and more. Systematic hyperparameter tuning can often improve performance more than architectural changes.

## Hyperparameter Categories

**Optimization hyperparameters**: Learning rate, batch size, optimizer choice, momentum, weight decay, learning rate schedule.

**Architecture hyperparameters**: Number of layers, hidden dimensions, activation functions, attention heads.

**Regularization hyperparameters**: Dropout rate, weight decay, data augmentation intensity, early stopping patience.

## The Search Problem

Hyperparameter optimization is an outer loop around the training process. For each configuration $\lambda$, we train a model and evaluate its validation performance $f(\lambda)$. The goal is to find:

$$\lambda^* = \arg\min_\lambda f(\lambda)$$

This is expensive: each evaluation requires a full training run. The search space is often mixed (continuous, discrete, conditional), high-dimensional, and the objective $f$ is noisy.

## Search Strategies

| Strategy | Computational Cost | Best For |
|---|---|---|
| Grid search | Exponential in dimensions | Low-dimensional, discrete spaces |
| Random search | User-defined budget | Most practical default |
| Bayesian optimization | Moderate | Expensive-to-evaluate models |
| Successive halving / Hyperband | Budget-efficient | Large search spaces |

## Practical Guidelines

1. **Start with established defaults**: Adam with LR 0.001, batch size 32–128, standard architectures.
2. **Tune learning rate first**: It has the largest impact on performance.
3. **Use random search over grid search**: Random search explores the space more efficiently.
4. **Log everything**: Record all configurations and their results for analysis.
5. **Use early stopping**: Don't waste compute on clearly poor configurations.

## Key Takeaways

- Hyperparameter tuning is an outer optimization loop around training.
- Learning rate is typically the most important hyperparameter.
- Random search and Bayesian optimization are more efficient than grid search.
