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

## Successive Halving and Hyperband

A major bottleneck in hyperparameter optimization is that each evaluation requires a full training run. **Successive halving** addresses this by allocating a fixed budget across many configurations, progressively eliminating the worst performers.

The algorithm works as follows: start with $n$ random configurations, each trained for a small budget $B/n$. Keep the top half and double their budget. Repeat until one configuration remains. This trades off **exploration** (trying many configurations) against **exploitation** (giving more budget to promising ones).

**Hyperband** improves upon successive halving by running it multiple times with different trade-off parameters, hedging against the possibility that some good configurations are slow starters.

!!! tip "Successive Halving for Quant Models"
    In quantitative finance, successive halving is particularly useful when training across multiple market regimes or asset classes. You can quickly eliminate configurations that fail on even a single regime before committing resources to full walk-forward evaluation.

## Search Space Design

Effective hyperparameter tuning requires thoughtful search space design:

- **Log-uniform scales** for learning rate (e.g., $10^{-5}$ to $10^{-1}$) and weight decay
- **Conditional hyperparameters**: Optimizer-specific parameters (momentum only for SGD, beta values only for Adam)
- **Categorical choices**: Activation functions, normalization layers, loss functions
- **Integer parameters**: Number of layers, hidden dimensions (often powers of 2 for GPU efficiency)

## Automated Machine Learning (AutoML)

Modern AutoML frameworks combine hyperparameter tuning with architecture search:

- **Optuna**: Efficient Bayesian optimization with pruning support
- **Ray Tune**: Scalable, distributed hyperparameter tuning
- **Weights & Biases Sweeps**: Integrated experiment tracking and tuning

These tools implement many of the strategies above and handle the engineering challenges of parallel evaluation, checkpointing, and fault tolerance.

## Key Takeaways

- Hyperparameter tuning is an outer optimization loop around training.
- Learning rate is typically the most important hyperparameter.
- Random search and Bayesian optimization are more efficient than grid search.
- Successive halving and Hyperband make tuning budget-efficient by eliminating poor configurations early.
- Thoughtful search space design (log scales, conditional parameters) is as important as the search algorithm.
