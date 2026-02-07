# Overfitting and Underfitting

## Overview

Overfitting and underfitting are the two failure modes of statistical learning. Understanding them is prerequisite to building models that generalize from training data to unseen examples.

## Definitions

**Underfitting** occurs when a model is too simple to capture the underlying patterns in the data. The model has high error on both training and test data. Signs: training loss plateaus at a high value.

**Overfitting** occurs when a model memorizes training data—including its noise—rather than learning generalizable patterns. The model has low training error but high test error. Signs: growing gap between training and validation loss.

## The Complexity Spectrum

As model complexity increases (more parameters, more layers, wider networks):

1. **Low complexity**: High training error, high test error (underfitting).
2. **Optimal complexity**: Moderate training error, lowest test error.
3. **High complexity**: Low training error, increasing test error (overfitting).

This relationship is formalized by the bias-variance tradeoff (next sections).

## Deep Learning and Double Descent

Classical learning theory predicts monotonically increasing test error past the optimal complexity. However, modern deep networks exhibit **double descent**: after the overfitting peak at the interpolation threshold, test error decreases again as models become sufficiently over-parameterized to find smooth interpolating solutions.

## Remedies for Overfitting

- More data: the most reliable remedy.
- Data augmentation: artificially expand the effective dataset size.
- Regularization: weight decay, dropout, batch normalization.
- Early stopping: stop training when validation performance degrades.
- Reduce model capacity: fewer layers, fewer parameters.
- Ensemble methods: average predictions from multiple models.

## Remedies for Underfitting

- Increase model capacity: more layers, wider layers.
- Train longer: more epochs with appropriate learning rate.
- Reduce regularization: remove or reduce weight decay, dropout.
- Better features: feature engineering or more expressive input representations.

## Quantitative Finance Context

Overfitting is the dominant failure mode in financial ML due to low signal-to-noise ratios, non-stationary data, and limited effective sample sizes. A model that backtests profitably on historical data but fails in live trading is the quintessential overfitting example. Walk-forward validation is essential.

## Key Takeaways

- Underfitting: model too simple. Overfitting: model too complex.
- Validation loss relative to training loss diagnoses which regime you're in.
- Deep learning exhibits double descent, complicating the classical complexity narrative.
- In quantitative finance, overfitting is the primary concern and requires domain-specific validation strategies.
