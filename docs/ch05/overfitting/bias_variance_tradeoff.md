# Bias-Variance Tradeoff

## Overview

The bias-variance tradeoff formalizes the tension between model simplicity (bias) and model flexibility (variance). The expected test error decomposes as:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

## Definitions

For a model $\hat{f}$ trained on dataset $\mathcal{D}$ and evaluated at input $x$:

**Bias** measures systematic error—how far the average prediction (over all possible training sets) deviates from the true function:

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}_\mathcal{D}[\hat{f}(x)] - f(x)$$

**Variance** measures sensitivity to the particular training set:

$$\text{Var}[\hat{f}(x)] = \mathbb{E}_\mathcal{D}\left[(\hat{f}(x) - \mathbb{E}_\mathcal{D}[\hat{f}(x)])^2\right]$$

## The Tradeoff

Simple models (few parameters) have high bias and low variance—consistently wrong in the same way. Complex models (many parameters) have low bias and high variance—correct on average but sensitive to training data. The optimal model complexity minimizes the sum.

## Implications for Deep Learning

Deep networks have low bias (high expressiveness) but potentially high variance. Regularization techniques (dropout, weight decay, data augmentation, early stopping) reduce variance at the cost of slightly increased bias, improving overall error.

## Financial Context

Financial data has exceptionally high irreducible noise. This means even the optimal model has high error, and the variance component is particularly dangerous because it manifests as illusory patterns that disappear out of sample.

## Key Takeaways

- Test error = Bias² + Variance + Noise.
- Model complexity trades bias for variance.
- In finance, high irreducible noise makes variance control especially important.
