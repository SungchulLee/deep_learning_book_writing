# Chapter 3: Statistical Learning

## Overview

This chapter builds the statistical and probabilistic foundations that underpin virtually all of deep learning. Starting from Maximum Likelihood Estimation, we develop the core regression and classification models that serve as both practical tools and conceptual building blocks for the neural network architectures explored in later chapters.

!!! note "Why Statistical Learning Matters for Deep Learning"
    Every loss function, every training objective, and every regularization technique in deep learning has roots in classical statistical learning theory. Understanding these foundations transforms deep learning from a collection of recipes into a principled engineering discipline.

## Chapter Structure

### 3.1 Maximum Likelihood Estimation

MLE provides the unifying probabilistic framework for understanding model training. We derive the likelihood principle, develop the log-likelihood machinery, and show how MLE applies to both regression and classification settings. The key insight is that minimizing common loss functions (MSE, cross-entropy) is equivalent to maximizing likelihood under specific distributional assumptions.

**Sections**: [MLE Foundations](mle/mle.md) · [MLE for Regression](mle/mle_regression.md) · [MLE for Classification](mle/mle_classification.md) · [Probabilistic Interpretation](mle/probabilistic_interpretation.md)

### 3.2 Linear Regression

Linear regression is the simplest supervised learning model, yet it introduces nearly every concept needed for deep learning: loss functions, gradient descent, closed-form solutions, overfitting, and regularization. We cover the normal equation, gradient-based optimization in both NumPy and PyTorch, polynomial feature expansion, and $L_1$/$L_2$ regularization (Ridge and Lasso).

**Sections**: [Linear Regression](linear_regression/linear_regression.md) · [Closed-Form Solution](linear_regression/closed_form.md) · [Gradient Descent Solution](linear_regression/gd_solution.md) · [Polynomial Features](linear_regression/polynomial_features.md) · [Ridge Regression](linear_regression/ridge_regression.md) · [Lasso Regression](linear_regression/lasso_regression.md)

### 3.3 Logistic Regression

Logistic regression extends linear models to binary classification by introducing the sigmoid function and the cross-entropy loss. We derive the gradient computation in detail and examine decision boundaries, regularization, and the geometric interpretation of the learned classifier.

**Sections**: [Binary Classification](logistic_regression/binary_classification.md) · [Sigmoid Function](logistic_regression/sigmoid.md) · [Decision Boundary](logistic_regression/decision_boundary.md) · [Gradient Computation](logistic_regression/gradient.md) · [Regularized Logistic Regression](logistic_regression/regularized.md)

### 3.4 Softmax Regression

Softmax regression generalizes logistic regression to multiclass problems. We develop the softmax function, derive the cross-entropy loss for the multiclass setting, and address the numerical stability challenges that arise in practical implementations — issues that directly carry over to training deep neural networks.

**Sections**: [Multiclass Classification](softmax_regression/multiclass.md) · [Softmax Function](softmax_regression/softmax_function.md) · [Cross-Entropy Loss](softmax_regression/cross_entropy.md) · [Numerical Stability](softmax_regression/numerical_stability.md)

### 3.5 Loss Functions

A comprehensive treatment of loss functions used across deep learning. Beyond the standard MSE and cross-entropy, we cover focal loss for class imbalance, hinge loss for SVMs, Huber loss for robust regression, and KL divergence — including its properties, closed-form expressions for Gaussians, and its connection to Fisher information. We conclude with guidelines for custom loss design and loss function selection.

**Sections**: [Overview](loss/loss_overview.md) · [MSE and MAE](loss/mse_mae.md) · [Cross-Entropy](loss/cross_entropy.md) · [Binary Cross-Entropy](loss/bce.md) · Focal Loss · Hinge Loss · Huber Loss · [KL Divergence](loss/kl_divergence.md) · [KL Distance Axioms](loss/kl_distance_axioms.md) · [KL for Gaussians](loss/kl_gaussian.md) · [KL and Fisher Information](loss/kl_fisher_information.md) · Custom Loss Functions · Loss Function Selection

## Prerequisites

This chapter assumes familiarity with:

- **Linear algebra**: Matrix operations, vector spaces, projections (Chapter 1)
- **Calculus**: Partial derivatives, gradients, chain rule (Chapter 1)
- **Probability**: Probability distributions, conditional probability, Bayes' theorem (Chapter 2)
- **Python/PyTorch basics**: Tensor operations, autograd fundamentals (Chapter 2)

## Connections to Later Chapters

| Concept | Where It Leads |
|---|---|
| Gradient descent | Optimizers: SGD, Adam, learning rate schedules (Ch 4) |
| MLE / cross-entropy | Training objectives for all neural networks (Ch 5+) |
| Regularization ($L_1$, $L_2$) | Dropout, weight decay, batch normalization (Ch 4) |
| Softmax + numerical stability | Output layers in classification networks (Ch 5) |
| KL divergence | Variational autoencoders, knowledge distillation (Ch 10+) |
| Loss function design | Task-specific objectives in NLP, vision, finance (Ch 7+) |
