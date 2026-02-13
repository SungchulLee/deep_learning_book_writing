# Chapter 5: Training Pipeline

## Overview

Training a deep learning model is far more than writing a forward pass—it requires orchestrating data loading, optimization, regularization, evaluation, and persistence into a coherent pipeline. This chapter provides a systematic treatment of every component in the PyTorch training pipeline, from raw data ingestion through model deployment.

The canonical PyTorch training loop is deceptively simple:

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
```

Behind each line lies a rich set of design decisions—how data is batched and shuffled, which optimizer to use and how to schedule its learning rate, when to stop training, and how to evaluate the resulting model. This chapter unpacks each decision systematically.

## Chapter Structure

**Section 5.1 — Datasets** covers PyTorch's `Dataset` abstraction, built-in datasets for rapid prototyping, custom dataset implementations for real-world data, and the transform pipeline for preprocessing and augmentation.

**Section 5.2 — DataLoaders** addresses efficient data delivery to the training loop: batching strategies, shuffling and sampling (including class-balanced sampling for imbalanced datasets), custom collate functions, multi-process loading, and memory pinning for GPU transfer.

**Section 5.3 — Training Loop** presents the anatomy of the training loop itself, including the validation loop, metrics tracking, integration with logging frameworks (TensorBoard, Weights & Biases), gradient clipping, mixed precision training, and reproducibility.

**Section 5.4 — Optimizers** provides a comprehensive treatment of gradient-based optimization: from vanilla SGD through momentum variants, adaptive methods (Adagrad, RMSprop, Adam and its extensions), and second-order methods (L-BFGS). Each optimizer is presented with its update rule, mathematical motivation, and practical guidance.

**Section 5.5 — Learning Rate Schedulers** covers strategies for adjusting the learning rate during training—step decay, exponential decay, cosine annealing, warmup strategies, the OneCycleLR policy, and plateau-based reduction.

**Section 5.6 — Overfitting and Generalization** addresses the central challenge of statistical learning: the bias-variance tradeoff, formal decomposition, overfitting detection, cross-validation, proper train/validation/test splitting, and early stopping.

**Section 5.7 — Evaluation Metrics** surveys metrics for classification (accuracy, precision, recall, F1, ROC-AUC, calibration) and regression (MSE, MAE, R²), with emphasis on choosing metrics that align with the problem's economic or scientific objectives.

**Section 5.8 — Hyperparameter Tuning** covers systematic approaches to hyperparameter selection: grid search, random search, Bayesian optimization, and analysis of hyperparameter importance.

**Section 5.9 — Model Persistence** covers saving and loading models via state dicts and checkpoints, exporting to ONNX for cross-framework deployment, and TorchScript for production inference.

!!! note "Relationship to Chapter 2 (Scikit-learn)"
    Several topics in this chapter parallel coverage in **[Chapter 2: Scikit-learn](../ch02/index.md)**—specifically cross-validation (§2.4 vs §5.6), hyperparameter search (§2.4 vs §5.8), and evaluation metrics (§2.5 vs §5.7). The two chapters differ in scope:

    - **Chapter 2** covers these topics through scikit-learn's API (`GridSearchCV`, `cross_val_score`, `classification_report`), suitable for classical ML models and sklearn-wrapped PyTorch models via Skorch.
    - **Chapter 5** covers the same concepts implemented directly in PyTorch training loops, where you control the training step, handle GPU transfers, and integrate with PyTorch-native tools like `torch.optim.lr_scheduler` and `torch.utils.data`.

    If you are using scikit-learn estimators (including Skorch-wrapped neural networks), Chapter 2's treatment is more directly applicable. If you are writing custom PyTorch training loops, use this chapter.

## Quantitative Finance Context

Every component of the training pipeline has domain-specific considerations in quantitative finance:

- **Data**: Financial time series exhibit non-stationarity, regime changes, and temporal dependencies that violate the i.i.d. assumption underlying standard data loading and splitting strategies.
- **Training**: Surrogate pricing models, neural calibration of stochastic volatility surfaces, and learned hedging strategies all require careful training loop design.
- **Optimization**: The loss landscapes of financial models often feature sharp minima and ill-conditioning, making optimizer selection critical.
- **Evaluation**: Standard accuracy metrics rarely align with financial objectives—Sharpe ratios, P&L distributions, and calibration error are more relevant.
- **Overfitting**: With limited non-stationary data and high noise-to-signal ratios, overfitting is the dominant failure mode in financial ML. Walk-forward validation and purged cross-validation replace standard k-fold.

## Prerequisites

This chapter assumes familiarity with:

- PyTorch tensors and autograd (Chapter 3)
- Neural network modules and loss functions (Chapter 4)
- Basic probability and statistics (Chapter 2)
