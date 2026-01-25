# Loss Function Overview

Loss functions, also called cost functions or objective functions, are the mathematical compass that guides neural network training. They quantify how far a model's predictions deviate from the true values, providing the signal that optimizers use to update parameters.

## What is a Loss Function?

A loss function $\mathcal{L}(\hat{y}, y)$ measures the discrepancy between predicted values $\hat{y}$ and ground truth values $y$. The fundamental goal of training is to minimize this loss:

$$\theta^* = \arg\min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(x), y) \right]$$

where $f_\theta$ represents the model parameterized by $\theta$ and $\mathcal{D}$ is the data distribution.

The loss function serves several critical purposes: it provides a differentiable objective for gradient-based optimization, it encodes domain knowledge about what constitutes a "good" prediction, and it implicitly defines the probabilistic assumptions of the model through the connection to maximum likelihood estimation.

## Loss Functions in PyTorch

PyTorch provides three primary approaches for computing loss values, each with distinct use cases.

### Method 1: Manual Computation

Manual computation offers maximum flexibility and transparency, particularly useful for understanding fundamentals or implementing custom losses:

```python
import torch

# Sample data
actual = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])
predicted = torch.tensor([140.0, 210.0, 245.0, 310.0, 360.0])

# Manual MSE calculation
errors = actual - predicted
squared_errors = errors ** 2
mse_manual = torch.mean(squared_errors)
print(f"Manual MSE: {mse_manual.item():.4f}")  # Output: 82.0000
```

### Method 2: Functional API

The functional API (`torch.nn.functional`) provides optimized, stateless functions:

```python
import torch.nn.functional as F

mse_functional = F.mse_loss(predicted, actual)
print(f"Functional MSE: {mse_functional.item():.4f}")  # Output: 82.0000
```

### Method 3: Module Classes (Recommended)

The module-based approach (`torch.nn`) creates configurable, reusable loss objects—the preferred pattern for training loops:

```python
import torch.nn as nn

criterion = nn.MSELoss()
mse_class = criterion(predicted, actual)
print(f"Class-based MSE: {mse_class.item():.4f}")  # Output: 82.0000
```

The class-based approach offers configuration options, cleaner code organization, and seamless integration with PyTorch's module system.

## Reduction Methods

Loss functions aggregate per-sample errors into a scalar. The `reduction` parameter controls this aggregation:

```python
# Mean reduction (default): average of all errors
criterion_mean = nn.MSELoss(reduction='mean')
loss_mean = criterion_mean(predicted, actual)  # 82.0

# Sum reduction: sum of all errors
criterion_sum = nn.MSELoss(reduction='sum')
loss_sum = criterion_sum(predicted, actual)  # 410.0

# No reduction: individual errors preserved
criterion_none = nn.MSELoss(reduction='none')
loss_none = criterion_none(predicted, actual)  # tensor([100., 100., 25., 100., 100.])
```

The relationship between reduction methods follows: $\text{loss}_{\text{sum}} = \text{loss}_{\text{mean}} \times n$, where $n$ is the number of samples.

**Choosing reduction methods:** Use `'mean'` for most training scenarios as it provides scale-invariant gradients regardless of batch size. Use `'sum'` when gradient magnitude should scale with batch size. Use `'none'` when implementing custom weighting schemes or for per-sample analysis.

## Interpreting Loss Values

Loss values are relative measures—their absolute magnitude depends on the specific loss function and data scale. What matters is the trajectory:

```python
# Perfect predictions → Loss = 0
perfect = actual.clone()
print(f"Perfect: {criterion(perfect, actual).item():.4f}")  # 0.0000

# Better predictions → Lower loss
better = torch.tensor([148.0, 202.0, 249.0, 301.0, 351.0])
print(f"Better: {criterion(better, actual).item():.4f}")  # 5.0000

# Worse predictions → Higher loss
worse = torch.tensor([120.0, 230.0, 220.0, 330.0, 380.0])
print(f"Worse: {criterion(worse, actual).item():.4f}")  # 900.0000
```

## Taxonomy of Loss Functions

Loss functions broadly categorize into two families based on the prediction task:

| Category | Output Type | Common Losses | Typical Use Cases |
|----------|-------------|---------------|-------------------|
| Regression | Continuous values | MSE, MAE, Huber, Quantile | Price prediction, time series |
| Classification | Discrete categories | Cross-Entropy, BCE, Focal | Image classification, NLP |

Within each category, different losses encode different assumptions about error distributions and robustness properties. The following sections explore these in detail.

## Connection to Maximum Likelihood

Many loss functions emerge naturally from maximum likelihood estimation under specific distributional assumptions:

| Distribution | Log-Likelihood | Resulting Loss |
|--------------|----------------|----------------|
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $-\frac{1}{2\sigma^2}(y - \mu)^2$ | MSE |
| Laplace | $-\frac{1}{b}\|y - \mu\|$ | MAE |
| Bernoulli | $y\log(p) + (1-y)\log(1-p)$ | Binary Cross-Entropy |
| Categorical | $\sum_k y_k \log(p_k)$ | Cross-Entropy |

This connection provides principled guidance: choose the loss function whose implied distribution matches your beliefs about the data-generating process.

## Key Takeaways

Loss functions are not merely technical necessities but encode fundamental assumptions about what constitutes good predictions. The choice of loss function affects gradient dynamics, sensitivity to outliers, and the implicit probabilistic model. PyTorch's three-tier API (manual, functional, class-based) provides flexibility for different use cases, with the class-based approach recommended for production training loops.
