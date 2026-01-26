# Model Specification and Assumptions

## Overview

Linear regression is the foundational supervised learning algorithm that models the relationship between input features and a continuous target variable as a linear function. Understanding its formal specification and underlying assumptions is essential for proper application and interpretation.

## The Linear Model

### Univariate Linear Regression

For a single input feature $x$, the linear regression model is:

$$\hat{y} = wx + b$$

where:
- $\hat{y}$ is the predicted output
- $x$ is the input feature
- $w$ is the **weight** (slope) parameter
- $b$ is the **bias** (intercept) parameter

### Multivariate Linear Regression

For $d$ input features $\mathbf{x} = [x_1, x_2, \ldots, x_d]^T$, the model generalizes to:

$$\hat{y} = \mathbf{w}^T\mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

In matrix form for $n$ samples with design matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}$$

### Compact Notation with Bias Absorption

By augmenting the feature vector with a constant 1, we can absorb the bias into the weight vector:

$$\tilde{\mathbf{x}} = [1, x_1, x_2, \ldots, x_d]^T, \quad \tilde{\mathbf{w}} = [b, w_1, w_2, \ldots, w_d]^T$$

This gives:

$$\hat{y} = \tilde{\mathbf{w}}^T\tilde{\mathbf{x}}$$

## Statistical Model Specification

### Probabilistic Formulation

Linear regression assumes the target variable follows:

$$y = \mathbf{w}^T\mathbf{x} + b + \epsilon$$

where $\epsilon$ is a random error term capturing the deviation from the linear relationship.

### The Gaussian Noise Model

Under the standard assumptions, we model:

$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

This implies:

$$y | \mathbf{x} \sim \mathcal{N}(\mathbf{w}^T\mathbf{x} + b, \sigma^2)$$

The conditional distribution of $y$ given $\mathbf{x}$ is Gaussian with:
- Mean: $\mu = \mathbf{w}^T\mathbf{x} + b$
- Variance: $\sigma^2$ (constant)

## Classical Assumptions (Gauss-Markov)

For the Ordinary Least Squares (OLS) estimator to be the Best Linear Unbiased Estimator (BLUE), the following assumptions must hold:

### 1. Linearity

The relationship between features and target is linear in parameters:

$$E[y|\mathbf{x}] = \mathbf{w}^T\mathbf{x} + b$$

**Implications:**
- The model correctly specifies the functional form
- Nonlinear relationships require feature transformations

### 2. Strict Exogeneity

The error term has zero conditional mean:

$$E[\epsilon | \mathbf{X}] = 0$$

**Implications:**
- Features are uncorrelated with the error
- No omitted variable bias
- No measurement error in features

### 3. Homoscedasticity

The error variance is constant across all observations:

$$\text{Var}(\epsilon_i | \mathbf{x}_i) = \sigma^2 \quad \forall i$$

**Violations (Heteroscedasticity):**
- Common in financial data where variance scales with magnitude
- Detected via residual plots or formal tests (Breusch-Pagan, White)

### 4. No Autocorrelation

Errors are uncorrelated across observations:

$$\text{Cov}(\epsilon_i, \epsilon_j) = 0 \quad \forall i \neq j$$

**Implications:**
- Crucial for time series data
- Violated in presence of serial correlation
- Detected via Durbin-Watson test

### 5. No Perfect Multicollinearity

The design matrix $\mathbf{X}$ has full column rank:

$$\text{rank}(\mathbf{X}) = d$$

**Implications:**
- $\mathbf{X}^T\mathbf{X}$ is invertible
- Features are not perfectly linearly dependent
- Near-multicollinearity causes numerical instability

## PyTorch Model Implementation

### Basic Linear Model

```python
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    """
    Simple linear regression model: y = Xw + b
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features (default: 1)
    """
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearRegressionModel, self).__init__()
        # nn.Linear implements: y = x @ W.T + b
        # Weight shape: (output_dim, input_dim)
        # Bias shape: (output_dim,)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.linear(x)

# Example usage
n_features = 8
model = LinearRegressionModel(input_dim=n_features)

# Inspect parameters
print(f"Weight shape: {model.linear.weight.shape}")  # (1, 8)
print(f"Bias shape: {model.linear.bias.shape}")      # (1,)
```

### Manual Implementation (Educational)

```python
def linear_model_manual(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Manual linear regression forward pass
    
    Args:
        X: Input features (n_samples, n_features)
        w: Weight vector (n_features,) or (n_features, 1)
        b: Bias scalar or (1,)
    
    Returns:
        Predictions (n_samples,) or (n_samples, 1)
    """
    # Matrix multiplication for batch predictions
    return X @ w + b

# Example
n_samples, n_features = 100, 5
X = torch.randn(n_samples, n_features)
w = torch.randn(n_features, 1)
b = torch.tensor([0.5])

y_pred = linear_model_manual(X, w, b)
print(f"Predictions shape: {y_pred.shape}")  # (100, 1)
```

## Data Requirements

### Feature Scaling

Linear regression is sensitive to feature scales when using gradient-based optimization:

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardization: (x - mean) / std
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics!

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
```

**Why scaling matters:**
1. Gradient descent converges faster with normalized features
2. Prevents features with large magnitudes from dominating
3. Makes learning rate selection easier

### Data Shapes

```python
# Correct shapes for PyTorch
# X: (n_samples, n_features)
# y: (n_samples, 1) or (n_samples,)

X = torch.randn(100, 8)        # 100 samples, 8 features
y = torch.randn(100, 1)        # 100 targets (column vector)

# Common mistake: flat y array
y_flat = torch.randn(100)      # Shape (100,)
y_correct = y_flat.reshape(-1, 1)  # Shape (100, 1)
```

## Verification of Assumptions

### Checking Linearity

```python
import matplotlib.pyplot as plt

def plot_partial_residuals(X, y, feature_idx, model):
    """Plot residuals against individual features"""
    with torch.no_grad():
        y_pred = model(X)
        residuals = y - y_pred
    
    plt.scatter(X[:, feature_idx].numpy(), residuals.numpy(), alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(f'Feature {feature_idx}')
    plt.ylabel('Residuals')
    plt.title('Partial Residual Plot')
```

### Checking Homoscedasticity

```python
def plot_residuals_vs_fitted(y_true, y_pred):
    """Check for constant variance"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 4))
    
    # Residuals vs fitted values
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred.numpy(), residuals.numpy(), alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    
    # Residual histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals.numpy(), bins=30, edgecolor='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
```

## Summary

| Component | Description | PyTorch Implementation |
|-----------|-------------|----------------------|
| Model | $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ | `nn.Linear(in_features, out_features)` |
| Parameters | Weight $\mathbf{w}$, Bias $b$ | `model.weight`, `model.bias` |
| Forward Pass | Matrix multiplication + bias | `model(X)` |
| Assumptions | Linearity, exogeneity, homoscedasticity, no autocorrelation, no multicollinearity | Validate via residual diagnostics |

## Key Takeaways

1. **Linear in parameters**: The model is linear in weights, but features can be transformed
2. **Assumptions matter**: Violations affect estimator properties and inference
3. **Feature scaling**: Essential for gradient-based optimization
4. **Shape consistency**: Maintain proper tensor dimensions throughout
5. **Diagnostic checks**: Always verify assumptions with residual analysis

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
- PyTorch Documentation: [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
