# L1 Regularization (Lasso)

## Overview

L1 regularization, also known as Lasso (Least Absolute Shrinkage and Selection Operator), adds a penalty proportional to the absolute value of model weights to the loss function. This technique promotes sparsity in the learned parameters, effectively performing automatic feature selection by driving irrelevant feature weights to exactly zero.

## Mathematical Formulation

### Standard Loss with L1 Penalty

For a loss function $\mathcal{L}(\theta)$ with parameters $\theta = \{w_1, w_2, \ldots, w_n\}$, L1 regularization modifies the objective function:

$$
\mathcal{L}_{\text{L1}}(\theta) = \mathcal{L}(\theta) + \lambda \sum_{i=1}^{n} |w_i|
$$

where:

- $\mathcal{L}(\theta)$ is the original loss function (e.g., MSE, cross-entropy)
- $\lambda \geq 0$ is the regularization strength (hyperparameter)
- $\sum_{i=1}^{n} |w_i|$ is the L1 norm of the weight vector

### Linear Regression with L1 (Lasso Regression)

For linear regression with design matrix $X \in \mathbb{R}^{m \times n}$, target $y \in \mathbb{R}^m$, and weights $w \in \mathbb{R}^n$:

$$
\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \|w\|_1
$$

Expanding the terms:

$$
\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2m} \sum_{j=1}^{m} \left( \sum_{i=1}^{n} x_{ji} w_i - y_j \right)^2 + \lambda \sum_{i=1}^{n} |w_i|
$$

### Gradient and Subgradient

The L1 norm is not differentiable at $w_i = 0$. We use the subgradient:

$$
\frac{\partial}{\partial w_i} |w_i| = 
\begin{cases}
+1 & \text{if } w_i > 0 \\
-1 & \text{if } w_i < 0 \\
[-1, +1] & \text{if } w_i = 0
\end{cases}
$$

This can be written using the sign function:

$$
\frac{\partial}{\partial w_i} |w_i| = \text{sign}(w_i) = 
\begin{cases}
+1 & \text{if } w_i > 0 \\
-1 & \text{if } w_i < 0 \\
0 & \text{if } w_i = 0
\end{cases}
$$

The full gradient for the L1-regularized loss:

$$
\nabla_{w} \mathcal{L}_{\text{L1}} = \nabla_{w} \mathcal{L} + \lambda \cdot \text{sign}(w)
$$

## Geometric Interpretation

### Constraint Region

L1 regularization is equivalent to constrained optimization with an L1 ball constraint:

$$
\min_w \mathcal{L}(w) \quad \text{subject to} \quad \|w\|_1 \leq t
$$

The L1 ball in 2D forms a diamond (rotated square) shape with corners on the axes:

$$
\|w\|_1 = |w_1| + |w_2| \leq t
$$

### Sparsity from Geometry

The diamond shape of the L1 constraint region has **corners** at the coordinate axes. When the loss function's contours intersect the constraint region, they are more likely to touch at these corners, resulting in solutions where some weights are exactly zero.

**Comparison with L2 (circular constraint):**

| Property | L1 (Diamond) | L2 (Circle) |
|----------|--------------|-------------|
| Corner points | Yes (on axes) | No |
| Sparse solutions | Yes | Rarely |
| Differentiable | No (at corners) | Yes |

## Sparsity and Feature Selection

### Why L1 Produces Sparse Solutions

Consider the optimization landscape for a single weight $w_i$:

1. **Away from zero**: The gradient includes $\lambda \cdot \text{sign}(w_i)$, pushing $w_i$ toward zero
2. **At zero**: The subgradient includes the interval $[-\lambda, +\lambda]$
3. **Condition for staying at zero**: If the gradient of the original loss at $w_i = 0$ lies within $[-\lambda, +\lambda]$, then $w_i = 0$ is optimal

This means features with weak predictive power have their weights driven to exactly zero.

### Soft Thresholding Operator

For L1-regularized least squares, the closed-form solution for each coordinate (in coordinate descent) is:

$$
w_i^* = S_{\lambda}(z_i) = \text{sign}(z_i) \cdot \max(|z_i| - \lambda, 0)
$$

where $z_i$ is the ordinary least squares solution for $w_i$ with other weights fixed. This is the **soft thresholding** or **shrinkage** operator.

## PyTorch Implementation

### Manual L1 Regularization

```python
import torch
import torch.nn as nn

def l1_regularization(model: nn.Module, lambda_l1: float) -> torch.Tensor:
    """
    Compute L1 regularization penalty for all model parameters.
    
    Args:
        model: Neural network model
        lambda_l1: Regularization strength
        
    Returns:
        L1 penalty term
    """
    l1_penalty = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_penalty = l1_penalty + torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty


class L1RegularizedTrainer:
    """Trainer with L1 regularization support."""
    
    def __init__(self, model, criterion, optimizer, lambda_l1=0.01):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lambda_l1 = lambda_l1
    
    def train_step(self, X, y):
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(X)
        loss = self.criterion(predictions, y)
        
        # Add L1 penalty
        l1_penalty = l1_regularization(self.model, self.lambda_l1)
        total_loss = loss + l1_penalty
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return loss.item(), l1_penalty.item()
```

### Using PyTorch's Built-in Regularizers

```python
import torch.nn as nn
from torch.nn.utils import parametrize

class L1Regularizer(nn.Module):
    """L1 weight regularizer as a parametrization."""
    
    def __init__(self, lambda_l1: float):
        super().__init__()
        self.lambda_l1 = lambda_l1
    
    def forward(self, weight):
        return weight
    
    def right_inverse(self, weight):
        return weight


def add_l1_regularization_to_loss(model, base_loss, lambda_l1):
    """Add L1 regularization to any loss function."""
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return base_loss + lambda_l1 * l1_norm
```

### Neural Network with L1 Regularization

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SparseNN(nn.Module):
    """Neural network designed for sparse feature learning."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_l1_norm(self):
        """Compute L1 norm of all weights."""
        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return l1_norm
    
    def count_zero_weights(self, threshold=1e-6):
        """Count weights that are effectively zero."""
        total = 0
        zeros = 0
        for param in self.parameters():
            total += param.numel()
            zeros += (param.abs() < threshold).sum().item()
        return zeros, total


def train_with_l1(model, train_loader, val_loader, 
                  lambda_l1=0.01, epochs=100, lr=0.001):
    """
    Train model with L1 regularization.
    
    Args:
        model: Neural network
        train_loader: Training data loader
        val_loader: Validation data loader
        lambda_l1: L1 regularization strength
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Training history dictionary
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'l1_norm': [], 'sparsity': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            l1_penalty = lambda_l1 * model.get_l1_norm()
            loss = mse_loss + l1_penalty
            
            loss.backward()
            optimizer.step()
            train_loss += mse_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        # Track metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        zeros, total = model.count_zero_weights()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['l1_norm'].append(model.get_l1_norm().item())
        history['sparsity'].append(zeros / total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Sparsity={zeros/total:.2%}")
    
    return history
```

### Proximal Gradient Descent for L1

For more efficient optimization with L1, use proximal gradient descent:

```python
def proximal_l1(weights: torch.Tensor, lambda_l1: float, 
                lr: float) -> torch.Tensor:
    """
    Proximal operator for L1 regularization (soft thresholding).
    
    Args:
        weights: Parameter tensor
        lambda_l1: Regularization strength
        lr: Learning rate
        
    Returns:
        Soft-thresholded weights
    """
    threshold = lambda_l1 * lr
    return torch.sign(weights) * torch.clamp(torch.abs(weights) - threshold, min=0)


class ProximalL1Optimizer:
    """Optimizer implementing proximal gradient descent for L1."""
    
    def __init__(self, model, lr=0.01, lambda_l1=0.01):
        self.model = model
        self.lr = lr
        self.lambda_l1 = lambda_l1
    
    def step(self):
        """Perform one optimization step with proximal update."""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Gradient step
                    param.data -= self.lr * param.grad
                    # Proximal step (soft thresholding)
                    param.data = proximal_l1(param.data, self.lambda_l1, self.lr)
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
```

## Scikit-learn Implementation

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def lasso_feature_selection(X, y, alphas=None):
    """
    Perform feature selection using Lasso with cross-validation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        alphas: Regularization values to try
        
    Returns:
        Selected feature indices and Lasso model
    """
    # Standardize features (important for L1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use cross-validation to find optimal alpha
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)
    
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_scaled, y)
    
    print(f"Optimal alpha: {lasso_cv.alpha_:.6f}")
    print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}/{len(lasso_cv.coef_)}")
    
    # Get selected features
    selected_features = np.where(lasso_cv.coef_ != 0)[0]
    
    return selected_features, lasso_cv, scaler


def compare_regularization_strengths(X, y, alphas):
    """Compare sparsity patterns across different alpha values."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        n_nonzero = np.sum(lasso.coef_ != 0)
        results.append({
            'alpha': alpha,
            'n_nonzero': n_nonzero,
            'coefficients': lasso.coef_.copy()
        })
    
    return results
```

## Regularization Path

The **regularization path** shows how coefficients change as $\lambda$ varies:

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

def plot_lasso_path(X, y, eps=1e-3, n_alphas=100):
    """
    Plot the Lasso regularization path.
    
    Args:
        X: Feature matrix
        y: Target vector
        eps: Length of path (alpha_min / alpha_max)
        n_alphas: Number of alpha values
    """
    # Standardize
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    y_centered = y - y.mean()
    
    # Compute Lasso path
    alphas, coefs, _ = lasso_path(X_scaled, y_centered, eps=eps, n_alphas=n_alphas)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[0]):
        plt.plot(alphas, coefs[i], label=f'Feature {i}')
    
    plt.xscale('log')
    plt.xlabel('Regularization strength (α)')
    plt.ylabel('Coefficient value')
    plt.title('Lasso Regularization Path')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.gca().invert_xaxis()  # Decreasing alpha
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    
    return alphas, coefs
```

## Comparison: L1 vs L2

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | $\lambda \sum \|w_i\|$ | $\lambda \sum w_i^2$ |
| Constraint shape | Diamond | Circle |
| Sparse solutions | Yes | No |
| Feature selection | Built-in | No |
| Differentiable | No (at 0) | Yes |
| Correlated features | Selects one | Shares weight |
| Closed-form solution | No | Yes |

## Hyperparameter Selection

### Cross-Validation for $\lambda$

```python
from sklearn.model_selection import cross_val_score
import numpy as np

def select_lambda_cv(X, y, lambdas, cv=5):
    """
    Select optimal lambda using cross-validation.
    
    Args:
        X: Features
        y: Targets
        lambdas: Lambda values to evaluate
        cv: Number of CV folds
        
    Returns:
        Optimal lambda and CV scores
    """
    from sklearn.linear_model import Lasso
    
    cv_scores = []
    for lam in lambdas:
        model = Lasso(alpha=lam)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())
    
    optimal_idx = np.argmin(cv_scores)
    return lambdas[optimal_idx], cv_scores
```

### Information Criteria

Alternatively, use AIC or BIC for model selection:

$$
\text{AIC} = 2k - 2\ln(\hat{L})
$$
$$
\text{BIC} = k\ln(n) - 2\ln(\hat{L})
$$

where $k$ is the number of non-zero parameters.

## Applications in Deep Learning

### Sparse Input Layers

L1 regularization on the first layer promotes input feature selection:

```python
class SparseInputNetwork(nn.Module):
    """Network with L1-regularized input layer for feature selection."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, input_l1=0.01):
        super().__init__()
        self.input_l1 = input_l1
        
        # First layer (heavily regularized)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.hidden = nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        return self.hidden(x)
    
    def get_input_l1_penalty(self):
        """L1 penalty for input layer only."""
        return self.input_l1 * torch.sum(torch.abs(self.input_layer.weight))
```

## Practical Guidelines

### When to Use L1 Regularization

1. **Feature selection is needed**: When you have many features and want to identify the most important ones
2. **Interpretability matters**: Sparse models are easier to interpret
3. **High-dimensional data**: When $p > n$ (more features than samples)
4. **Suspected irrelevant features**: When many features are likely noise

### Choosing $\lambda$

- **Too small**: Weak regularization, overfitting, many non-zero weights
- **Too large**: Underfitting, too few features selected
- **Use cross-validation**: Let the data guide the choice
- **Consider the scale**: Standardize features before applying L1

### Common Pitfalls

1. **Feature scaling**: L1 penalizes based on magnitude; unscaled features lead to biased selection
2. **Correlated features**: L1 arbitrarily selects one among correlated features
3. **Non-differentiability**: Standard gradient descent may struggle; use proximal methods
4. **Bias-variance trade-off**: Higher $\lambda$ increases bias but reduces variance

## References

1. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity: The Lasso and Generalizations*. CRC Press.
3. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
