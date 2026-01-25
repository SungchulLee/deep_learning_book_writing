# L2 Regularization (Ridge)

## Overview

L2 regularization, also known as Ridge regression or weight decay, adds a penalty proportional to the squared magnitude of model weights to the loss function. Unlike L1 regularization, L2 encourages small but non-zero weights, resulting in smooth weight distributions that prevent any single weight from becoming too large.

## Mathematical Formulation

### Standard Loss with L2 Penalty

For a loss function $\mathcal{L}(\theta)$ with parameters $\theta = \{w_1, w_2, \ldots, w_n\}$, L2 regularization modifies the objective:

$$
\mathcal{L}_{\text{L2}}(\theta) = \mathcal{L}(\theta) + \lambda \sum_{i=1}^{n} w_i^2
$$

Equivalently, using vector notation with the squared L2 norm:

$$
\mathcal{L}_{\text{L2}}(\theta) = \mathcal{L}(\theta) + \lambda \|w\|_2^2
$$

where:

- $\mathcal{L}(\theta)$ is the original loss function
- $\lambda \geq 0$ is the regularization strength
- $\|w\|_2^2 = w^T w = \sum_{i=1}^{n} w_i^2$ is the squared L2 norm

### Linear Regression with L2 (Ridge Regression)

For linear regression with design matrix $X \in \mathbb{R}^{m \times n}$, target $y \in \mathbb{R}^m$, and weights $w \in \mathbb{R}^n$:

$$
\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \|w\|_2^2
$$

Expanding:

$$
\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2m} (Xw - y)^T(Xw - y) + \lambda w^T w
$$

### Gradient Derivation

The L2 penalty is differentiable everywhere:

$$
\frac{\partial}{\partial w_i} \left( \lambda \sum_{j=1}^{n} w_j^2 \right) = 2\lambda w_i
$$

In vector form:

$$
\nabla_w \left( \lambda \|w\|_2^2 \right) = 2\lambda w
$$

The full gradient for L2-regularized loss:

$$
\nabla_w \mathcal{L}_{\text{L2}} = \nabla_w \mathcal{L} + 2\lambda w
$$

### Closed-Form Solution for Ridge Regression

Setting the gradient to zero:

$$
\nabla_w \mathcal{L}_{\text{Ridge}} = \frac{1}{m} X^T(Xw - y) + 2\lambda w = 0
$$

Solving for $w$:

$$
\frac{1}{m} X^T X w + 2\lambda w = \frac{1}{m} X^T y
$$

$$
\left( \frac{1}{m} X^T X + 2\lambda I \right) w = \frac{1}{m} X^T y
$$

$$
w^* = \left( X^T X + 2m\lambda I \right)^{-1} X^T y
$$

**Key insight**: The term $2m\lambda I$ ensures the matrix is always invertible, even when $X^T X$ is singular (e.g., when $n > m$).

## Geometric Interpretation

### Constraint Region

L2 regularization is equivalent to constrained optimization with an L2 ball:

$$
\min_w \mathcal{L}(w) \quad \text{subject to} \quad \|w\|_2^2 \leq t
$$

The L2 ball in 2D is a circle:

$$
\|w\|_2^2 = w_1^2 + w_2^2 \leq t
$$

### Why L2 Shrinks but Doesn't Sparsify

The circular constraint region has no corners. The loss function's contours typically intersect the circle at a point where both coordinates are non-zero. This results in:

- All weights being shrunk toward zero
- But rarely exactly zero
- Smooth, continuous weight distributions

### Bayesian Interpretation

L2 regularization corresponds to a **Gaussian prior** on the weights:

$$
p(w) = \mathcal{N}(0, \sigma^2 I)
$$

The regularization strength relates to the prior variance:

$$
\lambda = \frac{1}{2\sigma^2}
$$

**MAP estimation** with this prior gives the Ridge solution:

$$
w_{\text{MAP}} = \arg\max_w \left[ \log p(y|X, w) + \log p(w) \right]
$$

## Weight Decay Interpretation

### Connection to Gradient Descent

In gradient descent, the L2 penalty adds a term that shrinks weights at each step:

$$
w_{t+1} = w_t - \eta \nabla_w \mathcal{L} - 2\eta\lambda w_t
$$

Rearranging:

$$
w_{t+1} = (1 - 2\eta\lambda) w_t - \eta \nabla_w \mathcal{L}
$$

The factor $(1 - 2\eta\lambda)$ multiplies the current weights, causing them to **decay** toward zero at each step. This is why L2 regularization is often called **weight decay**.

### AdamW: Decoupled Weight Decay

Standard Adam with L2 regularization doesn't perfectly decouple the adaptive learning rate from weight decay. AdamW fixes this:

```python
# Standard Adam with L2 (not ideal)
gradient = grad + 2 * lambda * w
# The regularization term gets scaled by adaptive lr

# AdamW (proper weight decay)
w = w - lr * adam_step(grad) - lr * lambda * w
# Weight decay applied directly, not through gradient
```

## PyTorch Implementation

### Manual L2 Regularization

```python
import torch
import torch.nn as nn
import torch.optim as optim

def l2_regularization(model: nn.Module, lambda_l2: float) -> torch.Tensor:
    """
    Compute L2 regularization penalty.
    
    Args:
        model: Neural network model
        lambda_l2: Regularization strength
        
    Returns:
        L2 penalty term (squared L2 norm of weights)
    """
    l2_penalty = torch.tensor(0., device=next(model.parameters()).device)
    for param in model.parameters():
        l2_penalty = l2_penalty + torch.sum(param ** 2)
    return lambda_l2 * l2_penalty


class L2RegularizedModel(nn.Module):
    """Model with built-in L2 regularization computation."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def l2_penalty(self, lambda_l2=0.01):
        """Compute L2 penalty for all parameters."""
        penalty = 0
        for param in self.parameters():
            penalty += torch.sum(param ** 2)
        return lambda_l2 * penalty
```

### Using Optimizer's weight_decay Parameter

PyTorch optimizers have built-in weight decay:

```python
# SGD with weight decay (implements L2 regularization)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# Adam with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# AdamW - proper decoupled weight decay (recommended)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Note**: The `weight_decay` parameter implements weight decay as:

$$
w_{t+1} = w_t - \eta \nabla \mathcal{L} - \eta \cdot \text{weight\_decay} \cdot w_t
$$

This is equivalent to L2 with $\lambda = \text{weight\_decay} / 2$ in the loss formulation.

### Complete Training Loop with L2

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_with_l2_regularization(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lambda_l2: float = 0.01,
    epochs: int = 100,
    lr: float = 0.001,
    use_adamw: bool = True
) -> dict:
    """
    Train model with L2 regularization.
    
    Args:
        model: Neural network
        train_loader: Training data
        val_loader: Validation data
        lambda_l2: L2 regularization strength
        epochs: Number of epochs
        lr: Learning rate
        use_adamw: Whether to use AdamW (True) or manual L2 (False)
        
    Returns:
        Training history
    """
    criterion = nn.MSELoss()
    
    if use_adamw:
        # Use optimizer's built-in weight decay
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2*lambda_l2)
        manual_l2 = False
    else:
        # Manual L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=lr)
        manual_l2 = True
    
    history = {'train_loss': [], 'val_loss': [], 'weight_norm': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            if manual_l2:
                l2_penalty = l2_regularization(model, lambda_l2)
                total_loss = loss + l2_penalty
            else:
                total_loss = loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        # Compute weight statistics
        total_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['weight_norm'].append(total_norm)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, "
                  f"Val={val_loss/len(val_loader):.4f}, ||w||={total_norm:.4f}")
    
    return history
```

### Selective L2 Regularization

Apply different regularization strengths to different layers:

```python
def create_param_groups_with_l2(model, base_lr=0.001, 
                                 layer_decay_rates=None):
    """
    Create parameter groups with layer-specific L2 regularization.
    
    Args:
        model: Neural network
        base_lr: Base learning rate
        layer_decay_rates: Dict mapping layer names to weight decay values
        
    Returns:
        List of parameter groups for optimizer
    """
    if layer_decay_rates is None:
        layer_decay_rates = {}
    
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Find matching layer decay rate
        weight_decay = 0.01  # default
        for layer_name, decay in layer_decay_rates.items():
            if layer_name in name:
                weight_decay = decay
                break
        
        # Don't regularize biases (common practice)
        if 'bias' in name:
            weight_decay = 0.0
        
        param_groups.append({
            'params': param,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    
    return param_groups


# Example usage
model = L2RegularizedModel(input_dim=20, hidden_dims=[128, 64], output_dim=1)

# Different regularization for different layers
layer_decays = {
    'network.0': 0.001,  # First layer: light regularization
    'network.2': 0.01,   # Second layer: medium regularization
    'network.4': 0.1,    # Output layer: heavy regularization
}

param_groups = create_param_groups_with_l2(model, layer_decay_rates=layer_decays)
optimizer = optim.AdamW(param_groups)
```

## Scikit-learn Implementation

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def ridge_regression_analysis(X, y, alphas=None):
    """
    Analyze Ridge regression across different regularization strengths.
    
    Args:
        X: Feature matrix
        y: Target vector
        alphas: Regularization values to try
        
    Returns:
        Optimal model and analysis results
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)
    
    # Cross-validation to find optimal alpha
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_scaled, y)
    
    print(f"Optimal alpha: {ridge_cv.alpha_:.6f}")
    print(f"R² score: {ridge_cv.score(X_scaled, y):.4f}")
    print(f"Coefficient range: [{ridge_cv.coef_.min():.4f}, {ridge_cv.coef_.max():.4f}]")
    print(f"Coefficient L2 norm: {np.linalg.norm(ridge_cv.coef_):.4f}")
    
    return ridge_cv, scaler


def plot_ridge_coefficients(X, y, alphas):
    """Visualize how coefficients change with regularization strength."""
    import matplotlib.pyplot as plt
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    coefs = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        coefs.append(ridge.coef_)
    
    coefs = np.array(coefs)
    
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], label=f'Feature {i}')
    
    plt.xscale('log')
    plt.xlabel('Regularization strength (α)')
    plt.ylabel('Coefficient value')
    plt.title('Ridge Coefficient Shrinkage')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=8)
    
    return coefs
```

## Singular Value Decomposition Perspective

Ridge regression has an elegant interpretation via SVD. If $X = U \Sigma V^T$, then:

$$
w_{\text{Ridge}} = V D_\lambda \Sigma^{-1} U^T y
$$

where $D_\lambda$ is a diagonal matrix with entries:

$$
d_i = \frac{\sigma_i^2}{\sigma_i^2 + \lambda}
$$

**Interpretation**: Ridge shrinks coefficients in directions of small singular values more aggressively. This prevents overfitting on noisy directions while preserving signal in strong directions.

```python
def ridge_via_svd(X, y, lambda_reg):
    """
    Compute Ridge solution using SVD.
    
    This reveals how Ridge shrinks along different
    singular value directions.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    # Shrinkage factors
    d = s ** 2 / (s ** 2 + lambda_reg)
    
    # Ridge solution
    w_ridge = V @ np.diag(d / s) @ U.T @ y
    
    return w_ridge, d, s
```

## Effective Degrees of Freedom

Ridge regression has effective degrees of freedom:

$$
\text{df}_\lambda = \sum_{i=1}^{n} \frac{\sigma_i^2}{\sigma_i^2 + \lambda} = \text{tr}(X(X^TX + \lambda I)^{-1}X^T)
$$

This measures model complexity: as $\lambda \to 0$, $\text{df} \to n$ (OLS); as $\lambda \to \infty$, $\text{df} \to 0$.

## Comparison: L2 vs L1

| Aspect | L2 (Ridge) | L1 (Lasso) |
|--------|------------|------------|
| Penalty | $\lambda \sum w_i^2$ | $\lambda \sum \|w_i\|$ |
| Constraint shape | Circle/Sphere | Diamond/Cross-polytope |
| Sparse solutions | No | Yes |
| Closed-form | Yes | No |
| Differentiable | Yes | No (at 0) |
| Correlated features | Shares weight equally | Selects one |
| Bayesian prior | Gaussian | Laplace |

## When to Use L2 Regularization

### Good Use Cases

1. **Prevent large weights**: When extreme weights cause instability
2. **Correlated features**: L2 handles multicollinearity gracefully
3. **All features relevant**: When you believe all features contribute
4. **Numerical stability**: Adding $\lambda I$ ensures invertibility
5. **Deep learning**: Standard regularization for neural networks

### Hyperparameter Selection

```python
from sklearn.model_selection import GridSearchCV

def select_optimal_l2(model_class, X, y, param_grid, cv=5):
    """
    Select optimal L2 regularization strength via grid search.
    
    Args:
        model_class: Model class (e.g., Ridge)
        X: Features
        y: Targets
        param_grid: Parameter grid (e.g., {'alpha': [0.01, 0.1, 1.0]})
        cv: Cross-validation folds
        
    Returns:
        Best model and search results
    """
    grid_search = GridSearchCV(
        model_class(),
        param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.cv_results_
```

## Practical Guidelines

### Regularization Strength Selection

- **Too small ($\lambda \to 0$)**: Behaves like unregularized model, potential overfitting
- **Too large ($\lambda \to \infty$)**: All weights shrink to zero, underfitting
- **Optimal**: Balance bias-variance trade-off

### Feature Scaling

**Always standardize features before applying L2 regularization.** The penalty treats all weights equally, so features with different scales would be penalized unequally.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# Correct approach: scale then regularize
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
```

### Bias Terms

**Don't regularize bias terms.** The bias (intercept) shouldn't be penalized as it just shifts predictions:

```python
# In PyTorch, separate weights and biases
def l2_regularization_weights_only(model, lambda_l2):
    """L2 penalty on weights only, not biases."""
    penalty = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only weight matrices
            penalty += torch.sum(param ** 2)
    return lambda_l2 * penalty
```

## References

1. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
3. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
