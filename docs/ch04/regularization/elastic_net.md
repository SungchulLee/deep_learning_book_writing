# Elastic Net Regularization

## Overview

Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization, inheriting benefits from both approaches. It encourages sparse models like Lasso while maintaining the stability of Ridge regression, making it particularly effective when dealing with correlated features.

## Mathematical Formulation

### Combined Penalty

The Elastic Net penalty is a weighted combination of L1 and L2 norms:

$$
\Omega(w) = \alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2
$$

where:

- $\alpha \in [0, 1]$ is the mixing parameter
- $\alpha = 1$ gives pure L1 (Lasso)
- $\alpha = 0$ gives pure L2 (Ridge)
- $0 < \alpha < 1$ gives Elastic Net

### Full Objective Function

For linear regression:

$$
\mathcal{L}_{\text{ElasticNet}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \left( \alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2 \right)
$$

This can be rewritten with separate regularization parameters:

$$
\mathcal{L}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2
$$

where $\lambda_1 = \lambda \alpha$ and $\lambda_2 = \lambda (1 - \alpha) / 2$.

### Gradient

The gradient combines L1's subgradient and L2's gradient:

$$
\nabla_w \mathcal{L} = \frac{1}{m} X^T(Xw - y) + \lambda \alpha \cdot \text{sign}(w) + \lambda(1 - \alpha) w
$$

The non-differentiability at zero from the L1 term remains, requiring subgradient methods or proximal optimization.

## Why Elastic Net?

### Limitations of Pure L1 (Lasso)

1. **Correlated features**: Lasso arbitrarily selects one among correlated features
2. **Saturation**: When $n < p$, Lasso selects at most $n$ features
3. **Instability**: Small data changes can flip feature selection

### Limitations of Pure L2 (Ridge)

1. **No sparsity**: All coefficients remain non-zero
2. **No feature selection**: Cannot identify irrelevant features
3. **Interpretability**: Dense models are harder to interpret

### Elastic Net's Solution

Elastic Net addresses these by:

1. **Grouped selection**: Tends to select/deselect groups of correlated features together
2. **No saturation**: Can select more than $n$ features
3. **Stability**: L2 component stabilizes selection among correlated features

## Geometric Interpretation

### Constraint Region Shape

The Elastic Net constraint region interpolates between the L1 diamond and L2 circle:

$$
\alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2 \leq t
$$

This creates a "rounded diamond" shape with:

- Corners (from L1) that promote sparsity
- Curved edges (from L2) that prevent extreme corner solutions

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_elastic_net_constraint(alphas=[0.0, 0.3, 0.7, 1.0]):
    """Visualize Elastic Net constraint regions for different mixing parameters."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    theta = np.linspace(0, 2*np.pi, 1000)
    
    for ax, alpha in zip(axes, alphas):
        # For each angle, find the boundary point
        w1_vals = []
        w2_vals = []
        
        for t in theta:
            # Direction
            d1, d2 = np.cos(t), np.sin(t)
            
            # Find r such that alpha*||w||_1 + (1-alpha)/2*||w||_2^2 = 1
            # Using numerical search
            for r in np.linspace(0.01, 3, 1000):
                w1, w2 = r * d1, r * d2
                penalty = alpha * (abs(w1) + abs(w2)) + (1-alpha)/2 * (w1**2 + w2**2)
                if penalty >= 1:
                    w1_vals.append(w1)
                    w2_vals.append(w2)
                    break
        
        ax.plot(w1_vals, w2_vals, 'b-', linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'α = {alpha}' + (' (Ridge)' if alpha == 0 else ' (Lasso)' if alpha == 1 else ''))
        ax.set_xlabel('$w_1$')
        ax.set_ylabel('$w_2$')
    
    plt.tight_layout()
    return fig
```

## PyTorch Implementation

### Manual Elastic Net Regularization

```python
import torch
import torch.nn as nn
import torch.optim as optim

def elastic_net_penalty(model: nn.Module, lambda_reg: float, 
                        alpha: float) -> torch.Tensor:
    """
    Compute Elastic Net regularization penalty.
    
    Args:
        model: Neural network model
        lambda_reg: Overall regularization strength
        alpha: Mixing parameter (0=Ridge, 1=Lasso)
        
    Returns:
        Elastic Net penalty term
    """
    l1_penalty = torch.tensor(0., device=next(model.parameters()).device)
    l2_penalty = torch.tensor(0., device=next(model.parameters()).device)
    
    for param in model.parameters():
        l1_penalty = l1_penalty + torch.sum(torch.abs(param))
        l2_penalty = l2_penalty + torch.sum(param ** 2)
    
    return lambda_reg * (alpha * l1_penalty + (1 - alpha) / 2 * l2_penalty)


class ElasticNetRegularizedModel(nn.Module):
    """Neural network with Elastic Net regularization."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 lambda_reg=0.01, alpha=0.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        
        # Build network
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
    
    def get_elastic_net_penalty(self):
        """Compute Elastic Net penalty."""
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum((p ** 2).sum() for p in self.parameters())
        return self.lambda_reg * (self.alpha * l1_norm + 
                                   (1 - self.alpha) / 2 * l2_norm)
    
    def get_sparsity(self, threshold=1e-6):
        """Compute fraction of near-zero weights."""
        total = 0
        zeros = 0
        for param in self.parameters():
            total += param.numel()
            zeros += (param.abs() < threshold).sum().item()
        return zeros / total
```

### Training with Elastic Net

```python
def train_elastic_net_model(
    model: ElasticNetRegularizedModel,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    Train model with Elastic Net regularization.
    
    Args:
        model: Model with Elastic Net built-in
        train_loader: Training data
        val_loader: Validation data
        epochs: Number of epochs
        lr: Learning rate
        
    Returns:
        Training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'penalty': [], 'sparsity': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            penalty = model.get_elastic_net_penalty()
            total_loss = mse_loss + penalty
            
            total_loss.backward()
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
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['penalty'].append(model.get_elastic_net_penalty().item())
        history['sparsity'].append(model.get_sparsity())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, "
                  f"Val={val_loss/len(val_loader):.4f}, "
                  f"Sparsity={model.get_sparsity():.2%}")
    
    return history
```

### Proximal Gradient Descent for Elastic Net

```python
def proximal_elastic_net(w: torch.Tensor, lambda_reg: float, 
                          alpha: float, lr: float) -> torch.Tensor:
    """
    Proximal operator for Elastic Net.
    
    The proximal operator for Elastic Net is soft thresholding
    followed by scaling (due to the L2 component).
    
    Args:
        w: Weight tensor
        lambda_reg: Regularization strength
        alpha: Mixing parameter
        lr: Learning rate
        
    Returns:
        Updated weights after proximal step
    """
    # L1 threshold
    l1_threshold = lambda_reg * alpha * lr
    
    # L2 scaling factor
    l2_scale = 1.0 / (1.0 + lambda_reg * (1 - alpha) * lr)
    
    # Soft thresholding then scale
    soft_thresh = torch.sign(w) * torch.clamp(torch.abs(w) - l1_threshold, min=0)
    return l2_scale * soft_thresh


class ProximalElasticNetOptimizer:
    """Proximal gradient descent optimizer for Elastic Net."""
    
    def __init__(self, model, lr=0.01, lambda_reg=0.01, alpha=0.5):
        self.model = model
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.alpha = alpha
    
    def step(self):
        """Perform one proximal gradient step."""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Gradient step (for the smooth loss part)
                    param.data -= self.lr * param.grad
                    # Proximal step (for the Elastic Net penalty)
                    param.data = proximal_elastic_net(
                        param.data, self.lambda_reg, self.alpha, self.lr
                    )
    
    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
```

## Scikit-learn Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

def elastic_net_analysis(X, y, l1_ratios=None, alphas=None):
    """
    Comprehensive Elastic Net analysis with cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        l1_ratios: L1 ratio values to try (alpha in our notation)
        alphas: Regularization strengths to try
        
    Returns:
        Best model and analysis results
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)
    
    # Cross-validation for both parameters
    elastic_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=5,
        random_state=42,
        max_iter=10000
    )
    elastic_cv.fit(X_scaled, y)
    
    # Results
    n_nonzero = np.sum(elastic_cv.coef_ != 0)
    
    print("Elastic Net CV Results:")
    print(f"  Optimal alpha (λ): {elastic_cv.alpha_:.6f}")
    print(f"  Optimal l1_ratio: {elastic_cv.l1_ratio_:.2f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(elastic_cv.coef_)}")
    print(f"  R² score: {elastic_cv.score(X_scaled, y):.4f}")
    
    return elastic_cv, scaler


def compare_regularization_methods(X, y, alpha_values):
    """Compare Lasso, Ridge, and Elastic Net."""
    from sklearn.linear_model import Lasso, Ridge
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for alpha in alpha_values:
        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        lasso_nonzero = np.sum(lasso.coef_ != 0)
        
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        ridge_nonzero = np.sum(np.abs(ridge.coef_) > 1e-6)
        
        # Elastic Net (50% mix)
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
        elastic.fit(X_scaled, y)
        elastic_nonzero = np.sum(elastic.coef_ != 0)
        
        results[alpha] = {
            'lasso_nonzero': lasso_nonzero,
            'ridge_nonzero': ridge_nonzero,
            'elastic_nonzero': elastic_nonzero,
            'lasso_score': lasso.score(X_scaled, y),
            'ridge_score': ridge.score(X_scaled, y),
            'elastic_score': elastic.score(X_scaled, y)
        }
    
    return results
```

## Hyperparameter Selection

### Two-Dimensional Search

Elastic Net has two hyperparameters: $\lambda$ (overall strength) and $\alpha$ (L1/L2 mixing):

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

def grid_search_elastic_net(X, y):
    """
    Grid search over both Elastic Net hyperparameters.
    """
    param_grid = {
        'alpha': np.logspace(-4, 1, 20),  # Overall regularization
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]  # Mixing
    }
    
    grid_search = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.cv_results_
```

### Regularization Path

```python
from sklearn.linear_model import enet_path
import matplotlib.pyplot as plt

def plot_elastic_net_path(X, y, l1_ratio=0.5, eps=1e-3):
    """
    Plot the Elastic Net regularization path.
    
    Args:
        X: Feature matrix
        y: Target vector
        l1_ratio: Fixed mixing parameter
        eps: Length of path
    """
    # Standardize
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    
    # Compute path
    alphas, coefs, _ = enet_path(
        X_centered, y_centered, 
        l1_ratio=l1_ratio,
        eps=eps
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[0]):
        plt.plot(alphas, coefs[i], label=f'Feature {i}')
    
    plt.xscale('log')
    plt.xlabel('Regularization strength (α)')
    plt.ylabel('Coefficient value')
    plt.title(f'Elastic Net Path (l1_ratio={l1_ratio})')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.gca().invert_xaxis()
    plt.legend(loc='best', fontsize=8)
    
    return alphas, coefs
```

## Theoretical Properties

### Grouping Effect

For strongly correlated features $x_i$ and $x_j$ with correlation $\rho$, Elastic Net coefficients satisfy:

$$
|w_i - w_j| \leq \frac{1}{\lambda(1-\alpha)} \|y\|_1 \sqrt{2(1 - \rho)}
$$

This means highly correlated features have similar coefficients, unlike Lasso which arbitrarily selects one.

### Unique Solution

Unlike Lasso (which may have multiple solutions), Elastic Net with $\alpha < 1$ has a unique solution due to the strictly convex L2 term.

### Oracle Property

Under certain conditions, Elastic Net achieves the oracle property—it selects the correct features with probability approaching 1 as sample size grows.

## Practical Guidelines

### Choosing the Mixing Parameter α

| Scenario | Recommended α |
|----------|---------------|
| Strong feature selection needed | 0.9 - 0.99 |
| Moderate sparsity | 0.5 - 0.7 |
| Stability with some sparsity | 0.1 - 0.3 |
| Highly correlated features | 0.1 - 0.5 |

### When to Use Elastic Net

1. **Correlated features**: When feature groups are correlated and you want group selection
2. **High dimensions**: When $p >> n$ and Lasso saturates
3. **Stability needed**: When consistent feature selection matters more than pure sparsity
4. **Uncertain L1 vs L2**: When unsure which regularization is better

### Comparison Summary

| Method | Sparsity | Stability | Correlated Features | Unique Solution |
|--------|----------|-----------|---------------------|-----------------|
| L1 (Lasso) | High | Low | Arbitrary selection | Not always |
| L2 (Ridge) | None | High | Equal weighting | Yes |
| Elastic Net | Moderate | Moderate-High | Grouped selection | Yes |

## Applications

### Feature Selection with Grouping

```python
def grouped_feature_selection(X, y, feature_groups, alpha=0.5):
    """
    Feature selection that respects feature groups.
    
    Args:
        X: Feature matrix
        y: Target
        feature_groups: Dict mapping group names to feature indices
        alpha: Elastic Net l1_ratio
        
    Returns:
        Selected feature groups and model
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Elastic Net
    model = ElasticNetCV(l1_ratio=alpha, cv=5)
    model.fit(X_scaled, y)
    
    # Analyze group selection
    selected_groups = {}
    for group_name, indices in feature_groups.items():
        group_coefs = model.coef_[indices]
        n_selected = np.sum(group_coefs != 0)
        n_total = len(indices)
        selected_groups[group_name] = {
            'selected': n_selected,
            'total': n_total,
            'ratio': n_selected / n_total,
            'mean_coef': np.mean(np.abs(group_coefs))
        }
    
    return selected_groups, model
```

## References

1. Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity*. CRC Press.
3. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. *Journal of Statistical Software*, 33(1), 1-22.
