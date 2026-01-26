# Convex vs Non-Convex Optimization

## Introduction

The distinction between **convex** and **non-convex** optimization is fundamental to understanding when gradient descent will succeed and what challenges we might face. This chapter explores these concepts, their implications for machine learning, and why non-convex optimization—despite its theoretical challenges—works remarkably well for deep learning.

## Convexity: Definitions

### Convex Sets

A set $\mathcal{C} \subseteq \mathbb{R}^n$ is **convex** if for any two points $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ and any $\lambda \in [0, 1]$:

$$\lambda \mathbf{x} + (1-\lambda)\mathbf{y} \in \mathcal{C}$$

**Intuition**: A line segment connecting any two points in the set lies entirely within the set.

```
Convex sets:                 Non-convex sets:
    _____                        ╱╲
   ╱     ╲                      ╱  ╲___
  │       │                    │      ╲
  │   •───•                    │   •   │
  │       │                     ╲__│___╱
   ╲_____╱                         │
                                   •
```

### Convex Functions

A function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is **convex** if for any $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$:

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

**Intuition**: A chord connecting any two points on the graph lies above (or on) the graph.

```
Convex function:              Non-convex function:
       │                             │
       │    ╱                        │   ╱╲
       │   ╱                         │  ╱  ╲
       │  ╱                          │ ╱    ╲__╱╲
       │_╱                           │╱          ╲
       └────────                     └────────────
```

### Strictly Convex Functions

A function is **strictly convex** if the inequality is strict for $\mathbf{x} \neq \mathbf{y}$ and $\lambda \in (0, 1)$:

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) < \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

### Equivalent Conditions

For twice-differentiable functions, convexity can be verified by:

1. **First-order condition** (tangent lies below function):
   $$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$$

2. **Second-order condition** (Hessian is positive semi-definite):
   $$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}$$

## Examples of Convex Functions

### Common Convex Functions

| Function | Formula | Domain |
|----------|---------|--------|
| Linear | $f(\mathbf{x}) = \mathbf{a}^T\mathbf{x} + b$ | $\mathbb{R}^n$ |
| Quadratic (PSD) | $f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}$ ($\mathbf{A} \succeq 0$) | $\mathbb{R}^n$ |
| Exponential | $f(x) = e^{ax}$ | $\mathbb{R}$ |
| Negative log | $f(x) = -\log(x)$ | $\mathbb{R}_{++}$ |
| Powers | $f(x) = x^p$ ($p \geq 1$ or $p \leq 0$) | $\mathbb{R}_+$ |
| Norms | $f(\mathbf{x}) = \|\mathbf{x}\|_p$ | $\mathbb{R}^n$ |
| Log-sum-exp | $f(\mathbf{x}) = \log(\sum e^{x_i})$ | $\mathbb{R}^n$ |

### Machine Learning Examples

**MSE Loss (Linear Regression)**:

$$L(\mathbf{w}) = \frac{1}{N}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

This is convex in $\mathbf{w}$ (quadratic with PSD Hessian).

**Logistic Regression Loss**:

$$L(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N \log(1 + e^{-y_i \mathbf{w}^T\mathbf{x}_i})$$

This is convex in $\mathbf{w}$.

## Properties of Convex Optimization

### Global Optimality

!!! theorem "Fundamental Property of Convex Functions"
    For a convex function $f$:
    
    - Every **local minimum** is a **global minimum**
    - If $f$ is strictly convex, the global minimum is **unique** (if it exists)

This is why convex optimization is considered "easy"—we don't need to worry about getting stuck in suboptimal local minima.

### Gradient Descent Convergence

For convex functions with Lipschitz-continuous gradients:

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2\eta t}$$

This guarantees convergence at rate $O(1/t)$.

For **strongly convex** functions (eigenvalues of Hessian bounded below):

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \left(1 - \frac{\mu}{L}\right)^t (f(\mathbf{x}_0) - f(\mathbf{x}^*))$$

This gives **linear convergence** (exponentially fast).

## Non-Convex Optimization

### What Makes a Function Non-Convex?

A function is non-convex if the convexity condition fails anywhere. Common causes:

1. **Multiple local minima**: The function has valleys at different heights
2. **Saddle points**: Regions that are convex in some directions, concave in others
3. **Non-convex constraints**: The feasible region is non-convex

### Neural Network Loss Functions

Neural network loss functions are **highly non-convex** due to:

1. **Non-linear activations**: ReLU, sigmoid, tanh
2. **Weight symmetries**: Permuting neurons doesn't change function
3. **Scaling ambiguities**: $w_1 \cdot a \cdot w_2 = (cw_1) \cdot a \cdot (w_2/c)$

```python
# Simple non-convex loss landscape
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def neural_loss(w1, w2, x=1, y=2):
    """Loss for y = w2 * ReLU(w1 * x)"""
    hidden = np.maximum(0, w1 * x)  # ReLU
    pred = w2 * hidden
    return (pred - y) ** 2

w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1, w2)
Z = np.vectorize(neural_loss)(W1, W2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.set_title('Non-Convex Neural Network Loss Surface')
plt.show()
```

### Challenges in Non-Convex Optimization

1. **No global optimality guarantee**: Gradient descent finds local minima, not necessarily global
2. **Saddle points**: Can slow down optimization significantly
3. **Initialization sensitivity**: Different starting points lead to different solutions
4. **Plateau regions**: Flat areas where gradients are near zero

## Why Does Deep Learning Work?

Despite non-convexity, deep learning succeeds remarkably well. Several factors contribute:

### 1. Loss Landscape Structure

Research has revealed that neural network loss landscapes have favorable properties:

- **Many equivalent minima**: Due to symmetries, many local minima achieve similar loss
- **Connected valleys**: Good solutions form connected regions
- **High-dimensional benefits**: Saddle points dominate over local minima in high dimensions

### 2. Overparameterization

When networks have many more parameters than data points:

- Interpolation becomes easier (many solutions exist)
- Optimization paths remain "wide"
- Implicit regularization selects good solutions

### 3. SGD as Regularizer

Stochastic gradient descent helps by:

- Adding noise that helps escape sharp minima
- Finding flat minima that generalize better
- Providing implicit regularization

### 4. Good Initialization

Proper initialization (Xavier, He, etc.) places us in favorable regions:

```python
# He initialization for ReLU networks
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Xavier initialization for tanh networks
nn.init.xavier_normal_(layer.weight)
```

## Visualizing Convexity

### 1D Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 4, 200)

# Convex: quadratic
convex = (x - 1) ** 2

# Non-convex: multiple minima
non_convex = x**4 - 4*x**2 + x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, convex, 'b-', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.scatter([1], [0], color='red', s=100, zorder=5, label='Global min')
ax1.set_title('Convex: One Global Minimum')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x, non_convex, 'b-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
# Mark local minima
ax2.scatter([-1.35, 1.46], [non_convex[np.argmin(np.abs(x+1.35))], 
                            non_convex[np.argmin(np.abs(x-1.46))]], 
            color='red', s=100, zorder=5, label='Local minima')
ax2.set_title('Non-Convex: Multiple Local Minima')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2D Contour Comparison

```python
# Convex: ellipsoid
def convex_2d(x, y):
    return x**2 + 2*y**2

# Non-convex: Rastrigin function
def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Convex
Z1 = convex_2d(X, Y)
ax1.contour(X, Y, Z1, levels=20)
ax1.set_title('Convex Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Non-convex
Z2 = rastrigin(X, Y)
ax2.contour(X, Y, Z2, levels=30)
ax2.set_title('Non-Convex Function (Rastrigin)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()
```

## Checking Convexity

### Hessian Analysis

For twice-differentiable functions:

```python
import torch

def check_convexity_numerical(f, x, epsilon=1e-3):
    """
    Check local convexity via Hessian eigenvalues
    """
    n = len(x)
    H = torch.zeros((n, n))
    
    x = x.detach().requires_grad_(True)
    
    # Compute Hessian
    for i in range(n):
        # Gradient
        y = f(x)
        grad = torch.autograd.grad(y, x, create_graph=True)[0]
        
        # Second derivatives
        for j in range(n):
            H[i, j] = torch.autograd.grad(
                grad[i], x, retain_graph=True
            )[0][j]
    
    # Check eigenvalues
    eigenvalues = torch.linalg.eigvalsh(H)
    
    if torch.all(eigenvalues >= -epsilon):
        return "Locally convex (H ≽ 0)"
    elif torch.all(eigenvalues <= epsilon):
        return "Locally concave (H ≼ 0)"
    else:
        return "Non-convex (indefinite H)"
```

### Operations Preserving Convexity

Convexity is preserved under:

1. **Non-negative weighted sums**: $\sum \alpha_i f_i$ where $\alpha_i \geq 0$
2. **Composition with affine**: $f(\mathbf{Ax} + \mathbf{b})$
3. **Pointwise maximum**: $\max(f_1, f_2, \ldots, f_n)$
4. **Partial minimization**: $g(\mathbf{x}) = \min_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$

## Practical Implications

### For Convex Problems

- **Guaranteed convergence** to global minimum
- **Less sensitive** to initialization
- Use **standard optimizers**: SGD, Adam work well
- **Closed-form solutions** may exist (e.g., linear regression)

### For Non-Convex Problems (Deep Learning)

- **Multiple restarts**: Try different initializations
- **Learning rate warmup**: Careful initial exploration
- **Regularization**: Helps find generalizable minima
- **Monitor training**: Watch for divergence, oscillation
- **Use adaptive optimizers**: Adam often more robust than SGD

### Code: Comparing Optimization on Both

```python
import torch
import torch.nn as nn

# Convex: Linear regression
def train_convex():
    X = torch.randn(100, 10)
    y = X @ torch.randn(10, 1) + 0.1 * torch.randn(100, 1)
    
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        loss = nn.MSELoss()(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()

# Non-convex: Neural network
def train_nonconvex():
    X = torch.randn(100, 10)
    y = torch.sin(X.sum(dim=1, keepdim=True))
    
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        loss = nn.MSELoss()(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()

# Run multiple times to see variance
convex_results = [train_convex() for _ in range(10)]
nonconvex_results = [train_nonconvex() for _ in range(10)]

print(f"Convex: mean={np.mean(convex_results):.4f}, std={np.std(convex_results):.4f}")
print(f"Non-convex: mean={np.mean(nonconvex_results):.4f}, std={np.std(nonconvex_results):.4f}")
```

## Key Takeaways

1. **Convex = Easy**: Local minima are global minima
2. **Neural networks are non-convex**: Multiple local minima, saddle points
3. **Non-convex works in practice**: Good landscape structure, overparameterization, SGD noise
4. **Initialization matters**: Especially for non-convex problems
5. **Optimization ≠ Generalization**: Finding the global minimum isn't always the goal
6. **Regularization helps**: Guides optimization toward generalizable solutions

## Connections to Other Topics

- **Critical Points**: See [Local Minima, Saddle Points, Plateaus](critical_points.md)
- **Initialization**: Explored in [Xavier Initialization](../../ch02/feedforward/depth_width.md)
- **Regularization**: See [L2 Regularization](../../ch02/regularization/l2_ridge.md)
- **Adaptive Optimizers**: Help with non-convex, see [Adam](../../ch02/optimizers/adam.md)

## Exercises

1. **Verify convexity**: Show that the following are convex:
   - $f(x) = |x|$
   - $f(x) = e^x$
   - $f(\mathbf{x}) = \max_i x_i$

2. **Hessian analysis**: For $f(x, y) = x^2 - y^2$:
   - Compute the Hessian
   - Classify the function (convex, concave, neither)
   - Identify any critical points

3. **Optimization experiment**: Create a non-convex 2D function with multiple local minima. Run gradient descent from 20 different random starting points. How many different local minima do you find?

4. **Neural network analysis**: Train a small neural network 50 times with different random seeds. Plot the histogram of final losses. What does this tell you about the loss landscape?

## References

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Choromanska, A., et al. (2015). The loss surfaces of multilayer networks. AISTATS.
- Li, H., et al. (2018). Visualizing the loss landscape of neural nets. NeurIPS.
- Fort, S., & Ganguli, S. (2019). Emergent properties of the local geometry of neural loss landscapes.
