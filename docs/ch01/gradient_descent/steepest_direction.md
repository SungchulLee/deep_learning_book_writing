# Gradient as Steepest Ascent Direction

## Introduction

The gradient is the cornerstone concept that gives "gradient descent" its name. Understanding why the gradient points in the direction of steepest ascent—and conversely, why the negative gradient points toward steepest descent—is essential for developing intuition about optimization algorithms.

This chapter provides both the mathematical derivation and geometric intuition behind this fundamental property.

## The Gradient: Definition and Notation

### Definition

For a scalar-valued function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, the **gradient** at a point $\mathbf{x}$ is the vector of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Notation Variants

You may encounter different notations for the gradient:

| Notation | Meaning |
|----------|---------|
| $\nabla f$ | Gradient of $f$ (nabla notation) |
| $\nabla_\theta L$ | Gradient of $L$ with respect to $\theta$ |
| $\frac{\partial L}{\partial \theta}$ | Partial derivative notation |
| $\text{grad } f$ | Alternative gradient notation |
| $f'(\mathbf{x})$ | Derivative notation (1D) |

### Example: Two-Variable Function

For $f(x, y) = x^2 + 3y^2$:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x \\ 6y \end{bmatrix}$$

At point $(1, 2)$:

$$\nabla f(1, 2) = \begin{bmatrix} 2 \\ 12 \end{bmatrix}$$

## Why Steepest Ascent?

### Directional Derivatives

The **directional derivative** measures how fast $f$ changes as we move in direction $\mathbf{u}$ (a unit vector):

$$D_\mathbf{u} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}$$

This can be computed as the **dot product** of the gradient and the direction:

$$D_\mathbf{u} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \|\nabla f\| \|\mathbf{u}\| \cos\theta$$

where $\theta$ is the angle between $\nabla f$ and $\mathbf{u}$.

### Maximizing the Directional Derivative

**Question**: In which direction $\mathbf{u}$ does $f$ increase most rapidly?

**Answer**: We want to maximize $D_\mathbf{u} f = \|\nabla f\| \cos\theta$.

Since $\|\nabla f\|$ is fixed and $\|\mathbf{u}\| = 1$:

- Maximum occurs when $\cos\theta = 1$ (i.e., $\theta = 0$)
- This means $\mathbf{u}$ is parallel to $\nabla f$

**Conclusion**: The gradient $\nabla f$ points in the direction of **steepest ascent**.

### Formal Theorem

!!! theorem "Gradient as Steepest Ascent"
    Let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be differentiable at $\mathbf{x}$ with $\nabla f(\mathbf{x}) \neq \mathbf{0}$. Then:
    
    1. The direction of maximum increase is $\mathbf{u}^* = \frac{\nabla f}{\|\nabla f\|}$
    2. The maximum rate of increase is $\|\nabla f(\mathbf{x})\|$
    3. The direction of maximum decrease is $-\mathbf{u}^*$

### Proof Sketch

For any unit vector $\mathbf{u}$:

$$D_\mathbf{u} f = \nabla f \cdot \mathbf{u} \leq \|\nabla f\| \cdot \|\mathbf{u}\| = \|\nabla f\|$$

by the Cauchy-Schwarz inequality. Equality holds when $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$.

## Geometric Interpretation

### Level Sets and Gradients

A **level set** (or contour) of $f$ is the set of points where $f$ has the same value:

$$\{x : f(\mathbf{x}) = c\}$$

**Key insight**: The gradient is **perpendicular** (orthogonal) to level sets.

```
                    ↑ ∇f
                    │
    ────────────────┼────────────── f = c + ε (higher)
                    │
    ────────────────●────────────── f = c (level set)
                    │
    ────────────────│────────────── f = c - ε (lower)
```

The gradient always points "uphill," perpendicular to the contour lines.

### Contour Plot Visualization

For $f(x, y) = x^2 + y^2$ (a paraboloid):

- Contours are concentric circles
- Gradient at any point points radially outward from origin
- Following negative gradient leads to the minimum at $(0, 0)$

```python
import numpy as np
import matplotlib.pyplot as plt

# Create grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Plot contours
plt.contour(X, Y, Z, levels=15)

# Plot gradient vectors at selected points
points = [(-2, 1), (1, 2), (2, -1)]
for px, py in points:
    grad = np.array([2*px, 2*py])
    plt.arrow(px, py, 0.3*grad[0], 0.3*grad[1], 
              head_width=0.15, color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradients Perpendicular to Contours')
plt.axis('equal')
plt.show()
```

## From Steepest Ascent to Gradient Descent

### The Descent Direction

Since $\nabla f$ points toward steepest ascent, **$-\nabla f$ points toward steepest descent**.

To minimize $f$, we should move in the direction of $-\nabla f$:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

This is the **gradient descent update rule**.

### Why Descent Reduces the Function Value

**First-order Taylor approximation**:

$$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta\mathbf{x}$$

For $\Delta\mathbf{x} = -\eta \nabla f$:

$$f(\mathbf{x} - \eta\nabla f) \approx f(\mathbf{x}) - \eta \|\nabla f\|^2$$

Since $\|\nabla f\|^2 \geq 0$, this shows that (for small $\eta$):

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t)$$

The function value **decreases** with each gradient descent step.

## Computing Gradients

### Manual Derivation

For loss function $L(w) = \frac{1}{N}\sum_{i=1}^N (wx_i - y_i)^2$:

**Step 1**: Expand
$$L(w) = \frac{1}{N}\sum_{i=1}^N (w^2x_i^2 - 2wx_iy_i + y_i^2)$$

**Step 2**: Differentiate term by term
$$\frac{dL}{dw} = \frac{1}{N}\sum_{i=1}^N (2wx_i^2 - 2x_iy_i) = \frac{2}{N}\sum_{i=1}^N x_i(wx_i - y_i)$$

### Automatic Differentiation

PyTorch computes gradients automatically:

```python
import torch

# Define variables with gradient tracking
x = torch.tensor([1., 2., 3., 4., 5.])
y = torch.tensor([2., 4., 6., 8., 10.])
w = torch.tensor(0.5, requires_grad=True)

# Forward pass
y_pred = w * x
loss = torch.mean((y_pred - y) ** 2)

# Backward pass - computes gradient
loss.backward()

print(f"Gradient dL/dw = {w.grad.item():.4f}")
```

### Chain Rule for Deep Networks

For composite functions $L = L_3 \circ L_2 \circ L_1$, the chain rule gives:

$$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial L_3} \cdot \frac{\partial L_3}{\partial L_2} \cdot \frac{\partial L_2}{\partial L_1} \cdot \frac{\partial L_1}{\partial \theta}$$

This is the foundation of **backpropagation**.

## Properties of the Gradient

### At Critical Points

When $\nabla f(\mathbf{x}) = \mathbf{0}$, the point $\mathbf{x}$ is a **critical point** (or stationary point). This could be:

- **Local minimum**: All eigenvalues of Hessian positive
- **Local maximum**: All eigenvalues of Hessian negative
- **Saddle point**: Mixed eigenvalues

### Gradient Magnitude

The magnitude $\|\nabla f\|$ indicates how steep the landscape is:

- **Large gradient**: Steep slope, far from critical point
- **Small gradient**: Flat region, near critical point
- **Zero gradient**: At a critical point

### Gradient Direction Changes

As optimization progresses:

- Early iterations: Large, consistent gradients
- Middle iterations: Gradients may change direction
- Near convergence: Gradients become small and may oscillate

## Practical Implications

### Why Normalize Gradients?

Sometimes we use **gradient direction** without magnitude:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \frac{\nabla f}{\|\nabla f\|}$$

Benefits:

- Constant step size regardless of gradient magnitude
- More stable in flat regions

Drawbacks:

- Loses information about landscape curvature
- May overshoot near minima

### Gradient Clipping

To prevent exploding gradients:

```python
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

This limits the gradient magnitude while preserving direction.

## Common Misconceptions

### Misconception 1: "Gradient descent always finds the global minimum"

**Reality**: Only guaranteed for convex functions. For non-convex functions (like neural networks), it finds local minima.

### Misconception 2: "Larger gradients mean faster convergence"

**Reality**: Very large gradients can cause overshooting and divergence. Moderate gradients with appropriate learning rates work best.

### Misconception 3: "The gradient points directly toward the minimum"

**Reality**: The gradient points toward steepest descent **locally**. The path to the minimum may be curved.

## Key Takeaways

1. **Gradient definition**: Vector of partial derivatives
2. **Steepest ascent**: Gradient points in direction of maximum increase
3. **Steepest descent**: Negative gradient minimizes function locally
4. **Perpendicular to contours**: Gradient is orthogonal to level sets
5. **Magnitude matters**: Indicates steepness; zero at critical points
6. **Automatic differentiation**: PyTorch computes gradients efficiently

## Connections to Other Topics

- **Computational Graphs**: See [Computational Graphs](../gradients/computational_graphs.md)
- **Vector-Jacobian Products**: Explained in [VJPs](../gradients/vector_jacobian_products.md)
- **Critical Points**: Detailed in [Local Minima, Saddle Points, Plateaus](critical_points.md)
- **Momentum**: Modifies steepest descent in [Classical Momentum](../../ch02/optimizers/classical_momentum.md)

## Exercises

1. **Compute gradients**: Find $\nabla f$ for:
   - $f(x, y) = x^2y + y^3$
   - $f(x, y, z) = e^{xy} + \sin(z)$
   - $f(\mathbf{w}) = \|\mathbf{Xw} - \mathbf{y}\|^2$

2. **Verify orthogonality**: Show that $\nabla f$ is perpendicular to the level curve $f(x,y) = c$ for $f(x,y) = x^2 + 4y^2$ at point $(2, 1)$.

3. **Directional derivatives**: For $f(x,y) = x^2 - xy + y^2$ at $(1, 1)$:
   - Compute the gradient
   - Find directional derivative in direction $(1, 0)$
   - Find the direction of steepest ascent

4. **PyTorch gradients**: Compute the gradient of $L = (w_1x_1 + w_2x_2 - y)^2$ with respect to $w_1$ and $w_2$ using autograd.

## References

- Stewart, J. (2015). *Calculus: Early Transcendentals*, Chapter 14.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chapter 9.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv:1609.04747.
