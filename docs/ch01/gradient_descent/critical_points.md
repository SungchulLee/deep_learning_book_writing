# Local Minima, Saddle Points, and Plateaus

## Introduction

When gradient descent converges, where does it end up? In convex optimization, the answer is simple: the global minimum. But for non-convex functions like neural network losses, the landscape is far more complex, featuring **local minima**, **saddle points**, and **plateaus**. Understanding these critical points is essential for diagnosing optimization problems and designing effective training strategies.

## Critical Points: Definition

A **critical point** (or stationary point) is any point $\mathbf{x}^*$ where the gradient vanishes:

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

At critical points, gradient descent stops making progress (the update is zero). The nature of the critical point determines whether we've found a good solution or are stuck.

## Types of Critical Points

### Classification via the Hessian

The **Hessian matrix** $\mathbf{H} = \nabla^2 f$ (matrix of second derivatives) classifies critical points:

| Hessian Property | Critical Point Type |
|-----------------|---------------------|
| All eigenvalues > 0 (positive definite) | **Local minimum** |
| All eigenvalues < 0 (negative definite) | **Local maximum** |
| Mixed positive and negative eigenvalues | **Saddle point** |
| Some eigenvalues = 0 | **Degenerate** (further analysis needed) |

## Local Minima

### Definition

A point $\mathbf{x}^*$ is a **local minimum** if:

$$f(\mathbf{x}^*) \leq f(\mathbf{x}) \quad \text{for all } \mathbf{x} \text{ in a neighborhood of } \mathbf{x}^*$$

### In Deep Learning

Research has shown that:

1. **Most local minima are good**: They achieve similar loss to the global minimum
2. **Bad local minima are rare**: Especially in overparameterized networks
3. **Local minima connect**: Good solutions form connected valleys

```python
# Experiment: Multiple runs often find different but equally good minima
import torch
import torch.nn as nn

def train_network(seed):
    torch.manual_seed(seed)
    
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    X = torch.randn(100, 10)
    y = torch.sin(X.sum(dim=1, keepdim=True))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(500):
        loss = nn.MSELoss()(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()

# Run with different seeds
losses = [train_network(seed) for seed in range(20)]
print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
```

## Saddle Points

### Definition

A **saddle point** is a critical point that is neither a maximum nor minimum. The Hessian has both positive and negative eigenvalues.

### Mathematical Example

For $f(x, y) = x^2 - y^2$:

- Gradient: $\nabla f = (2x, -2y)$
- Critical point: $(0, 0)$
- Hessian: $\mathbf{H} = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}$
- Eigenvalues: $+2$ and $-2$ → **Saddle point**

### Why Saddle Points Matter

In high dimensions, saddle points are **far more common** than local minima:

- For $n$ dimensions, a critical point has $n$ eigenvalues
- Each eigenvalue is positive or negative with ~50% probability
- Probability of local minimum (all positive): $(1/2)^n$ → tiny for large $n$

**Example**: In 1000 dimensions:
- Probability all eigenvalues positive: $2^{-1000} \approx 10^{-301}$
- Most critical points are saddle points!

### Gradient Descent Behavior at Saddle Points

Near saddle points, gradient descent experiences:

1. **Slow progress**: Gradients are small near the saddle
2. **Escape eventually**: Noise in SGD helps escape
3. **Direction matters**: Must find the "escape direction"

## Plateaus

### Definition

A **plateau** is a flat region where:

- Gradients are very small (but not exactly zero)
- Loss changes very slowly
- Optimization appears to stall

### Causes of Plateaus

1. **Activation saturation**: Sigmoid/tanh saturate at extreme values
   ```python
   x = torch.tensor(10.0, requires_grad=True)
   y = torch.sigmoid(x)  # approximately 1.0
   y.backward()
   print(x.grad)  # approximately 0.00005 (nearly zero!)
   ```

2. **Poor initialization**: Weights too large or small
3. **Vanishing gradients**: Deep networks without proper architecture
4. **Learning rate too small**: Progress is imperceptible

### Escaping Plateaus

**Strategies**:

1. **Increase learning rate**: Temporarily boost to escape
2. **Use momentum**: Accumulate velocity to traverse flat regions
3. **Learning rate warmup**: Gradual increase helps exploration
4. **Different initialization**: Try again with new random weights
5. **Batch normalization**: Prevents activation saturation
6. **ReLU instead of sigmoid**: Avoids saturation

## Practical Strategies

### Momentum Helps

Momentum accumulates velocity, helping traverse plateaus and escape saddle points:

```python
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9  # Momentum coefficient
)
```

### Adam's Advantages

Adam handles critical points well:

- Adaptive learning rates per parameter
- Momentum-like behavior
- Works well on plateaus

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Gradient Noise Helps

The noise in SGD helps escape saddle points by providing random perturbations that can push optimization away from unstable equilibria.

### Monitoring Critical Points

```python
def analyze_critical_point(model, loss_fn, X, y, epsilon=1e-5):
    """
    Analyze if current position is near a critical point
    """
    loss = loss_fn(model(X), y)
    loss.backward()
    
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    
    if grad_norm < epsilon:
        print(f"Near critical point (grad norm: {grad_norm:.2e})")
    elif grad_norm < 0.01:
        print(f"Possibly in plateau region (grad norm: {grad_norm:.4f})")
    else:
        print(f"Normal optimization (grad norm: {grad_norm:.4f})")
    
    return grad_norm
```

## Key Takeaways

1. **Critical points have zero gradient**: Gradient descent stops at these points

2. **Three types**:
   - Local minima: Good for optimization
   - Saddle points: Common in high dimensions, can slow training
   - Plateaus: Flat regions with near-zero gradients

3. **High dimensions favor saddle points**: Local minima become rare

4. **Good news for deep learning**:
   - Most local minima achieve similar loss
   - SGD noise helps escape saddle points
   - Proper architecture and initialization avoid plateaus

5. **Practical strategies**:
   - Use momentum and adaptive optimizers
   - Monitor gradient norms
   - Try different initializations
   - Avoid saturating activations

## Connections to Other Topics

- **Convexity**: See [Convex vs Non-Convex](convex_nonconvex.md)
- **Momentum**: Helps escape saddle points, see [Classical Momentum](../../ch05/optimizers/momentum.md)
- **Batch Normalization**: Prevents saturation, see [Batch Normalization](ch04/normalization/batch_norm.md)

## Exercises

1. **Classify critical points**: For $f(x, y) = x^3 - 3xy^2$, find all critical points, compute the Hessian at each, and classify each as min, max, or saddle.

2. **High-dimensional saddles**: Generate random symmetric matrices of size $n \times n$ for $n = 10, 100, 1000$. What fraction have all positive eigenvalues?

3. **Escape experiment**: Initialize gradient descent at $(0.001, 0.001)$ for $f(x,y) = x^2 - y^2$. Compare vanilla GD, GD with noise, and GD with momentum.

4. **Plateau detection**: Implement a training loop that automatically detects plateaus and increases the learning rate when stuck.

## References

- Dauphin, Y., et al. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. NeurIPS.
- Choromanska, A., et al. (2015). The loss surfaces of multilayer networks. AISTATS.
- Lee, J. D., et al. (2016). Gradient descent only converges to minimizers. COLT.
