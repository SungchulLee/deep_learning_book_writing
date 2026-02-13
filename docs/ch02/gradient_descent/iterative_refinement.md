# Optimization as Iterative Refinement

## Introduction

At its core, gradient descent embodies a fundamental principle: **complex problems can be solved through repeated small improvements**. Rather than finding the perfect solution in one step, we start with an initial guess and iteratively refine it until we reach a satisfactory answer.

This chapter establishes the conceptual foundation for understanding gradient descent as an iterative optimization algorithm—the workhorse behind modern machine learning and deep learning.

## The Optimization Problem

### What Are We Trying to Solve?

In machine learning, we frequently encounter **optimization problems** of the form:

$$\theta^* = \arg\min_{\theta} L(\theta)$$

where:

- $\theta$ represents the **parameters** of our model (weights, biases)
- $L(\theta)$ is the **loss function** (also called cost function or objective function)
- $\theta^*$ is the **optimal parameter** that minimizes the loss

The loss function quantifies "how wrong" our model's predictions are. Our goal is to find parameters that make predictions as accurate as possible—equivalently, that minimize the loss.

### Why Iterative Methods?

For simple problems like linear regression with few features, we can sometimes find **closed-form solutions** (e.g., the Normal Equations). However, most real-world problems lack such analytical solutions because:

1. **Non-linearity**: Neural networks introduce non-linear activation functions
2. **High dimensionality**: Modern models have millions or billions of parameters
3. **Complex loss landscapes**: The relationship between parameters and loss is intricate

Iterative methods provide a practical alternative: start somewhere, and keep improving.

## The Iterative Refinement Framework

### Core Algorithm Structure

Every iterative optimization algorithm follows this basic structure:

```
1. Initialize parameters θ₀ (often randomly)
2. Repeat until convergence:
   a. Evaluate current quality: compute L(θₜ)
   b. Determine update direction: compute Δθ
   c. Update parameters: θₜ₊₁ = θₜ + Δθ
3. Return final parameters θ*
```

The key question that distinguishes different algorithms is: **How do we determine the update direction Δθ?**

### Gradient Descent's Answer

Gradient descent provides an elegant answer: **move in the direction that decreases the loss most rapidly**. This direction is the negative of the gradient:

$$\Delta\theta = -\eta \nabla_\theta L(\theta)$$

where:

- $\nabla_\theta L(\theta)$ is the **gradient** (vector of partial derivatives)
- $\eta > 0$ is the **learning rate** (step size)
- The negative sign ensures we move toward lower loss values

## Mathematical Formulation

### The Gradient Descent Update Rule

The fundamental update equation is:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

This can be understood component-wise. For each parameter $\theta_j$:

$$\theta_{j,t+1} = \theta_{j,t} - \eta \frac{\partial L}{\partial \theta_j}\bigg|_{\theta_t}$$

### Intuition: Rolling Downhill

Imagine standing on a hilly landscape where your altitude represents the loss value. The gradient tells you which direction is "uphill" (steepest ascent). By moving in the opposite direction, you descend toward a valley (minimum).

The learning rate determines your step size:

- **Small learning rate**: Careful, small steps—slow but steady
- **Large learning rate**: Bold, large steps—fast but potentially overshooting

## A Concrete Example

### Problem Setup

Consider fitting a simple linear model $y = wx$ to data where the true relationship is $y = 2x$.

**Training data:**
```python
X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
```

**Loss function (Mean Squared Error):**

$$L(w) = \frac{1}{N}\sum_{i=1}^{N}(wx_i - y_i)^2$$

**Gradient:**

$$\frac{dL}{dw} = \frac{2}{N}\sum_{i=1}^{N}x_i(wx_i - y_i)$$

### Iteration Trace

Starting with $w_0 = 0$ and learning rate $\eta = 0.01$:

| Iteration | Weight $w$ | Loss $L(w)$ | Gradient $\frac{dL}{dw}$ |
|-----------|------------|-------------|--------------------------|
| 0         | 0.000      | 44.000      | -44.000                  |
| 1         | 0.440      | 26.854      | -33.880                  |
| 2         | 0.779      | 16.402      | -26.100                  |
| ...       | ...        | ...         | ...                      |
| 20        | 1.997      | 0.000       | -0.018                   |

The weight converges toward the optimal value $w^* = 2$.

### PyTorch Implementation

```python
import torch

# Data
X = torch.tensor([1., 2., 3., 4., 5.])
y = torch.tensor([2., 4., 6., 8., 10.])

# Initialize parameter with gradient tracking
w = torch.tensor(0.0, requires_grad=True)

# Hyperparameters
learning_rate = 0.01
n_iterations = 100

# Gradient descent loop
for t in range(n_iterations):
    # Forward pass: compute predictions
    y_pred = w * X
    
    # Compute loss
    loss = torch.mean((y_pred - y) ** 2)
    
    # Backward pass: compute gradient
    loss.backward()
    
    # Update parameter (with no_grad to prevent tracking)
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # Zero gradient for next iteration
    w.grad.zero_()

print(f"Learned weight: {w.item():.4f}")  # ≈ 2.0
```

## Convergence Properties

### When Does Gradient Descent Converge?

Gradient descent is guaranteed to converge to a **global minimum** when:

1. The loss function is **convex**
2. The learning rate is **sufficiently small**
3. The gradient is **Lipschitz continuous**

For non-convex functions (like neural network losses), convergence to a **local minimum** is typical.

### Convergence Rate

The rate at which gradient descent converges depends on:

- **Condition number**: Ratio of largest to smallest eigenvalues of the Hessian
- **Learning rate**: Must be tuned appropriately
- **Loss landscape geometry**: Steep valleys slow convergence

For convex functions with Lipschitz gradients:

$$L(\theta_t) - L(\theta^*) \leq \frac{\|\theta_0 - \theta^*\|^2}{2\eta t}$$

This shows **sublinear convergence**: the error decreases as $O(1/t)$.

## Stopping Criteria

### When to Stop Iterating?

Common stopping criteria include:

1. **Maximum iterations**: Stop after $T$ iterations
   ```python
   if t >= max_iterations:
       break
   ```

2. **Loss threshold**: Stop when loss is small enough
   ```python
   if loss < epsilon:
       break
   ```

3. **Gradient norm**: Stop when gradients are near zero
   ```python
   if torch.norm(gradient) < epsilon:
       break
   ```

4. **Relative improvement**: Stop when progress stalls
   ```python
   if abs(loss_prev - loss) / loss_prev < epsilon:
       break
   ```

### Practical Recommendations

- Use a combination of criteria (e.g., max iterations AND gradient threshold)
- Monitor validation loss to detect overfitting
- Implement early stopping based on validation performance

## Visualizing Iterative Refinement

### Loss Curve

The loss should decrease over iterations, typically showing:

- **Rapid initial decrease**: Large gradients drive fast improvement
- **Gradual slowdown**: Smaller gradients near the minimum
- **Plateau**: Convergence when gradient approaches zero

```
Loss
  │
  │╲
  │ ╲
  │  ╲__
  │     ╲___________
  └────────────────────→ Iteration
```

### Parameter Trajectory

In 2D parameter space, the optimization path reveals:

- Direct descent toward minimum (for well-conditioned problems)
- Zigzag patterns (for ill-conditioned problems)
- Spiraling (when momentum is added)

## Key Takeaways

1. **Gradient descent is iterative**: Start with a guess, repeatedly improve
2. **Gradients guide updates**: Move opposite to the gradient direction
3. **Learning rate controls step size**: Balance speed vs. stability
4. **Convergence is guaranteed** for convex problems with proper learning rate
5. **Stopping criteria** prevent infinite loops and overfitting

## Connections to Other Topics

- **Learning Rate**: Detailed in [Learning Rate and Step Size](learning_rate.md)
- **Gradient Computation**: See [Autograd Fundamentals](../autograd/gradient_computation.md)
- **Convexity**: Explored in [Convex vs Non-Convex Optimization](convex_nonconvex.md)
- **Advanced Optimizers**: Build on this foundation in [Optimizers](../../ch05/optimizers/optimizer_overview.md)

## Exercises

1. **Manual computation**: Trace 5 iterations of gradient descent for $L(w) = (w-3)^2$ starting from $w_0 = 0$ with $\eta = 0.1$.

2. **Convergence analysis**: Plot the loss curve for different learning rates (0.01, 0.1, 0.5). What happens when the learning rate is too large?

3. **Stopping criteria**: Implement gradient descent with all four stopping criteria. Compare their behavior on a simple quadratic loss.

4. **2D visualization**: For $L(w, b) = (w-2)^2 + (b-3)^2$, plot the optimization trajectory in parameter space.

## References

- Cauchy, A. (1847). Méthode générale pour la résolution des systèmes d'équations simultanées.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 4.
