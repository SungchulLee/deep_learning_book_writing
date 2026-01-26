# Gradient Descent Solution for Linear Regression

## Overview

While linear regression has a closed-form solution via the Normal Equations, gradient descent offers a scalable, iterative alternative that forms the foundation for training neural networks. Understanding gradient descent for linear regression builds essential intuition for deep learning optimization.

## The Optimization Problem

### Objective Function

We minimize the Mean Squared Error (MSE):

$$J(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i - b)^2$$

In matrix form:

$$J(\boldsymbol{\theta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2$$

### Gradient Computation

The gradient with respect to parameters:

$$\nabla_{\boldsymbol{\theta}} J = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) = \frac{2}{n}\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$$

Component-wise:

$$\frac{\partial J}{\partial w_j} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)x_{ij}$$

$$\frac{\partial J}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

## Gradient Descent Variants

### Batch Gradient Descent

Uses the entire dataset for each update:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}^{(t)})$$

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def batch_gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 100,
    verbose: bool = True
) -> dict:
    """
    Batch Gradient Descent for Linear Regression
    
    Uses entire dataset for each gradient computation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples, 1)
        learning_rate: Step size η
        n_epochs: Number of iterations
    
    Returns:
        Dictionary with trained parameters and history
    """
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = torch.zeros(n_features, 1, requires_grad=False)
    b = torch.zeros(1, requires_grad=False)
    
    # History tracking
    history = {'loss': [], 'w': [], 'b': []}
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = X @ w + b
        
        # Compute loss
        loss = torch.mean((y - y_pred) ** 2)
        
        # Compute gradients manually
        error = y_pred - y  # (n, 1)
        grad_w = (2.0 / n_samples) * (X.T @ error)  # (d, 1)
        grad_b = (2.0 / n_samples) * torch.sum(error)  # scalar
        
        # Update parameters
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        
        # Store history
        history['loss'].append(loss.item())
        history['w'].append(w.clone())
        history['b'].append(b.item())
        
        if verbose and (epoch + 1) % (n_epochs // 10) == 0:
            print(f"Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
    
    return {'w': w, 'b': b, 'history': history}

# Example
torch.manual_seed(42)
n, d = 100, 3
X = torch.randn(n, d)
true_w = torch.tensor([[2.0], [-1.0], [0.5]])
true_b = 1.0
y = X @ true_w + true_b + 0.3 * torch.randn(n, 1)

print("Batch Gradient Descent:")
print("=" * 50)
result = batch_gradient_descent(X, y, learning_rate=0.1, n_epochs=100)

print(f"\nTrue w: {true_w.squeeze().tolist()}")
print(f"Learned w: {result['w'].squeeze().tolist()}")
print(f"True b: {true_b}")
print(f"Learned b: {result['b'].item():.4f}")
```

### Stochastic Gradient Descent (SGD)

Uses a single sample for each update:

```python
def stochastic_gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 100
) -> dict:
    """
    Stochastic Gradient Descent - one sample per update
    
    Pros: Fast updates, can escape local minima
    Cons: High variance, noisy convergence
    """
    n_samples, n_features = X.shape
    
    w = torch.zeros(n_features, 1)
    b = torch.zeros(1)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        epoch_loss = 0.0
        for i in range(n_samples):
            # Single sample
            xi = X_shuffled[i:i+1]  # (1, d)
            yi = y_shuffled[i:i+1]  # (1, 1)
            
            # Forward
            y_pred = xi @ w + b
            
            # Loss (for this sample)
            loss = (yi - y_pred) ** 2
            epoch_loss += loss.item()
            
            # Gradient (for single sample, no averaging)
            error = y_pred - yi
            grad_w = 2 * (xi.T @ error)
            grad_b = 2 * error.squeeze()
            
            # Update
            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b
        
        history['loss'].append(epoch_loss / n_samples)
    
    return {'w': w, 'b': b, 'history': history}

print("\nStochastic Gradient Descent:")
print("=" * 50)
result_sgd = stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=50)
print(f"Final loss: {result_sgd['history']['loss'][-1]:.6f}")
```

### Mini-Batch Gradient Descent

Uses small batches—the practical choice:

```python
def mini_batch_gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    n_epochs: int = 100
) -> dict:
    """
    Mini-Batch Gradient Descent - best of both worlds
    
    Balances computation efficiency with gradient quality.
    """
    n_samples, n_features = X.shape
    
    w = torch.zeros(n_features, 1)
    b = torch.zeros(1)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            # Get batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_n = X_batch.shape[0]
            
            # Forward
            y_pred = X_batch @ w + b
            
            # Compute batch loss
            loss = torch.mean((y_batch - y_pred) ** 2)
            epoch_loss += loss.item()
            n_batches += 1
            
            # Gradient (averaged over batch)
            error = y_pred - y_batch
            grad_w = (2.0 / batch_n) * (X_batch.T @ error)
            grad_b = (2.0 / batch_n) * torch.sum(error)
            
            # Update
            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b
        
        history['loss'].append(epoch_loss / n_batches)
    
    return {'w': w, 'b': b, 'history': history}

print("\nMini-Batch Gradient Descent (batch_size=32):")
print("=" * 50)
result_mb = mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.1, n_epochs=50)
print(f"Final loss: {result_mb['history']['loss'][-1]:.6f}")
```

## PyTorch Autograd Implementation

### Using requires_grad

```python
def gradient_descent_autograd(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 100
) -> dict:
    """
    Gradient Descent using PyTorch autograd
    
    Let PyTorch compute gradients automatically.
    """
    n_features = X.shape[1]
    
    # Parameters with gradient tracking
    w = torch.zeros(n_features, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        # Forward pass (builds computational graph)
        y_pred = X @ w + b
        
        # Compute loss
        loss = torch.mean((y - y_pred) ** 2)
        
        # Backward pass (computes gradients)
        loss.backward()
        
        # Update parameters (no gradient tracking)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        # Zero gradients for next iteration
        w.grad.zero_()
        b.grad.zero_()
        
        history['loss'].append(loss.item())
    
    return {'w': w.detach(), 'b': b.detach().item(), 'history': history}

print("\nGradient Descent with Autograd:")
print("=" * 50)
result_auto = gradient_descent_autograd(X, y, learning_rate=0.1, n_epochs=100)
print(f"Final loss: {result_auto['history']['loss'][-1]:.6f}")
```

### Using nn.Module and Optimizer

```python
def gradient_descent_pytorch_style(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.01,
    n_epochs: int = 100
) -> dict:
    """
    Standard PyTorch training pattern
    """
    n_features = X.shape[1]
    
    # Model
    model = nn.Linear(n_features, 1)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        # Forward
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # Backward
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        history['loss'].append(loss.item())
    
    return {
        'w': model.weight.detach(),
        'b': model.bias.detach().item(),
        'history': history
    }

print("\nPyTorch Standard Training:")
print("=" * 50)
result_pytorch = gradient_descent_pytorch_style(X, y, learning_rate=0.1, n_epochs=100)
print(f"Final loss: {result_pytorch['history']['loss'][-1]:.6f}")
```

## Learning Rate Analysis

### Effect of Learning Rate

```python
def learning_rate_experiment(X, y):
    """
    Demonstrate effect of different learning rates
    """
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    results = {}
    
    for lr in learning_rates:
        try:
            result = batch_gradient_descent(X, y, learning_rate=lr, 
                                           n_epochs=100, verbose=False)
            results[lr] = result['history']['loss']
        except:
            results[lr] = [float('nan')] * 100
    
    # Plot
    plt.figure(figsize=(10, 6))
    for lr, losses in results.items():
        if not np.isnan(losses[-1]):
            plt.plot(losses, label=f'lr={lr}', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Effect of Learning Rate on Convergence')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    print("\nLearning Rate Analysis:")
    print("-" * 40)
    for lr, losses in results.items():
        final = losses[-1] if not np.isnan(losses[-1]) else "DIVERGED"
        print(f"lr={lr:<6}: Final loss = {final}")
    
    return results

# lr_results = learning_rate_experiment(X, y)
```

### Optimal Learning Rate

For linear regression with MSE, the optimal learning rate is bounded by:

$$\eta < \frac{2}{\lambda_{\max}(\mathbf{X}^T\mathbf{X}/n)}$$

where $\lambda_{\max}$ is the largest eigenvalue.

```python
def compute_optimal_learning_rate(X: torch.Tensor) -> float:
    """
    Compute theoretical upper bound for stable learning rate
    """
    n = X.shape[0]
    XtX = X.T @ X / n
    eigenvalues = torch.linalg.eigvalsh(XtX)
    lambda_max = eigenvalues.max()
    
    optimal_lr = 2.0 / lambda_max.item()
    
    print(f"Largest eigenvalue of X'X/n: {lambda_max.item():.4f}")
    print(f"Theoretical max learning rate: {optimal_lr:.4f}")
    print(f"Safe learning rate (half): {optimal_lr/2:.4f}")
    
    return optimal_lr / 2

safe_lr = compute_optimal_learning_rate(X)
```

## Convergence Analysis

### Convergence Rate

For convex quadratic objectives (like MSE in linear regression):

- **Gradient Descent**: Linear convergence $O(\kappa^t)$ where $\kappa$ is condition number
- **SGD**: Sublinear convergence $O(1/t)$ with diminishing step sizes

```python
def analyze_convergence(X, y, learning_rate=0.1, n_epochs=200):
    """
    Analyze convergence properties of gradient descent
    """
    # Get optimal solution (via normal equations)
    ones = torch.ones(X.shape[0], 1)
    X_aug = torch.cat([ones, X], dim=1)
    theta_opt = torch.linalg.lstsq(X_aug, y).solution
    
    optimal_loss = torch.mean((y - X_aug @ theta_opt) ** 2).item()
    
    # Train with gradient descent
    result = batch_gradient_descent(X, y, learning_rate=learning_rate, 
                                   n_epochs=n_epochs, verbose=False)
    
    # Compute suboptimality gap
    losses = result['history']['loss']
    gaps = [loss - optimal_loss for loss in losses]
    
    print("Convergence Analysis:")
    print(f"Optimal loss: {optimal_loss:.6f}")
    print(f"Final GD loss: {losses[-1]:.6f}")
    print(f"Suboptimality gap: {gaps[-1]:.2e}")
    
    # Estimate convergence rate
    if len(gaps) > 10:
        # Linear convergence: gap_t = c * r^t
        # log(gap_t) = log(c) + t*log(r)
        log_gaps = np.log(np.array(gaps[10:]) + 1e-10)
        epochs = np.arange(10, len(gaps))
        
        # Linear fit to estimate rate
        slope, intercept = np.polyfit(epochs, log_gaps, 1)
        conv_rate = np.exp(slope)
        
        print(f"Estimated convergence rate: {conv_rate:.4f}")
        print(f"(Rate < 1 indicates linear convergence)")
    
    return gaps

# gaps = analyze_convergence(X, y, learning_rate=0.1)
```

## Comparison of Methods

```python
def compare_all_methods(X, y, n_epochs=100):
    """
    Compare all gradient descent variants
    """
    print("=" * 60)
    print("COMPARISON OF GRADIENT DESCENT METHODS")
    print("=" * 60)
    
    # Optimal solution
    ones = torch.ones(X.shape[0], 1)
    X_aug = torch.cat([ones, X], dim=1)
    theta_opt = torch.linalg.lstsq(X_aug, y).solution
    optimal_loss = torch.mean((y - X_aug @ theta_opt) ** 2).item()
    print(f"Optimal loss (Normal Equations): {optimal_loss:.6f}\n")
    
    methods = {
        'Batch GD': lambda: batch_gradient_descent(X, y, 0.1, n_epochs, False),
        'SGD': lambda: stochastic_gradient_descent(X, y, 0.01, n_epochs),
        'Mini-batch (32)': lambda: mini_batch_gradient_descent(X, y, 32, 0.1, n_epochs),
        'Mini-batch (16)': lambda: mini_batch_gradient_descent(X, y, 16, 0.1, n_epochs),
    }
    
    results = {}
    print(f"{'Method':<20} {'Final Loss':<15} {'Gap to Optimal':<15}")
    print("-" * 50)
    
    for name, method in methods.items():
        result = method()
        final_loss = result['history']['loss'][-1]
        gap = final_loss - optimal_loss
        results[name] = result['history']['loss']
        print(f"{name:<20} {final_loss:<15.6f} {gap:<15.2e}")
    
    return results

# comparison_results = compare_all_methods(X, y)
```

## Practical Considerations

### When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| Normal Equations | Small datasets (d < 10,000) | Exact, one-shot | O(d³) complexity |
| Batch GD | Medium datasets | Stable convergence | Full data each step |
| Mini-batch GD | Large datasets | Balanced | Requires tuning |
| SGD | Very large / streaming | Fast updates | Noisy, slow convergence |

### Hyperparameter Guidelines

```python
def hyperparameter_recommendations(n_samples, n_features):
    """
    Practical recommendations for gradient descent hyperparameters
    """
    print("Hyperparameter Recommendations:")
    print("=" * 50)
    print(f"Dataset: {n_samples} samples, {n_features} features\n")
    
    # Batch size
    if n_samples < 1000:
        rec_batch = min(32, n_samples)
    elif n_samples < 10000:
        rec_batch = 64
    else:
        rec_batch = 128
    
    print(f"Batch size: {rec_batch}")
    print(f"  - Powers of 2 for GPU efficiency")
    print(f"  - Larger batches: more stable, less frequent updates")
    print(f"  - Smaller batches: more noise, better generalization\n")
    
    # Learning rate
    print(f"Learning rate: Start with 0.01 or 0.1")
    print(f"  - Too high: divergence")
    print(f"  - Too low: slow convergence")
    print(f"  - Use learning rate schedules for fine-tuning\n")
    
    # Epochs
    n_batches_per_epoch = n_samples // rec_batch
    print(f"Epochs: Start with 100, monitor validation loss")
    print(f"  - {n_batches_per_epoch} batches per epoch")
    print(f"  - Use early stopping to prevent overfitting")

hyperparameter_recommendations(1000, 10)
```

## Summary

### Gradient Descent Algorithm

```
Initialize θ randomly
For epoch = 1 to n_epochs:
    For each batch (X_b, y_b):
        1. Forward: ŷ = X_b @ θ
        2. Loss: L = MSE(y_b, ŷ)
        3. Gradient: ∇L = (2/n) * X_b.T @ (ŷ - y_b)
        4. Update: θ = θ - η * ∇L
```

### Key Takeaways

1. **Gradient descent** iteratively minimizes the loss function
2. **Mini-batch** is the practical choice for most applications
3. **Learning rate** is the most important hyperparameter
4. **Autograd** computes gradients automatically in PyTorch
5. **Convergence** is guaranteed for convex problems with proper learning rate

## References

- Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
- Ruder, S. (2016). "An overview of gradient descent optimization algorithms"
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 8
