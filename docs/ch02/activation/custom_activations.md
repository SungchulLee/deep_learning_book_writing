# Custom Activation Functions

## Overview

While built-in activation functions cover most use cases, creating custom activations allows you to:
- Implement novel research ideas
- Add domain-specific inductive biases
- Create learnable activation functions
- Experiment with task-specific designs

This section covers how to implement custom activations in PyTorch with proper gradient support.

## Learning Objectives

By the end of this section, you will understand:

1. How to create custom activation modules
2. Implementing learnable (parametric) activations
3. Ensuring proper gradient flow
4. Testing custom activations
5. Best practices for deployment

---

## Creating Custom Activations

### Method 1: Simple nn.Module

The easiest approach is inheriting from `nn.Module`:

```python
import torch
import torch.nn as nn
import math

class CustomSmooth(nn.Module):
    """
    Custom smooth activation: f(x) = x * tanh(sqrt(x² + 1))
    
    Properties:
    - Smooth everywhere
    - Approximately linear for small |x|
    - Bounded growth rate for large |x|
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.sqrt(x**2 + 1))
```

### Method 2: With Configuration Parameters

```python
class ScaledActivation(nn.Module):
    """Activation with configurable scaling."""
    
    def __init__(self, scale=1.0, shift=0.0):
        super().__init__()
        self.scale = scale
        self.shift = shift
    
    def forward(self, x):
        return self.scale * torch.relu(x + self.shift)
    
    def extra_repr(self):
        return f'scale={self.scale}, shift={self.shift}'
```

### Method 3: Learnable Parameters

For parametric activations where parameters are learned during training:

```python
class ParametricActivation(nn.Module):
    """
    Learnable activation similar to PReLU.
    f(x) = x if x > 0 else alpha * x
    
    Alpha is learned during training.
    """
    
    def __init__(self, num_parameters=1, init_alpha=0.25):
        super().__init__()
        # nn.Parameter registers the tensor as a learnable parameter
        self.alpha = nn.Parameter(torch.full((num_parameters,), init_alpha))
    
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)
    
    def extra_repr(self):
        return f'num_parameters={self.alpha.numel()}'
```

---

## Practical Custom Activations

### Soft Clipping Activation

Bounds output smoothly without hard cutoffs:

```python
class SoftClip(nn.Module):
    """
    Soft clipping: f(x) = scale * tanh(x / scale)
    
    Smoothly bounds output to [-scale, scale].
    Unlike hard clipping, gradients flow everywhere.
    """
    
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return self.scale * torch.tanh(x / self.scale)

# Usage
soft_clip = SoftClip(scale=5.0)
x = torch.tensor([-100., -1., 0., 1., 100.])
print(soft_clip(x))  # Bounded to approximately [-5, 5]
```

### Adaptive Mixture Activation

Learns a weighted combination of multiple activations:

```python
class AdaptiveMixture(nn.Module):
    """
    Learnable mixture of ReLU, Tanh, and Identity.
    Network learns optimal combination during training.
    """
    
    def __init__(self):
        super().__init__()
        # Learnable mixing weights (3 activations)
        self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x):
        # Softmax ensures weights sum to 1
        w = torch.softmax(self.weights, dim=0)
        
        # Compute each activation
        relu_out = torch.relu(x)
        tanh_out = torch.tanh(x)
        identity_out = x
        
        # Weighted combination
        return w[0] * relu_out + w[1] * tanh_out + w[2] * identity_out
    
    def get_weights(self):
        """Return current mixing weights."""
        return torch.softmax(self.weights, dim=0).detach()
```

### Polynomial Activation

```python
class PolynomialActivation(nn.Module):
    """
    Learnable polynomial: f(x) = a₀ + a₁x + a₂x² + a₃x³
    
    Can approximate various activation shapes.
    """
    
    def __init__(self, degree=3):
        super().__init__()
        # Initialize with approximate ReLU coefficients
        self.coeffs = nn.Parameter(torch.zeros(degree + 1))
        with torch.no_grad():
            self.coeffs[1] = 0.5  # Linear term
    
    def forward(self, x):
        result = self.coeffs[0]
        x_power = x
        for i in range(1, len(self.coeffs)):
            result = result + self.coeffs[i] * x_power
            x_power = x_power * x
        return result
```

---

## Testing Custom Activations

### Gradient Verification

Always verify gradients work correctly:

```python
def test_gradient_flow(activation, input_shape=(10, 32)):
    """Test that gradients flow through the activation."""
    x = torch.randn(*input_shape, requires_grad=True)
    y = activation(x)
    loss = y.sum()
    loss.backward()
    
    # Check gradients exist and are finite
    assert x.grad is not None, "No gradient computed"
    assert torch.isfinite(x.grad).all(), "Gradient contains inf/nan"
    print(f"✓ Gradients OK: mean={x.grad.mean():.4f}, std={x.grad.std():.4f}")

# Test custom activation
custom = CustomSmooth()
test_gradient_flow(custom)
```

### Numerical Gradient Check

```python
def numerical_gradient_check(activation, x, eps=1e-5):
    """Compare analytical gradients to numerical approximation."""
    x = x.clone().requires_grad_(True)
    y = activation(x)
    y.sum().backward()
    analytical_grad = x.grad.clone()
    
    # Numerical gradient
    numerical_grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_flat = x.view(-1)
        
        # f(x + eps)
        x_flat[i] += eps
        y_plus = activation(x.view(x.shape)).sum()
        
        # f(x - eps)
        x_flat[i] -= 2 * eps
        y_minus = activation(x.view(x.shape)).sum()
        
        # Restore and compute gradient
        x_flat[i] += eps
        numerical_grad.view(-1)[i] = (y_plus - y_minus) / (2 * eps)
    
    # Compare
    diff = (analytical_grad - numerical_grad).abs().max()
    print(f"Max gradient difference: {diff:.2e}")
    assert diff < 1e-4, "Gradient check failed!"
    print("✓ Numerical gradient check passed")

# Test
x = torch.randn(5)
numerical_gradient_check(CustomSmooth(), x)
```

### Integration Test in Network

```python
def test_in_network(activation, epochs=100):
    """Test activation in a real training scenario."""
    from sklearn.datasets import make_moons
    
    # Create dataset
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(1)
    
    # Simple network
    model = nn.Sequential(
        nn.Linear(2, 32),
        activation,
        nn.Linear(32, 32),
        activation,
        nn.Linear(32, 1)
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) > 0.5).float()
        accuracy = (preds == y).float().mean()
    
    print(f"✓ Training completed. Accuracy: {accuracy:.4f}")
    return accuracy

# Test
test_in_network(CustomSmooth())
```

---

## Learnable Activation Example: Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableActivation(nn.Module):
    """
    Fully learnable activation that adapts its shape during training.
    
    Combines:
    1. Learnable negative slope (like PReLU)
    2. Learnable scaling
    3. Optional learnable shift
    
    f(x) = scale * (max(x, 0) + alpha * min(x, 0)) + shift
    """
    
    def __init__(self, num_channels=1, init_alpha=0.25, init_scale=1.0, 
                 learnable_shift=False):
        super().__init__()
        
        self.alpha = nn.Parameter(torch.full((num_channels,), init_alpha))
        self.scale = nn.Parameter(torch.full((num_channels,), init_scale))
        
        if learnable_shift:
            self.shift = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_buffer('shift', torch.zeros(num_channels))
    
    def forward(self, x):
        # Reshape parameters for broadcasting
        # Assumes x has shape [..., num_channels]
        alpha = self.alpha.view(1, -1)
        scale = self.scale.view(1, -1)
        shift = self.shift.view(1, -1)
        
        # PReLU-like operation
        pos = F.relu(x)
        neg = self.alpha * (-F.relu(-x))
        
        return scale * (pos + neg) + shift
    
    def extra_repr(self):
        return (f'num_channels={self.alpha.numel()}, '
                f'learnable_shift={isinstance(self.shift, nn.Parameter)}')


class NetworkWithLearnableActivation(nn.Module):
    """Example network using learnable activation."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = LearnableActivation(num_channels=hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = LearnableActivation(num_channels=hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)
    
    def print_activation_stats(self):
        """Print learned activation parameters."""
        print("Activation 1:")
        print(f"  Alpha: mean={self.act1.alpha.mean():.4f}, "
              f"std={self.act1.alpha.std():.4f}")
        print(f"  Scale: mean={self.act1.scale.mean():.4f}, "
              f"std={self.act1.scale.std():.4f}")
        
        print("Activation 2:")
        print(f"  Alpha: mean={self.act2.alpha.mean():.4f}, "
              f"std={self.act2.alpha.std():.4f}")
        print(f"  Scale: mean={self.act2.scale.mean():.4f}, "
              f"std={self.act2.scale.std():.4f}")
```

---

## Best Practices

### 1. Start Simple

```python
# ✅ Good: Simple, understandable
class SimpleCustom(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)  # Just Swish

# ❌ Avoid: Overly complex without justification
class OverlyComplex(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) * torch.tanh(x) * (x + 1) / (x.abs() + 1)
```

### 2. Ensure Gradient Flow

```python
# ✅ Good: Gradients flow everywhere
def good_activation(x):
    return x * torch.sigmoid(x)

# ❌ Bad: Zero gradient for many inputs
def bad_activation(x):
    return torch.where(x > 0, x, torch.zeros_like(x) * x)  # Gradient is 0 for x < 0
```

### 3. Handle Edge Cases

```python
class RobustActivation(nn.Module):
    def forward(self, x):
        # Clamp to prevent numerical issues
        x = torch.clamp(x, -50, 50)
        
        # Use numerically stable operations
        return x * torch.sigmoid(x)
```

### 4. Document Thoroughly

```python
class WellDocumented(nn.Module):
    """
    Custom activation: f(x) = x * tanh(ln(1 + exp(x)))
    
    Properties:
        - Smooth and differentiable everywhere
        - Bounded below by approximately -0.3
        - Unbounded above
        - Similar to Mish but slightly different shape
    
    Args:
        None
    
    Shape:
        - Input: (*)
        - Output: (*) (same shape as input)
    
    Example:
        >>> m = WellDocumented()
        >>> x = torch.randn(10)
        >>> y = m(x)
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
```

---

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Simple activations** | Inherit from nn.Module, implement forward() |
| **Learnable params** | Use nn.Parameter for trainable values |
| **Testing** | Verify gradients, test in real network |
| **Documentation** | Describe shape, properties, use cases |
| **Robustness** | Handle numerical edge cases |

!!! tip "When to Create Custom Activations"
    Create custom activations when:
    
    - Built-in activations don't meet your needs
    - You want to add task-specific inductive biases
    - Experimenting with novel architectures
    - Research purposes
    
    Stick with standard activations when:
    
    - Standard options work well for your task
    - You need reproducibility and comparability
    - Deployment simplicity is important
