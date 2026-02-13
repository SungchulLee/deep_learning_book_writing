# Polynomial Fitting to Sine Curve: Six Implementations

## Overview

This document demonstrates polynomial fitting to the sine function using six progressively more PyTorch-idiomatic implementations. Starting from raw NumPy and building up to a customized `nn.Module`, each version introduces one new PyTorch concept while solving the same problem: fitting $y = \sin(x)$ with a polynomial $y = a + bx + cx^2 + dx^3$.

The data is shared across all implementations:

```python
import numpy as np
import torch
import math

# Generate training data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)
```

## Implementation 1: Pure NumPy

No PyTorch. Manual gradient computation using NumPy arrays.

```python
import numpy as np
import math

# Data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Random initialization
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: y_pred = a + b*x + c*x^2 + d*x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Loss: MSE
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backward pass: manual gradient computation
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
```

**Key concept:** Manual gradient derivation. For $\mathcal{L} = \sum_i (y_{\text{pred},i} - y_i)^2$:

$$\frac{\partial \mathcal{L}}{\partial a} = \sum_i 2(y_{\text{pred},i} - y_i), \quad \frac{\partial \mathcal{L}}{\partial b} = \sum_i 2(y_{\text{pred},i} - y_i) \cdot x_i$$

## Implementation 2: PyTorch Tensors (Manual Gradients)

Replace NumPy arrays with PyTorch tensors, but still compute gradients manually.

```python
import torch
import math

dtype = torch.float
device = torch.device("cpu")

# Data as tensors
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Random initialization
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backward pass: manual gradients (same math, torch ops)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update (no_grad to prevent tracking)
    with torch.no_grad():
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

**Key concept:** `torch.no_grad()` is needed during parameter updates to prevent the update operations from being tracked in the computational graph.

## Implementation 3: PyTorch Autograd

Let PyTorch compute gradients automatically via `requires_grad=True`.

```python
import torch
import math

dtype = torch.float
device = torch.device("cpu")

# Data (no gradient needed)
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Parameters with gradient tracking
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Backward pass: autograd computes all gradients
    loss.backward()

    # Update parameters (must be inside no_grad)
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Zero gradients for next iteration
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

**Key concept:** `loss.backward()` replaces all manual gradient computation. Gradients must be zeroed after each update to prevent accumulation.

## Implementation 4: PyTorch nn.Module

Use `nn.Linear` to define the model, replacing manual parameter management.

```python
import torch
import math

# Data: x has shape (2000,), need to create polynomial features
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Create polynomial features: [1, x, x^2, x^3] -> shape (2000, 4)
# Using Horner-like column construction
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)  # Shape: (2000, 3) for x, x^2, x^3

# nn.Linear(3, 1) computes y = x @ w^T + b, giving a + b*x + c*x^2 + d*x^3
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)  # Flatten (2000, 1) -> (2000,)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    y_pred = model(xx)

    # Loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Update parameters manually
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

print(f'Result: y = {model[0].bias.item()} + '
      f'{model[0].weight[0, 0].item()} x + '
      f'{model[0].weight[0, 1].item()} x^2 + '
      f'{model[0].weight[0, 2].item()} x^3')
```

**Key concept:** `nn.Linear` manages parameters automatically. `model.zero_grad()` zeros all parameter gradients. `model.parameters()` iterates over all learnable parameters.

## Implementation 5: PyTorch Optimizer

Replace manual parameter updates with `torch.optim.SGD`.

```python
import torch
import math

# Data
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

# Use an optimizer instead of manual updates
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for t in range(2000):
    # Forward pass
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Optimizer step replaces manual parameter update
    optimizer.step()

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + '
      f'{linear_layer.weight[0, 0].item()} x + '
      f'{linear_layer.weight[0, 1].item()} x^2 + '
      f'{linear_layer.weight[0, 2].item()} x^3')
```

**Key concept:** `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` is the canonical PyTorch training loop pattern.

## Implementation 6: Custom nn.Module

Define a custom module with `nn.Module` for full control over the architecture.

```python
import torch
import math

class Polynomial3(torch.nn.Module):
    """Custom module for cubic polynomial: y = a + b*x + c*x^2 + d*x^3"""
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        return (f'y = {self.a.item():.4f} + {self.b.item():.4f} x + '
                f'{self.c.item():.4f} x^2 + {self.d.item():.4f} x^3')

# Data
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Model, loss, optimizer
model = Polynomial3()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for t in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```

**Key concept:** `nn.Parameter` wraps a tensor as a learnable parameter. Custom `nn.Module` subclasses define `__init__` (parameters and sub-modules) and `forward` (computation).

## Progression Summary

| Implementation | Gradients | Parameters | Updates | PyTorch Level |
|---|---|---|---|---|
| 1. NumPy | Manual | NumPy arrays | Manual | None |
| 2. Torch Tensors | Manual | Tensors | Manual | Basic |
| 3. Autograd | Automatic | `requires_grad` | Manual | Intermediate |
| 4. nn.Module | Automatic | `nn.Linear` | Manual | Standard |
| 5. Optimizer | Automatic | `nn.Linear` | `optim.SGD` | Standard |
| 6. Custom Module | Automatic | `nn.Parameter` | `optim.SGD` | Advanced |

Each step eliminates one piece of manual work, converging on the idiomatic PyTorch pattern: define a model (`nn.Module`), specify a loss function, choose an optimizer, and run the training loop.
