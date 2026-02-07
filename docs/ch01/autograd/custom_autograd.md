# Custom Autograd Functions

## Overview

While PyTorch's built-in operations cover most use cases, there are situations where you need to define custom forward and backward passes: implementing operations with no native PyTorch equivalent, optimizing memory by writing custom backward logic, incorporating non-differentiable operations with hand-derived gradients, or wrapping external C/CUDA code. The `torch.autograd.Function` class provides the mechanism for defining custom autograd functions that integrate seamlessly with PyTorch's computational graph.

## Learning Objectives

By the end of this section, you will be able to:

1. Define custom autograd functions using `torch.autograd.Function`
2. Implement correct forward and backward methods with `ctx` for saving state
3. Apply `gradcheck` and `gradgradcheck` to verify gradient correctness
4. Handle functions with multiple inputs and outputs
5. Implement practical custom functions for clipping, activation, and numerical stability

## The `torch.autograd.Function` Interface

### Structure

A custom autograd function subclasses `torch.autograd.Function` and implements two static methods:

- **`forward(ctx, *args)`** — performs the computation and saves any tensors needed for the backward pass
- **`backward(ctx, *grad_outputs)`** — computes gradients with respect to each input

```python
import torch

class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save tensors needed for backward
        ctx.save_for_backward(input)
        # Compute and return output
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, = ctx.saved_tensors
        # Compute gradient: d(clamp)/d(input)
        grad_input = grad_output * (input > 0).float()
        return grad_input
```

### Applying Custom Functions

Use the `.apply()` method (never instantiate the class directly):

```python
# Create an alias for convenience
my_relu = MyFunction.apply

x = torch.randn(5, requires_grad=True)
y = my_relu(x)
loss = y.sum()
loss.backward()

print(f"x:      {x.data}")
print(f"x.grad: {x.grad}")
```

## The `ctx` Object

### Saving Tensors: `ctx.save_for_backward`

Use `ctx.save_for_backward()` to store tensors needed during the backward pass. This is the **only** safe way to save input/output tensors — it enables proper memory management and error checking:

```python
class QuadraticFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, c):
        """Compute y = ax² + bx + c"""
        ctx.save_for_backward(x, a, b)
        return a * x ** 2 + b * x + c
    
    @staticmethod
    def backward(ctx, grad_output):
        """dy/dx = 2ax + b, dy/da = x², dy/db = x, dy/dc = 1"""
        x, a, b = ctx.saved_tensors
        grad_x = grad_output * (2 * a * x + b)
        grad_a = grad_output * x ** 2
        grad_b = grad_output * x
        grad_c = grad_output  # dy/dc = 1
        return grad_x, grad_a, grad_b, grad_c
```

**Rules for `save_for_backward`:**

- Only tensors can be saved this way
- Input and output tensors should be saved via `save_for_backward` (not as Python attributes)
- Non-tensor data (ints, bools, etc.) can be stored as regular attributes on `ctx`

### Saving Non-Tensor Data

For non-tensor metadata, assign directly to `ctx`:

```python
class PowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent):
        """Compute x^n where n is a Python int."""
        ctx.save_for_backward(x)
        ctx.exponent = exponent    # Non-tensor: store as attribute
        return x ** exponent
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = ctx.exponent
        # d(x^n)/dx = n·x^(n-1)
        grad_x = grad_output * n * x ** (n - 1)
        return grad_x, None   # None for non-differentiable exponent
```

### Returning `None` for Non-Differentiable Inputs

`backward` must return one gradient per `forward` input. For inputs that don't require gradients (constants, integer arguments, etc.), return `None`:

```python
class ScaledActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, use_relu):
        """Apply scaled activation. scale and use_relu are not differentiable."""
        ctx.save_for_backward(x)
        ctx.scale = scale
        ctx.use_relu = use_relu
        out = torch.relu(x) if use_relu else torch.tanh(x)
        return scale * out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if ctx.use_relu:
            grad_x = grad_output * ctx.scale * (x > 0).float()
        else:
            grad_x = grad_output * ctx.scale * (1 - torch.tanh(x) ** 2)
        return grad_x, None, None  # None for scale and use_relu
```

## Verifying Gradients with `gradcheck`

### Numerical Gradient Checking

`torch.autograd.gradcheck` compares your analytical backward against finite-difference approximations. Always use this to verify custom functions:

```python
import torch
from torch.autograd import gradcheck

class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sigmoid = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(sigmoid)
        return sigmoid
    
    @staticmethod
    def backward(ctx, grad_output):
        sigmoid, = ctx.saved_tensors
        # σ'(x) = σ(x)(1 - σ(x))
        return grad_output * sigmoid * (1 - sigmoid)

# Test with double precision for numerical accuracy
x = torch.randn(5, dtype=torch.double, requires_grad=True)
result = gradcheck(MySigmoid.apply, (x,), eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {result}")
```

**Important:** Always use `dtype=torch.double` for gradient checking — single precision introduces too much numerical error.

### Checking Higher-Order Gradients

If your function supports higher-order derivatives, use `gradgradcheck`:

```python
from torch.autograd import gradgradcheck

x = torch.randn(3, dtype=torch.double, requires_grad=True)
result = gradgradcheck(MySigmoid.apply, (x,))
print(f"Second-order gradient check passed: {result}")
```

## Practical Examples

### Example 1: Straight-Through Estimator

The straight-through estimator (STE) uses a non-differentiable operation in the forward pass but passes gradients through unchanged in the backward pass:

```python
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """Binarize: output is 0 or 1."""
        return (x > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradient through unchanged (identity)."""
        return grad_output

ste_binarize = StraightThroughEstimator.apply

x = torch.tensor([-0.5, 0.3, 1.2, -0.1], requires_grad=True)
y = ste_binarize(x)
loss = y.sum()
loss.backward()

print(f"x:      {x.data}")
print(f"y:      {y.data}")        # [0, 1, 1, 0]
print(f"x.grad: {x.grad}")       # [1, 1, 1, 1] — gradient passes through
```

### Example 2: Numerically Stable Log-Sum-Exp

```python
class StableLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """Compute log(sum(exp(x))) with numerical stability."""
        c = x.max()  # Shift for stability
        exp_shifted = torch.exp(x - c)
        sum_exp = exp_shifted.sum()
        ctx.save_for_backward(exp_shifted)
        ctx.sum_exp = sum_exp
        return c + torch.log(sum_exp)
    
    @staticmethod
    def backward(ctx, grad_output):
        """d/dx_i log(sum(exp(x))) = exp(x_i) / sum(exp(x)) = softmax(x)_i"""
        exp_shifted, = ctx.saved_tensors
        softmax = exp_shifted / ctx.sum_exp
        return grad_output * softmax

# Verify
x = torch.randn(5, dtype=torch.double, requires_grad=True)
assert gradcheck(StableLogSumExp.apply, (x,))

# Compare with built-in
x_float = x.float().detach().requires_grad_(True)
custom = StableLogSumExp.apply(x_float)
builtin = torch.logsumexp(x_float.detach().requires_grad_(True), dim=0)
print(f"Custom:  {custom.item():.6f}")
print(f"Built-in: {builtin.item():.6f}")
```

### Example 3: Custom Gradient Clipping Function

```python
class ClipGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_value):
        """Forward pass is identity; backward clips the gradient."""
        ctx.clip_value = clip_value
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        """Clip gradient to [-clip_value, clip_value]."""
        clipped = grad_output.clamp(-ctx.clip_value, ctx.clip_value)
        return clipped, None

clip_grad = ClipGradient.apply

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = clip_grad(x, 0.5)
loss = (y ** 2).sum()   # d(loss)/dx = 2x = [2, 4, 6]
loss.backward()

print(f"x.grad (clipped to ±0.5): {x.grad}")  # [0.5, 0.5, 0.5]
```

### Example 4: Function with Multiple Outputs

```python
class SplitAndTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """Split x into positive and negative parts."""
        ctx.save_for_backward(x)
        pos = torch.relu(x)
        neg = torch.relu(-x)
        return pos, neg
    
    @staticmethod
    def backward(ctx, grad_pos, grad_neg):
        """Backward receives one gradient per output."""
        x, = ctx.saved_tensors
        mask_pos = (x > 0).float()
        mask_neg = (x < 0).float()
        grad_x = grad_pos * mask_pos - grad_neg * mask_neg
        return grad_x

x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
pos, neg = SplitAndTransform.apply(x)

loss = pos.sum() + neg.sum()
loss.backward()
print(f"x:      {x.data}")
print(f"x.grad: {x.grad}")  # [-1, -1, 0, 1, 1]
```

## Supporting Higher-Order Gradients

To make a custom function compatible with `create_graph=True` (needed for higher-order derivatives), the backward pass must itself use differentiable PyTorch operations:

```python
class DifferentiableClamp(torch.autograd.Function):
    """Clamp with straight-through gradient, supporting higher-order derivatives."""
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        ctx.save_for_backward(x)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return x.clamp(min_val, max_val)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Use differentiable operations so higher-order grads work
        mask = ((x >= ctx.min_val) & (x <= ctx.max_val)).float()
        return grad_output * mask, None, None
```

## Integration with `nn.Module`

Custom autograd functions are typically wrapped in `nn.Module` for use in model definitions:

```python
import torch
import torch.nn as nn

class SmoothL1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        ctx.beta = beta
        abs_x = x.abs()
        out = torch.where(abs_x < beta, 0.5 * x**2 / beta, abs_x - 0.5 * beta)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        beta = ctx.beta
        abs_x = x.abs()
        grad_x = torch.where(abs_x < beta, x / beta, x.sign())
        return grad_output * grad_x, None

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, prediction, target):
        diff = prediction - target
        return SmoothL1Function.apply(diff, self.beta).mean()

# Usage in a model
criterion = SmoothL1Loss(beta=0.5)
pred = torch.randn(10, requires_grad=True)
target = torch.randn(10)
loss = criterion(pred, target)
loss.backward()
```

## Best Practices

### Do's

1. **Always verify with `gradcheck`** — numerical checking catches subtle errors
2. **Use `save_for_backward` for tensors** — it integrates with PyTorch's memory management
3. **Use `double` precision** in gradient checks for reliable numerical differentiation
4. **Return `None`** for inputs that don't require gradients
5. **Document the mathematical gradient** alongside the code

### Don'ts

1. **Don't store tensors as `ctx` attributes** — use `ctx.save_for_backward` instead
2. **Don't modify saved tensors** — they share storage with the forward pass
3. **Don't forget any return values in `backward`** — one per `forward` input
4. **Don't use in-place operations** on saved tensors in backward

## Summary

| Component | Purpose |
|-----------|---------|
| `torch.autograd.Function` | Base class for custom differentiable operations |
| `forward(ctx, *args)` | Define the forward computation |
| `backward(ctx, *grad_outputs)` | Define gradient computation (one grad per output) |
| `ctx.save_for_backward()` | Store tensors for backward pass |
| `ctx.saved_tensors` | Retrieve stored tensors in backward |
| `.apply()` | Invoke the custom function |
| `gradcheck()` | Verify backward against finite differences |
| `gradgradcheck()` | Verify second-order gradients |

## Common Pitfalls

1. **Shape mismatches** — `backward` must return gradients with the same shapes as the corresponding `forward` inputs
2. **Missing `None` returns** — every `forward` argument must have a corresponding return in `backward`
3. **Not using `save_for_backward`** — storing tensors as plain attributes can cause memory issues and errors with `retain_graph`
4. **Single-precision `gradcheck`** — always use `torch.double` for gradient verification

## References

- PyTorch Custom Function Tutorial: [https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)
- Extending Autograd: [https://pytorch.org/docs/stable/notes/extending.html](https://pytorch.org/docs/stable/notes/extending.html)
- `torch.autograd.Function` API: [https://pytorch.org/docs/stable/autograd.html#function](https://pytorch.org/docs/stable/autograd.html#function)
