# In-place Operations

In-place operations modify tensors directly without creating new storage. They are identified by an underscore suffix (e.g., `add_`, `mul_`). Understanding when to use them is crucial for both performance and correctness.

## Naming Convention

In-place operations end with an underscore:

```python
import torch

x = torch.tensor([1., 2., 3.])

# Out-of-place: creates new tensor
y = x.add(10)  # x unchanged, y is new

# In-place: modifies x directly
x.add_(10)     # x is modified
```

## Basic In-place Operations

### Arithmetic

```python
x = torch.tensor([1., 2., 3., 4., 5.])

x.add_(10)        # x = x + 10
x.sub_(5)         # x = x - 5
x.mul_(2)         # x = x * 2
x.div_(4)         # x = x / 4
x.pow_(2)         # x = x ** 2
x.neg_()          # x = -x
x.abs_()          # x = |x|
x.sqrt_()         # x = √x
```

### Clamping

```python
x = torch.randn(5)

x.clamp_(min=0)           # ReLU-like
x.clamp_(max=1)           # Upper bound
x.clamp_(min=-1, max=1)   # Range clamp
```

### Initialization

```python
x = torch.empty(3, 3)

x.fill_(7.0)              # All elements to 7
x.zero_()                 # All zeros
x.uniform_(-1, 1)         # Uniform random
x.normal_(mean=0, std=1)  # Normal random
```

### Copying

```python
src = torch.randn(3, 3)
dst = torch.empty(3, 3)

dst.copy_(src)  # Copy src data into dst
```

## Memory and Aliasing

In-place operations modify the underlying storage:

```python
a = torch.randn(3, 4)
b = a  # b is NOT a copy, it's an alias

# In-place on a affects b
a.add_(1)
print(torch.equal(a, b))  # True - same data
```

This is crucial when tensors share storage:

```python
x = torch.arange(12)
view = x.view(3, 4)

# Modifying view affects x
view.fill_(0)
print(x)  # All zeros!
```

## Autograd Restrictions

### Leaf Tensors with requires_grad=True

In-place operations on leaf tensors with gradients are **prohibited**:

```python
leaf = torch.tensor([1., 2., 3.], requires_grad=True)

try:
    leaf.add_(1)  # ERROR!
except RuntimeError as e:
    print("Cannot modify leaf tensor in-place")
```

**Why?** The autograd system needs the original values to compute gradients.

### Solutions

**Solution 1: Out-of-place operation**

```python
leaf = torch.tensor([1., 2., 3.], requires_grad=True)
result = leaf + 1  # Creates new tensor, gradient flows
```

**Solution 2: Use torch.no_grad() for parameter updates**

```python
param = torch.tensor([1., 2., 3.], requires_grad=True)

# During training loop:
with torch.no_grad():
    param.add_(1)  # OK inside no_grad()
```

**Solution 3: Use optimizer**

```python
import torch.optim as optim

param = torch.nn.Parameter(torch.randn(3))
optimizer = optim.SGD([param], lr=0.1)

# Proper update flow
loss = param.sum()
loss.backward()
optimizer.step()  # Updates param safely
```

### Non-leaf Tensors

In-place operations on intermediate results also break autograd:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2  # y is non-leaf

try:
    y.add_(1)  # ERROR! Breaks gradient computation
except RuntimeError as e:
    print("Cannot modify non-leaf in-place")
```

## When to Use In-place Operations

### ✓ Good Use Cases

**1. Parameter updates inside no_grad:**

```python
with torch.no_grad():
    for param in model.parameters():
        param.add_(-learning_rate * param.grad)
```

**2. Explicit tensor initialization:**

```python
weights = torch.empty(100, 50)
weights.normal_(0, 0.01)  # Initialize with small random values
```

**3. Memory-critical situations:**

```python
# When you're sure you won't need the original
buffer = torch.zeros(1000, 1000)
buffer.uniform_(0, 1)  # Saves memory allocation
```

**4. Tensor not used in autograd:**

```python
# Preprocessing tensors
data = torch.randn(100, 50)  # No requires_grad
data.clamp_(0, 1)   # Safe normalization
```

### ✗ Avoid In-place Operations

**1. On leaf tensors with requires_grad:**

```python
# DON'T
param = torch.randn(3, requires_grad=True)
param.mul_(2)  # ERROR

# DO
param = torch.randn(3, requires_grad=True)
param = param * 2  # Creates new tensor
```

**2. On intermediate computation results:**

```python
x = torch.randn(3, requires_grad=True)
y = x.relu()

# DON'T
y.add_(1)  # Breaks autograd

# DO
y = y + 1  # Safe
```

**3. When code clarity is more important:**

```python
# Less clear
x.mul_(2).add_(1).pow_(2)

# More clear
y = (x * 2 + 1) ** 2
```

**4. When tensors might be aliased:**

```python
x = torch.randn(10)
y = x[::2]  # View of x

# Dangerous: affects both x and y
y.fill_(0)

# Safer: make a copy first
y = x[::2].clone()
y.fill_(0)
```

## Indexed In-place Assignment

Assignment via indexing is in-place:

```python
x = torch.zeros(5)

# In-place assignment
x[1:4] = torch.tensor([10., 20., 30.])
print(x)  # tensor([0., 10., 20., 30., 0.])

# Boolean mask assignment
x[x > 15] = -1
print(x)  # tensor([0., 10., -1., -1., 0.])
```

## Common In-place Patterns

### Gradient Clipping

```python
def clip_gradients(model, max_norm):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.clamp_(-max_norm, max_norm)
```

### Weight Initialization

```python
def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()
```

### Masked Fill for Attention

```python
scores = torch.randn(4, 4)
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()

scores.masked_fill_(mask, float('-inf'))
```

## Quick Reference

| Operation | Description |
|-----------|-------------|
| `add_(x)` | Add x |
| `sub_(x)` | Subtract x |
| `mul_(x)` | Multiply by x |
| `div_(x)` | Divide by x |
| `pow_(x)` | Raise to power x |
| `neg_()` | Negate |
| `abs_()` | Absolute value |
| `sqrt_()` | Square root |
| `clamp_(min, max)` | Clamp to range |
| `fill_(x)` | Fill with value |
| `zero_()` | Fill with zeros |
| `uniform_(a, b)` | Uniform random |
| `normal_(μ, σ)` | Normal random |
| `copy_(src)` | Copy from source |
| `masked_fill_(mask, val)` | Fill where mask is True |

## Key Takeaways

1. **In-place operations end with `_`**
2. **Forbidden on leaf tensors** with `requires_grad=True`
3. **Use `torch.no_grad()`** for safe parameter updates
4. **Be aware of aliasing** - views share storage
5. **Prefer out-of-place** for clarity in most code
6. **Use in-place** for initialization and memory-critical situations
