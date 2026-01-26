# In-place Operations

In-place operations modify tensors directly without creating new storage. They are identified by an underscore suffix (e.g., `add_`, `mul_`). Understanding when to use them is crucial for both performance and correctness.

## Naming Convention

In-place operations end with an underscore:

```python
import torch

x = torch.tensor([1., 2., 3.])

# Out-of-place: creates new tensor, x unchanged
y = x.add(10)
print(f"x: {x}")  # tensor([1., 2., 3.])
print(f"y: {y}")  # tensor([11., 12., 13.])

# In-place: modifies x directly
x.add_(10)
print(f"x after add_: {x}")  # tensor([11., 12., 13.])
```

## Basic In-place Operations

### Arithmetic Operations

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
x.exp_()          # x = e^x
x.log_()          # x = log(x)
```

### Clamping and Rounding

```python
x = torch.randn(5)

x.clamp_(min=0)              # ReLU-like (floor at 0)
x.clamp_(max=1)              # Ceiling at 1
x.clamp_(min=-1, max=1)      # Clamp to range [-1, 1]

x.floor_()                   # Round down
x.ceil_()                    # Round up
x.round_()                   # Round to nearest
x.trunc_()                   # Truncate toward zero
```

### Initialization Operations

```python
x = torch.empty(3, 3)

x.fill_(7.0)                 # All elements set to 7
x.zero_()                    # All zeros
x.uniform_(-1, 1)            # Uniform random in [-1, 1]
x.normal_(mean=0, std=1)     # Normal distribution
x.exponential_(lambd=1)      # Exponential distribution
x.random_(0, 10)             # Random integers in [0, 10)
```

### Element-wise Operations

```python
x = torch.randn(5)

x.sigmoid_()                 # Sigmoid activation
x.tanh_()                    # Tanh activation
x.relu_()                    # ReLU activation (equivalent to clamp_(min=0))
x.sign_()                    # Sign function (-1, 0, or 1)
```

### Copying

```python
src = torch.randn(3, 3)
dst = torch.empty(3, 3)

dst.copy_(src)  # Copy src data into dst
```

## Memory and Aliasing

### Shared Storage

In-place operations modify the underlying storage, affecting all tensors that share it:

```python
a = torch.randn(3, 4)
b = a  # b is an alias, NOT a copy

# In-place on a affects b
a.add_(1)
print(torch.equal(a, b))  # True - same data
```

### Views Share Storage

```python
x = torch.arange(12)
view = x.view(3, 4)

# Modifying view affects x
view.fill_(0)
print(x)  # All zeros!

# Modifying x affects view
x.fill_(1)
print(view)  # All ones!
```

### Slices are Views

```python
x = torch.arange(10, dtype=torch.float32)
slice_view = x[2:5]

# In-place on slice affects original
slice_view.mul_(10)
print(x)  # tensor([0., 1., 20., 30., 40., 5., 6., 7., 8., 9.])
```

## Autograd Restrictions

### Leaf Tensors with requires_grad

In-place operations on leaf tensors with `requires_grad=True` are **prohibited**:

```python
leaf = torch.tensor([1., 2., 3.], requires_grad=True)

try:
    leaf.add_(1)  # RuntimeError!
except RuntimeError as e:
    print("Cannot modify leaf tensor in-place")
```

**Why?** The autograd system needs the original values to compute gradients during backpropagation.

### Non-leaf Intermediate Tensors

In-place operations on intermediate computation results also break autograd:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2  # y is non-leaf (result of computation)

try:
    y.add_(1)  # RuntimeError!
except RuntimeError as e:
    print("Cannot modify intermediate tensor in-place")
```

### Version Counter

PyTorch tracks tensor modifications with a version counter:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2

print(f"y._version: {y._version}")  # 0

# This bypasses autograd (dangerous!)
y.data.add_(1)

print(f"y._version: {y._version}")  # 1

# backward() may now fail or give incorrect gradients
```

## Safe In-place Patterns

### Inside `torch.no_grad()`

Parameter updates during training:

```python
param = torch.tensor([1., 2., 3.], requires_grad=True)
grad = torch.tensor([0.1, 0.2, 0.3])

# Safe: inside no_grad context
with torch.no_grad():
    param.add_(-0.01 * grad)  # OK
    # or equivalently:
    param -= 0.01 * grad      # Also in-place
```

### On Tensors Without Gradients

```python
# Preprocessing - no gradients involved
data = torch.randn(100, 50)  # requires_grad=False by default
data.clamp_(0, 1)            # Safe normalization
data.div_(data.max())        # Safe scaling
```

### Gradient Clipping

```python
def clip_gradients(model, max_norm):
    """Clip gradients in-place (gradients don't require grad)."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.clamp_(-max_norm, max_norm)
```

### Weight Initialization

```python
def init_weights(module):
    """Initialize weights in-place."""
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()

model.apply(init_weights)
```

### Masked Operations

```python
# Attention masking
scores = torch.randn(4, 4)
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()

scores.masked_fill_(mask, float('-inf'))
```

## Indexed Assignment

Assignment via indexing is implicitly in-place:

```python
x = torch.zeros(5)

# These are all in-place operations
x[0] = 1.0
x[1:4] = torch.tensor([10., 20., 30.])
x[x > 15] = -1

print(x)  # tensor([1., 10., -1., -1., 0.])
```

### Scatter and Index Operations

```python
x = torch.zeros(5)
indices = torch.tensor([0, 2, 4])
values = torch.tensor([1., 2., 3.])

# In-place scatter
x.scatter_(0, indices, values)
print(x)  # tensor([1., 0., 2., 0., 3.])

# In-place index copy
x.index_copy_(0, indices, values * 10)
print(x)  # tensor([10., 0., 20., 0., 30.])

# In-place index fill
x.index_fill_(0, torch.tensor([1, 3]), -1)
print(x)  # tensor([10., -1., 20., -1., 30.])
```

## When to Use In-place Operations

### ✓ Good Use Cases

**1. Parameter updates inside `no_grad()`:**
```python
with torch.no_grad():
    for param in model.parameters():
        param.add_(-learning_rate * param.grad)
```

**2. Tensor initialization:**
```python
weights = torch.empty(100, 50)
weights.normal_(0, 0.01)
```

**3. Memory-critical situations:**
```python
# Large tensor, can't afford extra allocation
buffer = torch.zeros(10000, 10000)
buffer.uniform_(0, 1)
```

**4. Preprocessing (no gradients):**
```python
images = load_images()  # No requires_grad
images.div_(255.0)      # Normalize to [0, 1]
```

**5. Modifying gradients:**
```python
# Gradients themselves don't require grad
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        param.grad.mul_(0.5)  # Scale gradients
```

### ✗ Avoid In-place Operations

**1. On leaf tensors with `requires_grad=True`:**
```python
# DON'T
param = torch.randn(3, requires_grad=True)
param.mul_(2)  # RuntimeError

# DO
param = torch.randn(3, requires_grad=True)
param = param * 2  # Creates new tensor
```

**2. On intermediate computation results:**
```python
x = torch.randn(3, requires_grad=True)
y = x.relu()

# DON'T
y.add_(1)  # RuntimeError

# DO
y = y + 1
```

**3. When tensors might be aliased unexpectedly:**
```python
x = torch.randn(10)
y = x[::2]  # View of x

# Dangerous: affects both x and y
y.fill_(0)

# Safer: clone first if independence needed
y = x[::2].clone()
y.fill_(0)  # Only affects y
```

**4. When code clarity matters more than memory:**
```python
# Less readable
x.mul_(2).add_(1).pow_(2)

# More readable
y = (x * 2 + 1) ** 2
```

## Common Mistakes

### Forgetting About Aliasing

```python
# Subtle bug
def process(tensor):
    tensor.mul_(2)  # Modifies the original!
    return tensor

x = torch.randn(5)
y = process(x)
# x has been modified too!
```

**Fix:**
```python
def process(tensor):
    result = tensor.clone()
    result.mul_(2)
    return result
```

### In-place in Forward Pass

```python
class BadLayer(torch.nn.Module):
    def forward(self, x):
        x.relu_()  # DON'T: modifies input, breaks autograd
        return x

class GoodLayer(torch.nn.Module):
    def forward(self, x):
        return x.relu()  # DO: creates new tensor
```

### Chained In-place Operations

```python
# All operations modify the same tensor
x = torch.randn(5)
x.add_(1).mul_(2).sub_(3)  # Each step modifies x

# Be explicit about what you want
x = torch.randn(5)
y = (x + 1) * 2 - 3  # New tensor, x unchanged
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
| `exp_()` | Exponential |
| `log_()` | Natural logarithm |
| `clamp_(min, max)` | Clamp to range |
| `floor_()` | Round down |
| `ceil_()` | Round up |
| `round_()` | Round to nearest |
| `fill_(x)` | Fill with value |
| `zero_()` | Fill with zeros |
| `uniform_(a, b)` | Uniform random |
| `normal_(μ, σ)` | Normal random |
| `copy_(src)` | Copy from source |
| `masked_fill_(mask, val)` | Fill where mask is True |
| `scatter_(dim, idx, src)` | Scatter values |
| `index_fill_(dim, idx, val)` | Fill at indices |
| `index_copy_(dim, idx, src)` | Copy at indices |

## Key Takeaways

1. **In-place operations end with `_`** (e.g., `add_`, `mul_`)
2. **Prohibited on leaf tensors** with `requires_grad=True`
3. **Prohibited on intermediate tensors** in computation graph
4. **Use `torch.no_grad()`** for safe parameter updates
5. **Be aware of aliasing** — views and slices share storage
6. **Prefer out-of-place** for clarity and autograd safety
7. **Use in-place** for initialization, preprocessing, and memory-critical code
8. **Indexed assignment is in-place** (`x[0] = 1` modifies `x`)

## See Also

- [Clone and Copy Operations](clone_and_copy.md) - Creating independent copies
- [Autograd Fundamentals](../gradients/autograd_fundamentals.md) - Gradient tracking
- [Memory Layout and Strides](../tensors/memory_layout_strides.md) - Storage details
