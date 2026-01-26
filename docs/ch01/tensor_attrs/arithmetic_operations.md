# Arithmetic Operations

## Overview

PyTorch supports comprehensive arithmetic operations on tensors, enabling element-wise computations, matrix operations, and complex mathematical transformations essential for deep learning.

## Basic Arithmetic

### Element-wise Operations

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([5.0, 6.0, 7.0, 8.0])

# Addition
add_result = a + b
print(f"a + b = {add_result}")  # tensor([6., 8., 10., 12.])

# Subtraction
sub_result = a - b
print(f"a - b = {sub_result}")  # tensor([-4., -4., -4., -4.])

# Multiplication (element-wise)
mul_result = a * b
print(f"a * b = {mul_result}")  # tensor([5., 12., 21., 32.])

# Division
div_result = a / b
print(f"a / b = {div_result}")  # tensor([0.2000, 0.3333, 0.4286, 0.5000])
```

### Scalar Operations

```python
t = torch.tensor([1.0, 2.0, 3.0])

# Scalar addition
print(t + 10)  # tensor([11., 12., 13.])

# Scalar multiplication
print(t * 3)   # tensor([3., 6., 9.])

# Scalar division
print(t / 2)   # tensor([0.5, 1.0, 1.5])

# Scalar power
print(t ** 2)  # tensor([1., 4., 9.])
```

## Functional Interface

Every operator has a corresponding function:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Equivalent forms
print(a + b)                    # Operator
print(torch.add(a, b))          # Function
print(a.add(b))                 # Method

# Functions allow output tensor specification
out = torch.empty(3)
torch.add(a, b, out=out)        # Store result in pre-allocated tensor
print(f"With output: {out}")
```

## In-Place Operations

Operations ending with `_` modify tensors in-place:

```python
t = torch.tensor([1.0, 2.0, 3.0])
print(f"Original: {t}")

t.add_(10)                      # In-place addition
print(f"After add_(10): {t}")   # tensor([11., 12., 13.])

t.mul_(2)                       # In-place multiplication
print(f"After mul_(2): {t}")    # tensor([22., 24., 26.])

t.div_(2)                       # In-place division
print(f"After div_(2): {t}")    # tensor([11., 12., 13.])
```

!!! warning "In-Place Operations and Autograd"
    In-place operations can break gradient computation. Avoid them during training
    if the tensor is part of the computation graph.

## Mathematical Functions

### Power and Roots

```python
t = torch.tensor([1.0, 4.0, 9.0, 16.0])

# Power
print(torch.pow(t, 2))          # tensor([1., 16., 81., 256.])
print(t ** 0.5)                 # Square root via power

# Square root
print(torch.sqrt(t))            # tensor([1., 2., 3., 4.])

# Cube root
print(torch.pow(t, 1/3))        # tensor([1., 1.587, 2.080, 2.520])
```

### Exponential and Logarithm

```python
t = torch.tensor([1.0, 2.0, 3.0])

# Exponential
print(torch.exp(t))             # e^t: tensor([2.7183, 7.3891, 20.0855])

# Natural logarithm
print(torch.log(t))             # ln(t): tensor([0., 0.6931, 1.0986])

# Log base 10
print(torch.log10(t))           # tensor([0., 0.3010, 0.4771])

# Log base 2
print(torch.log2(t))            # tensor([0., 1., 1.5850])

# Log with offset (numerically stable)
print(torch.log1p(t))           # ln(1 + t): tensor([0.6931, 1.0986, 1.3863])
```

### Trigonometric Functions

```python
t = torch.tensor([0.0, torch.pi/6, torch.pi/4, torch.pi/3, torch.pi/2])

# Basic trig
print(torch.sin(t))             # tensor([0., 0.5, 0.7071, 0.8660, 1.])
print(torch.cos(t))             # tensor([1., 0.8660, 0.7071, 0.5, 0.])
print(torch.tan(t))             # tensor([0., 0.5774, 1., 1.7321, inf])

# Inverse trig
x = torch.tensor([0.0, 0.5, 1.0])
print(torch.asin(x))            # arcsin
print(torch.acos(x))            # arccos
print(torch.atan(x))            # arctan

# Two-argument arctan (preserves quadrant)
y = torch.tensor([1.0, 1.0, -1.0, -1.0])
x = torch.tensor([1.0, -1.0, -1.0, 1.0])
print(torch.atan2(y, x))        # tensor([0.7854, 2.3562, -2.3562, -0.7854])
```

### Hyperbolic Functions

```python
t = torch.tensor([0.0, 1.0, 2.0])

print(torch.sinh(t))            # Hyperbolic sine
print(torch.cosh(t))            # Hyperbolic cosine
print(torch.tanh(t))            # Hyperbolic tangent (common activation)
```

## Rounding Operations

```python
t = torch.tensor([-1.7, -0.5, 0.3, 1.5, 2.8])

# Round to nearest integer
print(torch.round(t))           # tensor([-2., -0., 0., 2., 3.])

# Floor (round down)
print(torch.floor(t))           # tensor([-2., -1., 0., 1., 2.])

# Ceiling (round up)
print(torch.ceil(t))            # tensor([-1., -0., 1., 2., 3.])

# Truncate (toward zero)
print(torch.trunc(t))           # tensor([-1., -0., 0., 1., 2.])

# Fractional part
print(torch.frac(t))            # tensor([-0.7, -0.5, 0.3, 0.5, 0.8])
```

## Sign and Absolute Value

```python
t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# Absolute value
print(torch.abs(t))             # tensor([3., 1., 0., 1., 3.])

# Sign
print(torch.sign(t))            # tensor([-1., -1., 0., 1., 1.])

# Negative
print(torch.neg(t))             # tensor([3., 1., -0., -1., -3.])
```

## Clipping and Clamping

```python
t = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])

# Clamp to range
clamped = torch.clamp(t, min=-2.0, max=2.0)
print(f"Clamp to [-2, 2]: {clamped}")
# tensor([-2., -1., 0., 1., 2.])

# Clamp minimum only
print(torch.clamp(t, min=0))    # ReLU-like: tensor([0., 0., 0., 1., 5.])

# Clamp maximum only
print(torch.clamp(t, max=0))    # tensor([-5., -1., 0., 0., 0.])
```

## Division Variants

```python
a = torch.tensor([7.0, 8.0, 9.0])
b = torch.tensor([2.0, 3.0, 4.0])

# True division (default)
print(a / b)                    # tensor([3.5, 2.6667, 2.25])

# Floor division
print(torch.floor_divide(a, b)) # tensor([3., 2., 2.])
print(a // b)                   # Operator equivalent

# Modulo (remainder)
print(torch.fmod(a, b))         # tensor([1., 2., 1.])
print(a % b)                    # Operator equivalent

# Remainder (Python-style, handles negatives differently)
print(torch.remainder(a, b))    # tensor([1., 2., 1.])
```

## Special Functions

```python
# Error function (Gaussian distribution)
t = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.erf(t))

# Complementary error function
print(torch.erfc(t))

# Gamma function (generalized factorial)
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(torch.lgamma(t))          # Log-gamma
print(torch.exp(torch.lgamma(t)))  # Approximate gamma(n) = (n-1)!

# Digamma function
print(torch.digamma(t))
```

## Reciprocal and Square Root Reciprocal

```python
t = torch.tensor([1.0, 2.0, 4.0, 8.0])

# Reciprocal (1/x)
print(torch.reciprocal(t))      # tensor([1., 0.5, 0.25, 0.125])

# Reciprocal square root (1/sqrt(x))
print(torch.rsqrt(t))           # tensor([1., 0.7071, 0.5, 0.3536])
```

## Linear Algebra Shortcuts

```python
# Addmm: beta*input + alpha*(mat1 @ mat2)
input = torch.randn(2, 3)
mat1 = torch.randn(2, 4)
mat2 = torch.randn(4, 3)

result = torch.addmm(input, mat1, mat2, beta=0.5, alpha=2.0)
# Equivalent to: 0.5 * input + 2.0 * (mat1 @ mat2)

# Addmv: beta*input + alpha*(mat @ vec)
input = torch.randn(2)
mat = torch.randn(2, 3)
vec = torch.randn(3)

result = torch.addmv(input, mat, vec)
```

## Numerical Stability

### Log-Sum-Exp

```python
# Numerically stable log(sum(exp(x)))
t = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive (overflow risk)
# naive = torch.log(torch.sum(torch.exp(t)))  # inf!

# Stable version
stable = torch.logsumexp(t, dim=0)
print(f"Log-sum-exp: {stable}")
```

### Softmax-Related

```python
logits = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Softmax (numerically stable internally)
softmax = torch.softmax(logits, dim=0)
print(f"Softmax: {softmax}")

# Log-softmax (more stable for loss computation)
log_softmax = torch.log_softmax(logits, dim=0)
print(f"Log-softmax: {log_softmax}")
```

## Practical Examples

### Feature Scaling

```python
# Min-max scaling to [0, 1]
features = torch.randn(100, 10)

min_vals = features.min(dim=0).values
max_vals = features.max(dim=0).values
scaled = (features - min_vals) / (max_vals - min_vals)

print(f"Scaled range: [{scaled.min():.4f}, {scaled.max():.4f}]")
```

### Z-Score Normalization

```python
features = torch.randn(100, 10)

mean = features.mean(dim=0)
std = features.std(dim=0)
normalized = (features - mean) / std

print(f"Mean after normalization: {normalized.mean(dim=0)}")
print(f"Std after normalization: {normalized.std(dim=0)}")
```

### Geometric Mean

```python
t = torch.tensor([1.0, 2.0, 4.0, 8.0])

# Geometric mean via log-exp
geometric_mean = torch.exp(torch.log(t).mean())
print(f"Geometric mean: {geometric_mean}")  # ≈ 2.83
```

## Summary

| Category | Operations |
|----------|------------|
| Basic | `+`, `-`, `*`, `/`, `**` |
| Functions | `add`, `sub`, `mul`, `div`, `pow` |
| In-place | `add_`, `sub_`, `mul_`, `div_` |
| Math | `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh` |
| Rounding | `round`, `floor`, `ceil`, `trunc` |
| Sign/Abs | `abs`, `sign`, `neg` |
| Clipping | `clamp`, `clip` |
| Division | `//`, `%`, `floor_divide`, `fmod` |

## See Also

- [Broadcasting Rules](broadcasting_rules.md) - Shape compatibility
- [Reduction Operations](reduction_operations.md) - Aggregating values
- [Linear Algebra Operations](linalg_operations.md) - Matrix operations
