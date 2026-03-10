# Python for Deep Learning

Python is the dominant language for deep learning research and development. This page explains why Python became the standard and introduces the core language features that matter most for numerical computing.

## Definition

Python is a dynamically typed, interpreted language with a rich ecosystem of scientific computing libraries. In deep learning, Python serves as the high-level interface while performance-critical operations execute in compiled backends (C++, CUDA, Fortran). The key libraries are NumPy for array computation, PyTorch and TensorFlow for differentiable programming, and scikit-learn for classical machine learning.

## Explanation

Python dominates deep learning for three reasons:

- **Minimal boilerplate**: Python's clean syntax lets researchers express complex models in few lines. A fully connected layer, loss function, and training loop fit on a single screen.
- **Ecosystem depth**: PyTorch, NumPy, pandas, and Hugging Face Transformers provide production-grade implementations of nearly every technique in modern deep learning.
- **Interactive workflow**: Jupyter notebooks enable the exploratory, iterative style of work that deep learning research demands -- train for a few epochs, inspect gradients, adjust, repeat.

The key Python features for deep learning code are:

- **List comprehensions and generators** for data pipeline transformations
- **Slicing and broadcasting** via NumPy/PyTorch tensor semantics
- **Context managers** (`with torch.no_grad():`) for controlling autograd behavior
- **Decorators** (`@torch.jit.script`, `@torch.compile`) for compilation and optimization

## Examples

```python
import torch
import numpy as np

# NumPy: foundation of Python's numerical ecosystem
x_np = np.random.randn(3, 4)
print(f"NumPy array shape: {x_np.shape}")

# PyTorch: NumPy-like API with autograd and GPU support
x = torch.randn(3, 4, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(f"Gradient shape: {x.grad.shape}")
print(f"Gradient (should be 2*x):\n{x.grad}")

# List comprehension for batch processing
batch_sizes = [2 ** i for i in range(4, 8)]
print(f"Batch sizes: {batch_sizes}")

# Context manager to disable gradient tracking during inference
with torch.no_grad():
    pred = torch.sigmoid(torch.randn(5))
    print(f"Predictions: {pred}")
```
