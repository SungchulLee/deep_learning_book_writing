"""
==============================================================================
01_pytorch_basics.py
==============================================================================
DIFFICULTY: ⭐ (Beginner)

DESCRIPTION:
    Introduction to PyTorch tensors and automatic differentiation (autograd).
    This script covers the fundamental building blocks you'll use in all
    subsequent tutorials.

TOPICS COVERED:
    - Creating tensors in different ways
    - Tensor operations and broadcasting
    - Automatic differentiation with autograd
    - Computing gradients
    - Understanding requires_grad

PREREQUISITES:
    - Basic Python knowledge
    - Basic understanding of derivatives (calculus)

LEARNING OBJECTIVES:
    - Create and manipulate PyTorch tensors
    - Understand tensor operations
    - Use autograd to compute gradients automatically
    - Understand the computational graph

TIME: ~15 minutes
==============================================================================
"""

import torch
import numpy as np

print("=" * 70)
print("PART 1: CREATING TENSORS")
print("=" * 70)

# ============================================================================
# 1.1 Creating tensors from Python lists
# ============================================================================
print("\n1.1 Creating tensors from Python lists:")

# Create a 1D tensor (vector)
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D tensor: {tensor_1d}")
print(f"Shape: {tensor_1d.shape}")
print(f"Data type: {tensor_1d.dtype}")

# Create a 2D tensor (matrix)
tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
print(f"\n2D tensor:\n{tensor_2d}")
print(f"Shape: {tensor_2d.shape}")  # (2 rows, 3 columns)
print(f"Data type: {tensor_2d.dtype}")

# ============================================================================
# 1.2 Creating tensors with specific data types
# ============================================================================
print("\n1.2 Creating tensors with specific data types:")

# By default, integers create torch.int64 and floats create torch.float32
float_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}")

# We can specify the data type explicitly
float32_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Float32 tensor: {float32_tensor}, dtype: {float32_tensor.dtype}")

# ============================================================================
# 1.3 Creating tensors from NumPy arrays
# ============================================================================
print("\n1.3 Creating tensors from NumPy arrays:")

numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy: {tensor_from_numpy}")

# Convert tensor back to NumPy
back_to_numpy = tensor_from_numpy.numpy()
print(f"Back to NumPy: {back_to_numpy}")

# ============================================================================
# 1.4 Creating special tensors
# ============================================================================
print("\n1.4 Creating special tensors:")

# Zeros (all elements are 0)
zeros = torch.zeros(2, 3)  # 2 rows, 3 columns
print(f"Zeros:\n{zeros}")

# Ones (all elements are 1)
ones = torch.ones(2, 3)
print(f"\nOnes:\n{ones}")

# Random tensor (values between 0 and 1)
random = torch.rand(2, 3)
print(f"\nRandom:\n{random}")

# Random normal distribution (mean=0, std=1)
randn = torch.randn(2, 3)
print(f"\nRandom normal:\n{randn}")

# Tensor with a range of values
arange = torch.arange(0, 10, 2)  # start, end (exclusive), step
print(f"\nArange: {arange}")

# Tensor with evenly spaced values
linspace = torch.linspace(0, 1, 5)  # start, end (inclusive), number of points
print(f"Linspace: {linspace}")

print("\n" + "=" * 70)
print("PART 2: TENSOR OPERATIONS")
print("=" * 70)

# ============================================================================
# 2.1 Basic arithmetic operations
# ============================================================================
print("\n2.1 Basic arithmetic operations:")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Addition
print(f"a + b = {a + b}")

# Subtraction
print(f"a - b = {a - b}")

# Element-wise multiplication
print(f"a * b = {a * b}")

# Element-wise division
print(f"a / b = {a / b}")

# Power
print(f"a ** 2 = {a ** 2}")

# ============================================================================
# 2.2 Matrix operations
# ============================================================================
print("\n2.2 Matrix operations:")

# Create two matrices
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

# Element-wise multiplication
print(f"Element-wise multiplication:\n{A * B}")

# Matrix multiplication (proper matrix product)
print(f"\nMatrix multiplication (A @ B):\n{A @ B}")
# Alternative: torch.mm(A, B) or torch.matmul(A, B)

# Transpose
print(f"\nTranspose of A:\n{A.T}")

# ============================================================================
# 2.3 Reshaping tensors
# ============================================================================
print("\n2.3 Reshaping tensors:")

x = torch.arange(12)  # [0, 1, 2, ..., 11]
print(f"Original: {x}, shape: {x.shape}")

# Reshape to 3x4 matrix
x_reshaped = x.reshape(3, 4)
print(f"\nReshaped to 3x4:\n{x_reshaped}")

# Reshape to 2x6 matrix
x_reshaped2 = x.reshape(2, 6)
print(f"\nReshaped to 2x6:\n{x_reshaped2}")

# Use -1 to infer dimension
x_reshaped3 = x.reshape(4, -1)  # PyTorch infers 3 automatically
print(f"\nReshaped to 4x-1 (becomes 4x3):\n{x_reshaped3}")

print("\n" + "=" * 70)
print("PART 3: AUTOMATIC DIFFERENTIATION (AUTOGRAD)")
print("=" * 70)

# ============================================================================
# 3.1 Understanding requires_grad
# ============================================================================
print("\n3.1 Understanding requires_grad:")

# When requires_grad=True, PyTorch tracks all operations on this tensor
# to automatically compute gradients
x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")

# When we perform operations on x, PyTorch builds a computational graph
y = x ** 2  # y = x^2
print(f"\ny = x^2 = {y}")
print(f"y.requires_grad = {y.requires_grad}")  # Automatically True

# ============================================================================
# 3.2 Computing gradients
# ============================================================================
print("\n3.2 Computing gradients:")

# We can compute dy/dx using backward()
y.backward()  # Computes gradients

# The gradient dy/dx is stored in x.grad
# dy/dx = d(x^2)/dx = 2x = 2*2 = 4
print(f"dy/dx at x=2: {x.grad}")

# ============================================================================
# 3.3 More complex example
# ============================================================================
print("\n3.3 More complex example:")

# Let's compute the gradient of a more complex function
x = torch.tensor([3.0], requires_grad=True)
# f(x) = 3x^2 + 2x + 1
y = 3 * x**2 + 2 * x + 1
print(f"\nf(x) = 3x^2 + 2x + 1")
print(f"f(3) = {y.item()}")

# Compute gradient: df/dx = 6x + 2
y.backward()
print(f"df/dx at x=3: {x.grad.item()}")
print(f"Expected (6*3 + 2 = 20): 20")

# ============================================================================
# 3.4 Gradient accumulation
# ============================================================================
print("\n3.4 Gradient accumulation:")

# Important: gradients accumulate! We need to zero them before computing new ones
x = torch.tensor([2.0], requires_grad=True)

# First computation
y = x ** 2
y.backward()
print(f"First backward: x.grad = {x.grad}")

# Second computation WITHOUT zeroing gradients
y = x ** 3
y.backward()
print(f"Second backward (accumulated): x.grad = {x.grad}")
# Gradient accumulated: 4 (from x^2) + 12 (from x^3) = 16

# ============================================================================
# 3.5 Zeroing gradients
# ============================================================================
print("\n3.5 Zeroing gradients:")

x = torch.tensor([2.0], requires_grad=True)

# First computation
y = x ** 2
y.backward()
print(f"First backward: x.grad = {x.grad}")

# Zero the gradients before second computation
x.grad.zero_()  # The underscore means in-place operation
print(f"After zeroing: x.grad = {x.grad}")

# Second computation
y = x ** 3
y.backward()
print(f"Second backward (after zeroing): x.grad = {x.grad}")

# ============================================================================
# 3.6 Detaching from the computation graph
# ============================================================================
print("\n3.6 Detaching from the computation graph:")

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
# Sometimes we want to use a value without tracking gradients
y_detached = y.detach()  # Removes from computational graph
print(f"y_detached.requires_grad = {y_detached.requires_grad}")

# ============================================================================
# 3.7 No gradient context
# ============================================================================
print("\n3.7 No gradient context:")

# When we don't need gradients (e.g., during evaluation), use torch.no_grad()
x = torch.tensor([2.0], requires_grad=True)

with torch.no_grad():
    y = x ** 2
    print(f"Inside no_grad context, y.requires_grad = {y.requires_grad}")

# This is useful for inference/evaluation to save memory and computation

print("\n" + "=" * 70)
print("PART 4: PRACTICAL EXAMPLE - SIMPLE DERIVATIVE")
print("=" * 70)

# ============================================================================
# 4.1 Finding minimum of a function using gradient descent
# ============================================================================
print("\n4.1 Finding minimum of f(x) = (x-3)^2:")

# Initialize x
x = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.1
num_steps = 20

print(f"{'Step':<6} {'x':<10} {'f(x)':<10} {'df/dx':<10}")
print("-" * 40)

for step in range(num_steps):
    # Compute function value: f(x) = (x-3)^2
    f = (x - 3) ** 2
    
    # Compute gradient
    if x.grad is not None:
        x.grad.zero_()  # Zero previous gradients
    f.backward()
    
    print(f"{step:<6} {x.item():<10.4f} {f.item():<10.4f} {x.grad.item():<10.4f}")
    
    # Update x using gradient descent: x = x - learning_rate * gradient
    with torch.no_grad():  # Don't track this operation
        x -= learning_rate * x.grad

print(f"\nFinal x: {x.item():.6f}")
print(f"Expected minimum at x=3")
print(f"Final f(x): {((x - 3) ** 2).item():.6f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key Takeaways:
1. Tensors are the fundamental data structure in PyTorch
2. PyTorch provides automatic differentiation (autograd)
3. Set requires_grad=True to track operations for gradient computation
4. Use .backward() to compute gradients
5. Always zero gradients before computing new ones
6. Use torch.no_grad() during evaluation to save memory

Next Steps:
- Proceed to 02_linear_regression_numpy.py to implement linear regression
- Understand how gradients are used in optimization
- Learn how to build machine learning models
""")
