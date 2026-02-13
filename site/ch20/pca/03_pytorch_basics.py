#!/usr/bin/env python3
# ==========================================================
# 01_pytorch_basics.py
# ==========================================================
# PYTORCH FUNDAMENTALS FOR BEGINNERS
#
# This script introduces PyTorch tensors and basic operations.
# It covers everything you need to know before diving into PCA
# and more advanced topics.
#
# WHAT YOU'LL LEARN:
# 1. Creating tensors (from lists, NumPy, random initialization)
# 2. Tensor properties (shape, dtype, device)
# 3. Basic operations (arithmetic, matrix multiplication)
# 4. Reshaping and indexing
# 5. CPU vs GPU operations
# 6. Autograd basics (automatic differentiation)
#
# WHY PYTORCH?
# - Dynamic computational graphs (easy to debug)
# - Pythonic and intuitive API
# - Excellent GPU acceleration
# - Industry standard for deep learning
# - Large community and ecosystem
#
# PREREQUISITES: Basic Python knowledge
# TIME: ~15 minutes to read and run
#
# Run: python 01_pytorch_basics.py
# ==========================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("WELCOME TO PYTORCH!")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60 + "\n")

# ==========================================================
# SECTION 1: CREATING TENSORS
# ==========================================================
print("SECTION 1: Creating Tensors")
print("-" * 60)

# A tensor is the fundamental data structure in PyTorch
# Think of it as a multi-dimensional array (like NumPy ndarray)
# But with GPU acceleration and automatic differentiation!

# 1.1: Create tensor from Python list
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print("\n1.1 From Python list:")
print(f"tensor_from_list = {tensor_from_list}")
print(f"  Shape: {tensor_from_list.shape}")  # (5,)
print(f"  Type: {tensor_from_list.dtype}")   # torch.int64 (default for integers)

# 1.2: Create 2D tensor (matrix)
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print("\n1.2 2D Tensor (Matrix):")
print(f"matrix =\n{matrix}")
print(f"  Shape: {matrix.shape}")  # (2, 3) = 2 rows, 3 columns

# 1.3: Create tensor from NumPy array
numpy_array = np.array([1.0, 2.0, 3.0])
tensor_from_numpy = torch.from_numpy(numpy_array)
print("\n1.3 From NumPy array:")
print(f"tensor_from_numpy = {tensor_from_numpy}")
print(f"  Type: {tensor_from_numpy.dtype}")  # torch.float64

# 1.4: Special tensors - zeros, ones, random
zeros = torch.zeros(2, 3)           # 2x3 matrix of zeros
ones = torch.ones(3, 4)             # 3x4 matrix of ones
random = torch.randn(2, 2)          # 2x2 matrix from standard normal N(0,1)
random_uniform = torch.rand(3, 3)   # 3x3 matrix from uniform [0, 1)

print("\n1.4 Special tensors:")
print(f"zeros (2x3):\n{zeros}")
print(f"randn (2x2, standard normal):\n{random}")

# 1.5: Create tensor with specific dtype
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print("\n1.5 Specify dtype:")
print(f"float_tensor = {float_tensor}")
print(f"  Type: {float_tensor.dtype}")  # torch.float32

# 1.6: Identity matrix and eye
identity = torch.eye(3)  # 3x3 identity matrix
print("\n1.6 Identity matrix (3x3):")
print(identity)

# 1.7: Create sequence (like range)
sequence = torch.arange(0, 10, 2)  # Start, end, step
print("\n1.7 Sequence (arange):")
print(f"sequence = {sequence}")  # [0, 2, 4, 6, 8]

# 1.8: Linspace (evenly spaced values)
linspace = torch.linspace(0, 1, 5)  # 5 values from 0 to 1
print("\n1.8 Linspace:")
print(f"linspace = {linspace}")  # [0.0000, 0.2500, 0.5000, 0.7500, 1.0000]

# ==========================================================
# SECTION 2: TENSOR PROPERTIES
# ==========================================================
print("\n\nSECTION 2: Tensor Properties")
print("-" * 60)

# Every tensor has important properties you should know
example_tensor = torch.randn(3, 4, 5)  # 3D tensor

print(f"\nexample_tensor shape: {example_tensor.shape}")  # (3, 4, 5)
print(f"  Size: {example_tensor.size()}")                 # Same as shape
print(f"  Number of dimensions: {example_tensor.ndim}")   # 3
print(f"  Total elements: {example_tensor.numel()}")      # 3*4*5 = 60
print(f"  Data type: {example_tensor.dtype}")             # torch.float32
print(f"  Device: {example_tensor.device}")               # cpu or cuda:0

# Memory layout
print(f"  Is contiguous: {example_tensor.is_contiguous()}")  # True
print(f"  Memory size: {example_tensor.element_size() * example_tensor.numel()} bytes")

# ==========================================================
# SECTION 3: BASIC OPERATIONS
# ==========================================================
print("\n\nSECTION 3: Basic Operations")
print("-" * 60)

# 3.1: Element-wise operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("\n3.1 Element-wise operations:")
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")      # Element-wise addition
print(f"a - b = {a - b}")      # Element-wise subtraction
print(f"a * b = {a * b}")      # Element-wise multiplication
print(f"a / b = {a / b}")      # Element-wise division
print(f"a ** 2 = {a ** 2}")    # Element-wise power

# 3.2: Matrix operations
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

print("\n3.2 Matrix operations:")
print(f"A =\n{A}")
print(f"B =\n{B}")

# Matrix multiplication: A @ B
print(f"\nMatrix multiply (A @ B):\n{A @ B}")
print(f"Same as torch.mm(A, B):\n{torch.mm(A, B)}")

# Element-wise multiply (NOT matrix multiply!)
print(f"\nElement-wise multiply (A * B):\n{A * B}")

# Transpose
print(f"\nTranspose (A.T):\n{A.T}")

# 3.3: Reduction operations
data = torch.randn(3, 4)
print("\n3.3 Reduction operations:")
print(f"data =\n{data}")
print(f"\nMean: {data.mean()}")                    # Overall mean
print(f"Mean along rows (dim=0): {data.mean(dim=0)}")  # Mean of each column
print(f"Mean along cols (dim=1): {data.mean(dim=1)}")  # Mean of each row
print(f"Sum: {data.sum()}")
print(f"Max: {data.max()}")
print(f"Min: {data.min()}")
print(f"Standard deviation: {data.std()}")

# 3.4: Dot product and matrix-vector multiplication
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
M = torch.randn(3, 3)

print("\n3.4 Dot product and matrix-vector:")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"Dot product (v1 · v2): {torch.dot(v1, v2)}")  # Scalar
print(f"Matrix-vector (M @ v1): {M @ v1}")            # Vector

# ==========================================================
# SECTION 4: RESHAPING AND INDEXING
# ==========================================================
print("\n\nSECTION 4: Reshaping and Indexing")
print("-" * 60)

# 4.1: Reshaping
original = torch.arange(12)  # [0, 1, 2, ..., 11]
print("\n4.1 Reshaping:")
print(f"original (12,): {original}")

reshaped_view = original.view(3, 4)  # Reshape to 3x4 (must be contiguous)
print(f"\nview(3, 4):\n{reshaped_view}")

reshaped_reshape = original.reshape(2, 6)  # Reshape to 2x6 (copies if needed)
print(f"\nreshape(2, 6):\n{reshaped_reshape}")

# -1 means "infer this dimension"
auto_reshape = original.view(4, -1)  # 4 rows, infer columns (3)
print(f"\nview(4, -1):\n{auto_reshape}")

# Flatten to 1D
flattened = reshaped_view.flatten()
print(f"\nflatten(): {flattened}")

# 4.2: Indexing and slicing
matrix = torch.arange(20).reshape(4, 5)
print("\n4.2 Indexing and slicing:")
print(f"matrix (4x5):\n{matrix}")

print(f"\nmatrix[0]: {matrix[0]}")           # First row
print(f"matrix[:, 0]: {matrix[:, 0]}")       # First column
print(f"matrix[1, 2]: {matrix[1, 2]}")       # Element at (1, 2)
print(f"matrix[1:3, 2:4]:\n{matrix[1:3, 2:4]}")  # 2x2 submatrix

# Boolean indexing
mask = matrix > 10
print(f"\nmask (matrix > 10):\n{mask}")
print(f"matrix[mask]: {matrix[mask]}")  # Elements > 10

# 4.3: Concatenation and stacking
a = torch.ones(2, 3)
b = torch.zeros(2, 3)

print("\n4.3 Concatenation:")
print(f"a =\n{a}")
print(f"b =\n{b}")

concat_rows = torch.cat([a, b], dim=0)  # Stack vertically (4x3)
print(f"\ncat([a, b], dim=0):\n{concat_rows}")

concat_cols = torch.cat([a, b], dim=1)  # Stack horizontally (2x6)
print(f"\ncat([a, b], dim=1):\n{concat_cols}")

stacked = torch.stack([a, b])  # New dimension (2, 2, 3)
print(f"\nstack([a, b]): shape {stacked.shape}")

# ==========================================================
# SECTION 5: CPU VS GPU
# ==========================================================
print("\n\nSECTION 5: CPU vs GPU Operations")
print("-" * 60)

# Default: tensors are on CPU
cpu_tensor = torch.randn(3, 3)
print(f"\nDefault device: {cpu_tensor.device}")  # cpu

# Check GPU availability
if torch.cuda.is_available():
    print("✓ CUDA is available! Moving tensors to GPU...")
    
    # Move to GPU: .to('cuda') or .cuda()
    gpu_tensor = cpu_tensor.to('cuda')
    # Alternative: gpu_tensor = cpu_tensor.cuda()
    
    print(f"GPU device: {gpu_tensor.device}")  # cuda:0
    
    # Create tensor directly on GPU
    direct_gpu = torch.randn(3, 3, device='cuda')
    print(f"Direct GPU creation: {direct_gpu.device}")
    
    # GPU operations are MUCH faster for large tensors!
    large_cpu = torch.randn(1000, 1000)
    large_gpu = large_cpu.to('cuda')
    
    # Example: Matrix multiplication on GPU
    result_gpu = large_gpu @ large_gpu
    print(f"GPU computation result shape: {result_gpu.shape}")
    
    # Move back to CPU: .to('cpu') or .cpu()
    result_cpu = result_gpu.cpu()
    print(f"Moved back to CPU: {result_cpu.device}")
    
    # IMPORTANT: Can't mix CPU and GPU tensors!
    # This would error: cpu_tensor + gpu_tensor
    # Must move one to match the other's device
    
else:
    print("✗ CUDA not available. Running on CPU only.")
    print("  To use GPU: Install CUDA and GPU-enabled PyTorch")

# ==========================================================
# SECTION 6: AUTOGRAD - AUTOMATIC DIFFERENTIATION
# ==========================================================
print("\n\nSECTION 6: Autograd (Automatic Differentiation)")
print("-" * 60)

# This is PyTorch's "superpower" for deep learning!
# It automatically computes gradients for backpropagation.

print("\n6.1 Basic gradient computation:")

# Create tensor with requires_grad=True to track operations
x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"requires_grad: {x.requires_grad}")

# Define a function: y = 3x² + 2x + 1
y = 3 * x**2 + 2 * x + 1
print(f"y = 3x² + 2x + 1 = {y}")

# Compute gradient dy/dx
# Mathematically: dy/dx = 6x + 2
# At x=2: dy/dx = 6(2) + 2 = 14
y.backward()  # Compute gradients

print(f"dy/dx (gradient): {x.grad}")  # Should be 14.0
print(f"Expected: 6*2 + 2 = 14 ✓")

# 6.2: More complex example
print("\n6.2 Matrix gradients:")

# Input matrix
W = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
x_input = torch.randn(3)

# Forward pass: y = W @ x + b
y_out = W @ x_input + b
loss = y_out.sum()  # Scalar loss (required for backward)

print(f"W shape: {W.shape}")
print(f"loss: {loss.item():.4f}")

# Backward pass
loss.backward()

print(f"W.grad shape: {W.grad.shape}")  # Same as W.shape
print(f"b.grad shape: {b.grad.shape}")  # Same as b.shape

# These gradients are used to update W and b during training!

# 6.3: Important notes about autograd
print("\n6.3 Autograd important notes:")
print("  • Only works with floating-point tensors")
print("  • .backward() can only be called on scalars")
print("  • Gradients accumulate! Use .zero_grad() to reset")
print("  • Use torch.no_grad() for inference (faster, less memory)")

# Example: inference without gradients
with torch.no_grad():
    inference_result = W @ x_input + b
    print(f"  Inference (no grad tracking): {inference_result}")

# ==========================================================
# SECTION 7: PRACTICAL EXAMPLE - LINEAR REGRESSION
# ==========================================================
print("\n\nSECTION 7: Practical Example - Linear Regression")
print("-" * 60)

# Let's fit a line to data: y = 2x + 1 (with noise)
print("\nGenerating synthetic data: y = 2x + 1 + noise")

# True parameters
true_w = 2.0
true_b = 1.0

# Generate data
torch.manual_seed(42)
x_data = torch.linspace(0, 10, 100).unsqueeze(1)  # (100, 1)
y_data = true_w * x_data + true_b + 0.5 * torch.randn(100, 1)  # Add noise

# Initialize learnable parameters
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Training parameters
learning_rate = 0.01
n_epochs = 50

print(f"Initial w: {w.item():.4f}, b: {b.item():.4f}")
print(f"Target:  w: {true_w:.4f}, b: {true_b:.4f}")
print(f"\nTraining for {n_epochs} epochs...")

losses = []

for epoch in range(n_epochs):
    # Forward pass: y_pred = wx + b
    y_pred = x_data @ w + b
    
    # Compute loss (Mean Squared Error)
    loss = ((y_pred - y_data) ** 2).mean()
    losses.append(loss.item())
    
    # Backward pass
    loss.backward()
    
    # Update parameters (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: loss={loss.item():.4f}, "
              f"w={w.item():.4f}, b={b.item():.4f}")

print(f"\nFinal w: {w.item():.4f} (target: {true_w:.4f})")
print(f"Final b: {b.item():.4f} (target: {true_b:.4f})")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Data and fitted line
ax1.scatter(x_data.numpy(), y_data.numpy(), s=10, alpha=0.5, label='Data')
with torch.no_grad():
    y_pred_final = x_data @ w + b
ax1.plot(x_data.numpy(), y_pred_final.numpy(), 'r-', linewidth=2, 
         label=f'Fitted: y={w.item():.2f}x+{b.item():.2f}')
ax1.plot(x_data.numpy(), (true_w * x_data + true_b).numpy(), 'g--', 
         linewidth=2, label=f'True: y={true_w:.2f}x+{true_b:.2f}')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Linear Regression with PyTorch', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training loss
ax2.plot(losses, linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (MSE)', fontsize=12)
ax2.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pytorch_basics_regression.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: pytorch_basics_regression.png")

# ==========================================================
# SECTION 8: USEFUL PYTORCH FUNCTIONS
# ==========================================================
print("\n\nSECTION 8: Useful PyTorch Functions")
print("-" * 60)

print("\nMath functions:")
x = torch.tensor([0.0, 1.0, 2.0, 3.0])
print(f"x = {x}")
print(f"torch.sin(x) = {torch.sin(x)}")
print(f"torch.exp(x) = {torch.exp(x)}")
print(f"torch.log(x+1) = {torch.log(x + 1)}")
print(f"torch.sqrt(x) = {torch.sqrt(x)}")
print(f"torch.abs(x-2) = {torch.abs(x - 2)}")

print("\nComparison:")
print(f"torch.max(x, torch.tensor(1.5)): {torch.max(x, torch.tensor(1.5))}")
print(f"torch.clamp(x, 1, 2): {torch.clamp(x, 1, 2)}")  # Clip values

print("\nLinear algebra:")
A = torch.randn(3, 3)
print(f"Matrix norm: {torch.norm(A):.4f}")
print(f"Matrix determinant: {torch.det(A):.4f}")
eigenvalues, eigenvectors = torch.linalg.eig(A)
print(f"Eigenvalues shape: {eigenvalues.shape}")

# ==========================================================
# SUMMARY
# ==========================================================
print("\n\n" + "=" * 60)
print("SUMMARY: PYTORCH BASICS")
print("=" * 60)
print("✓ You now know:")
print("  1. How to create tensors (from lists, NumPy, random)")
print("  2. Tensor operations (arithmetic, matrix multiply)")
print("  3. Reshaping and indexing")
print("  4. CPU vs GPU computation")
print("  5. Automatic differentiation (autograd)")
print("  6. Training a simple model")
print("\n✓ Next steps:")
print("  • Practice with these operations")
print("  • Move on to 02_pca_2d_pytorch.py")
print("  • Explore torch.nn for neural networks")
print("=" * 60)

plt.show()
