"""
================================================================================
01_introduction.py - Introduction to PyTorch Tensors and Autograd
================================================================================

LEARNING OBJECTIVES:
- Understand PyTorch tensors (the fundamental data structure)
- Learn about automatic differentiation (autograd)
- Implement a simple gradient descent optimization
- Understand the basics of neural network training

PREREQUISITES:
- Basic Python programming
- Understanding of derivatives and gradients
- numpy basics (helpful but not required)

TIME TO COMPLETE: ~30 minutes

DIFFICULTY: ⭐☆☆☆☆ (Beginner)
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("PART 1: PYTORCH TENSORS - The Building Blocks")
print("="*80)

# ============================================================================
# 1.1: Creating Tensors
# ============================================================================
print("\n1.1: Creating Tensors")
print("-" * 40)

# Create tensors in various ways
tensor_from_list = torch.tensor([1.0, 2.0, 3.0])  # From Python list
tensor_from_numpy = torch.from_numpy(np.array([4.0, 5.0, 6.0]))  # From numpy
tensor_zeros = torch.zeros(3)  # All zeros
tensor_ones = torch.ones(3)    # All ones
tensor_random = torch.randn(3)  # Random from standard normal distribution

print(f"From list:   {tensor_from_list}")
print(f"From numpy:  {tensor_from_numpy}")
print(f"Zeros:       {tensor_zeros}")
print(f"Ones:        {tensor_ones}")
print(f"Random:      {tensor_random}")

# Shape and datatype
print(f"\nShape: {tensor_from_list.shape}")  # torch.Size([3])
print(f"Dtype: {tensor_from_list.dtype}")   # torch.float32
print(f"Device: {tensor_from_list.device}") # cpu or cuda

# ============================================================================
# 1.2: Tensor Operations
# ============================================================================
print("\n1.2: Tensor Operations")
print("-" * 40)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
add_result = x + y                    # [5.0, 7.0, 9.0]
multiply_result = x * y               # [4.0, 10.0, 18.0]
dot_product = torch.dot(x, y)         # 1*4 + 2*5 + 3*6 = 32

print(f"x + y = {add_result}")
print(f"x * y = {multiply_result}")
print(f"x · y = {dot_product.item()}")  # .item() extracts Python scalar

# Matrix operations
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
b = torch.tensor([1.0, 2.0])

matrix_vector_product = torch.matmul(A, b)  # or A @ b
print(f"\nMatrix-vector product:\n{matrix_vector_product}")


print("\n" + "="*80)
print("PART 2: AUTOMATIC DIFFERENTIATION (AUTOGRAD)")
print("="*80)

# ============================================================================
# 2.1: Computing Gradients
# ============================================================================
print("\n2.1: Computing Gradients")
print("-" * 40)

# Create a tensor and tell PyTorch to track operations on it
# requires_grad=True means we want to compute gradients with respect to this tensor
x = torch.tensor([2.0], requires_grad=True)

# Define a function: y = x^2
y = x ** 2

print(f"x = {x.item()}")
print(f"y = x^2 = {y.item()}")

# Compute gradient dy/dx
y.backward()  # This computes all gradients automatically!

# The gradient is stored in x.grad
print(f"dy/dx at x=2 is: {x.grad.item()}")
print(f"Mathematical verification: dy/dx = 2x = 2*2 = 4 ✓")

# ============================================================================
# 2.2: More Complex Function
# ============================================================================
print("\n2.2: More Complex Function")
print("-" * 40)

# Reset: create new tensor
x = torch.tensor([3.0], requires_grad=True)

# More complex function: y = 3x^2 + 2x + 1
y = 3 * x**2 + 2 * x + 1

print(f"x = {x.item()}")
print(f"y = 3x^2 + 2x + 1 = {y.item()}")

y.backward()
print(f"dy/dx at x=3 is: {x.grad.item()}")
print(f"Mathematical verification: dy/dx = 6x + 2 = 6*3 + 2 = 20 ✓")


print("\n" + "="*80)
print("PART 3: GRADIENT DESCENT OPTIMIZATION")
print("="*80)

# ============================================================================
# 3.1: Finding Minimum of a Function
# ============================================================================
print("\n3.1: Finding Minimum of y = (x - 3)^2")
print("-" * 40)

# We want to find x that minimizes y = (x - 3)^2
# The minimum is at x = 3, where y = 0

# Start with initial guess
x = torch.tensor([0.0], requires_grad=True)  # Starting point: x = 0
learning_rate = 0.1  # Step size for each update
num_iterations = 50

# Store history for plotting
x_history = []
y_history = []

print(f"Initial x: {x.item():.4f}")

for iteration in range(num_iterations):
    # Forward pass: compute the function value
    y = (x - 3) ** 2  # Function to minimize
    
    # Store history
    x_history.append(x.item())
    y_history.append(y.item())
    
    # Backward pass: compute gradient
    y.backward()  # Computes dy/dx
    
    # Update step: move in the direction that reduces y
    # x_new = x_old - learning_rate * gradient
    with torch.no_grad():  # We don't need gradients for the update step
        x -= learning_rate * x.grad  # This is gradient descent!
    
    # Zero the gradient for next iteration (important!)
    x.grad.zero_()
    
    # Print progress every 10 iterations
    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration+1:2d}: x = {x.item():.4f}, y = {y.item():.4f}")

print(f"\nFinal x: {x.item():.4f}")
print(f"Target x: 3.0000")
print(f"Final y: {y.item():.6f}")

# ============================================================================
# 3.2: Visualize Optimization Path
# ============================================================================
print("\n3.2: Visualizing the Optimization")
print("-" * 40)

# Create figure
plt.figure(figsize=(12, 5))

# Plot 1: Function and optimization path
plt.subplot(1, 2, 1)
x_range = np.linspace(-1, 6, 100)
y_range = (x_range - 3) ** 2

plt.plot(x_range, y_range, 'b-', linewidth=2, label='y = (x-3)²')
plt.plot(x_history, y_history, 'ro-', linewidth=1, markersize=4, 
         label='Optimization path', alpha=0.6)
plt.plot(x_history[0], y_history[0], 'go', markersize=10, label='Start')
plt.plot(x_history[-1], y_history[-1], 'r*', markersize=15, label='End')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Function and Optimization Path', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Convergence (how y decreases over time)
plt.subplot(1, 2, 2)
plt.plot(y_history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss (y value)', fontsize=12)
plt.title('Convergence: Loss over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see convergence better

plt.tight_layout()
plt.savefig('/home/claude/pytorch_logistic_regression_tutorial/01_basics/optimization_demo.png', 
            dpi=150, bbox_inches='tight')
print("Plot saved as: optimization_demo.png")


print("\n" + "="*80)
print("PART 4: KEY TAKEAWAYS")
print("="*80)

print("""
1. TENSORS are PyTorch's fundamental data structure
   - Like NumPy arrays but with GPU support and autograd
   - Created with torch.tensor(), torch.zeros(), torch.randn(), etc.

2. AUTOGRAD (Automatic Differentiation)
   - Set requires_grad=True to track operations
   - Call .backward() to compute all gradients
   - Gradients stored in .grad attribute

3. GRADIENT DESCENT
   - Iteratively update parameters to minimize a function
   - Update rule: x_new = x_old - learning_rate * gradient
   - Always call .zero_grad() between iterations!

4. TORCH VS NUMPY
   - torch.tensor vs np.array
   - PyTorch can run on GPU (CUDA)
   - PyTorch has built-in automatic differentiation
""")


print("\n" + "="*80)
print("EXERCISES (Try These!)")
print("="*80)

print("""
1. EASY: Change the learning rate to 0.01 and 0.5. 
   How does it affect convergence speed?

2. MEDIUM: Try minimizing y = x^4 - 4x^2 + 4
   This function has two local minima!
   Starting from x=0 and x=3, what do you get?

3. MEDIUM: Implement gradient descent for 2D function:
   y = (x1 - 2)^2 + (x2 + 1)^2
   Minimum should be at (2, -1)

4. HARD: Add momentum to gradient descent:
   velocity = 0.9 * velocity + learning_rate * gradient
   x = x - velocity
   Compare convergence with standard gradient descent.
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
You've learned the basics of PyTorch tensors and gradient descent!

Next tutorial: 02_simple_binary_classification.py
- Apply these concepts to a real machine learning problem
- Implement logistic regression from scratch
- Train a binary classifier

Ready to continue? Run:
    python 02_simple_binary_classification.py
""")
print("="*80)
