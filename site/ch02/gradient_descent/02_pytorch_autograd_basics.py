"""
================================================================================
Level 1 - Example 2: PyTorch Autograd Basics
================================================================================

LEARNING OBJECTIVES:
- Understand PyTorch's automatic differentiation (autograd)
- Learn about computational graphs
- Compare manual vs automatic gradient computation
- Master requires_grad and backward() methods

DIFFICULTY: ⭐ Beginner

TIME: 25-35 minutes

PREREQUISITES:
- Completed Example 01 (Manual Gradient Descent)
- Basic understanding of gradients

================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("PYTORCH AUTOGRAD: AUTOMATIC DIFFERENTIATION")
print("="*80)

# ============================================================================
# PART 1: Introduction to Tensors and requires_grad
# ============================================================================
print("\n" + "="*80)
print("PART 1: TENSORS WITH GRADIENT TRACKING")
print("="*80)

# Regular tensor (no gradient tracking)
x_no_grad = torch.tensor([1.0, 2.0, 3.0])
print(f"\nRegular tensor: {x_no_grad}")
print(f"requires_grad: {x_no_grad.requires_grad}")  # False by default

# Tensor with gradient tracking enabled
x_with_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"\nTensor with gradient tracking: {x_with_grad}")
print(f"requires_grad: {x_with_grad.requires_grad}")  # True
print("\n→ When requires_grad=True, PyTorch tracks all operations on this tensor!")

# ============================================================================
# PART 2: Simple Example - Computing Gradients
# ============================================================================
print("\n" + "="*80)
print("PART 2: SIMPLE GRADIENT COMPUTATION")
print("="*80)

# Let's compute the gradient of y = 3x² + 2x + 1 at x = 2
print("\nFunction: y = 3x² + 2x + 1")
print("Derivative: dy/dx = 6x + 2")
print("At x = 2: dy/dx = 6(2) + 2 = 14")

# PyTorch way
x = torch.tensor(2.0, requires_grad=True)
y = 3 * x**2 + 2 * x + 1

print(f"\nUsing PyTorch autograd:")
print(f"x = {x.item()}")
print(f"y = {y.item()}")

# Compute gradient dy/dx
y.backward()  # This populates x.grad with dy/dx

print(f"Computed gradient dy/dx = {x.grad.item()}")
print(f"Expected gradient = 14")
print(f"✓ Match!" if abs(x.grad.item() - 14) < 0.001 else "✗ Error!")

# ============================================================================
# PART 3: Understanding the Computational Graph
# ============================================================================
print("\n" + "="*80)
print("PART 3: COMPUTATIONAL GRAPH")
print("="*80)

print("""
When you perform operations on tensors with requires_grad=True,
PyTorch builds a computational graph:

    x (requires_grad=True)
    ↓
    x² (intermediate operation)
    ↓
    3*x² (intermediate operation)
    ↓
    3*x² + 2x (intermediate operation)
    ↓
    y = 3*x² + 2x + 1 (final output)

When you call y.backward():
- PyTorch traverses this graph backwards
- Applies chain rule automatically
- Accumulates gradients in x.grad
""")

# ============================================================================
# PART 4: Gradient Descent with PyTorch - Same Problem as Example 01
# ============================================================================
print("\n" + "="*80)
print("PART 4: LINEAR REGRESSION WITH AUTOGRAD")
print("="*80)

# Same dataset as Example 01: y = 2*x
X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

print("\nProblem: Fit y = w*x to find w ≈ 2")
print(f"Training data: X = {X.numpy()}")
print(f"               y = {y.numpy()}")

# Initialize weight with gradient tracking
w = torch.tensor(0.0, requires_grad=True)
print(f"\nInitial weight: w = {w.item():.3f}")

# Training hyperparameters
learning_rate = 0.01
n_epochs = 100

# Track progress
weight_history = [w.item()]
loss_history = []

print("\n" + "-"*60)
print("Training with PyTorch Autograd:")
print("-"*60)
print("Epoch | Weight    | Loss      | Gradient")
print("-"*60)

for epoch in range(n_epochs):
    # STEP 1: Forward pass
    y_pred = w * X
    
    # STEP 2: Compute loss
    loss = torch.mean((y_pred - y) ** 2)
    
    # STEP 3: Backward pass - compute gradients automatically!
    # IMPORTANT: We need to zero gradients first because they accumulate
    if w.grad is not None:
        w.grad.zero_()
    
    loss.backward()  # This computes dL/dw automatically!
    
    # STEP 4: Update weight
    # IMPORTANT: Use torch.no_grad() to prevent tracking this operation
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # Store history
    weight_history.append(w.item())
    loss_history.append(loss.item())
    
    # Print progress
    if epoch % 10 == 0:
        print(f"{epoch+1:4d}  | {w.item():9.6f} | {loss.item():9.6f} | {w.grad.item():9.6f}")

print("-"*60)
print(f"Final weight: w = {w.item():.6f}")
print(f"Target: 2.0")
print(f"Error: {abs(w.item() - 2.0):.6f}")

# ============================================================================
# PART 5: Comparing Manual vs Autograd
# ============================================================================
print("\n" + "="*80)
print("PART 5: MANUAL vs AUTOGRAD COMPARISON")
print("="*80)

print("""
MANUAL GRADIENT (from Example 01):
-----------------------------------
def compute_gradient(x, y_true, y_pred):
    # Manually derived: dL/dw = mean(2 * x * (y_pred - y_true))
    gradient = np.mean(2 * x * (y_pred - y_true))
    return gradient

AUTOGRAD (this example):
------------------------
loss = torch.mean((y_pred - y) ** 2)
loss.backward()  # Automatically computes all gradients!
gradient = w.grad  # PyTorch fills this in for us

ADVANTAGES OF AUTOGRAD:
✓ No manual calculus needed
✓ Handles complex models automatically
✓ Fewer bugs (no derivative mistakes)
✓ Efficient implementation
✓ Easy to modify model architecture

WHEN TO USE MANUAL:
• Learning/understanding gradients
• Simple educational examples
• Very custom operations
""")

# ============================================================================
# PART 6: Important Concepts - Gradient Accumulation
# ============================================================================
print("\n" + "="*80)
print("PART 6: GRADIENT ACCUMULATION (COMMON PITFALL!)")
print("="*80)

print("\nGradients ACCUMULATE by default. Watch what happens:\n")

# Example showing accumulation
x = torch.tensor(3.0, requires_grad=True)

# First backward pass
y1 = x ** 2
y1.backward()
print(f"After 1st backward: x.grad = {x.grad.item()}")  # Should be 6

# Second backward pass WITHOUT zeroing
x.grad.zero_()  # Comment this line to see accumulation!
y2 = x ** 3
y2.backward()
print(f"After 2nd backward: x.grad = {x.grad.item()}")  # Should be 27

print("""
IMPORTANT: Always zero gradients before backward pass!
    if w.grad is not None:
        w.grad.zero_()
    loss.backward()
""")

# ============================================================================
# PART 7: Detaching from Computational Graph
# ============================================================================
print("\n" + "="*80)
print("PART 7: DETACHING AND NO_GRAD")
print("="*80)

x = torch.tensor(5.0, requires_grad=True)
y = x ** 2

# Method 1: detach() - removes from graph, keeps value
y_detached = y.detach()
print(f"\nOriginal y.requires_grad: {y.requires_grad}")
print(f"Detached y.requires_grad: {y_detached.requires_grad}")

# Method 2: torch.no_grad() - temporarily disable tracking
with torch.no_grad():
    z = x ** 2
    print(f"\nInside no_grad, z.requires_grad: {z.requires_grad}")

print("""
USE CASES:
- detach(): When you want to use a value without gradients
- no_grad(): For inference or parameter updates
- Both prevent unnecessary graph building (saves memory!)
""")

# ============================================================================
# PART 8: Visualization
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Loss convergence
axes[0].plot(loss_history, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Loss Convergence (PyTorch Autograd)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Plot 2: Weight convergence
axes[1].plot(weight_history, 'g-', linewidth=2)
axes[1].axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Target (2.0)')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Weight (w)', fontsize=12)
axes[1].set_title('Weight Convergence to Optimal Value', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/pytorch_autograd.png', dpi=150)
print("\n✓ Plot saved as 'pytorch_autograd.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# PART 9: Practice Exercise
# ============================================================================
print("\n" + "="*80)
print("PRACTICE EXERCISE")
print("="*80)
print("""
Try implementing gradient descent for these functions:

1. Quadratic: y = ax² + bx + c
   - Use 3 parameters: a, b, c (all with requires_grad=True)
   - Fit to data: x=[1,2,3,4], y=[3,8,15,24]

2. Non-linear: y = a*sin(b*x) + c
   - Fit sine wave to noisy data
   - Experiment with learning rates

3. Multi-dimensional: z = ax + by + c
   - 2 input features (x, y)
   - 3 parameters (a, b, c)
""")

# ============================================================================
# PART 10: Key Takeaways
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. AUTOGRAD eliminates manual gradient computation
   - Set requires_grad=True on parameters
   - Build computational graph through operations
   - Call .backward() to compute all gradients

2. ALWAYS ZERO GRADIENTS before backward pass
   - Gradients accumulate by default
   - Use: w.grad.zero_() or optimizer.zero_grad()

3. USE torch.no_grad() for parameter updates
   - Prevents tracking operations we don't need to differentiate
   - Essential for inference (testing/prediction)
   - Saves memory and computation

4. LEAF TENSORS store gradients
   - Only tensors created with requires_grad=True
   - Intermediate tensors don't store gradients by default

5. COMPUTATION GRAPH is dynamic
   - Built during forward pass
   - Cleared after backward pass
   - Rebuilt on next forward pass
""")

print("="*80)
print("NEXT STEPS")
print("="*80)
print("""
You now understand automatic differentiation!
Next, move to Example 03 to apply this to a real regression problem.
""")
print("="*80)
