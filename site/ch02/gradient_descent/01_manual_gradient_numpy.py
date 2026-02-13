"""
================================================================================
Level 1 - Example 1: Manual Gradient Descent with NumPy
================================================================================

LEARNING OBJECTIVES:
- Understand gradient descent from first principles
- Manually compute gradients using calculus
- Implement gradient descent without any ML libraries
- Visualize the optimization process

DIFFICULTY: ⭐ Beginner

TIME: 20-30 minutes

PREREQUISITES:
- Basic Python
- Elementary calculus (derivatives)
- Understanding of linear functions

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: Understanding the Problem
# ============================================================================
print("="*80)
print("GRADIENT DESCENT: LEARNING FROM SCRATCH")
print("="*80)
print("\nProblem: We want to fit a line y = w*x to our data")
print("Goal: Find the best weight 'w' that minimizes prediction error\n")

# ============================================================================
# PART 2: Prepare Training Data
# ============================================================================
# Let's create a simple dataset where the true relationship is y = 2*x
# This means the optimal weight w* = 2

X_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # Input features
y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)  # True labels (y = 2*x)

print("Training Data:")
print(f"X (inputs):  {X_train}")
print(f"y (targets): {y_train}")
print(f"\nTrue relationship: y = 2*x (so optimal w = 2.0)\n")

# ============================================================================
# PART 3: Initialize Model Parameter
# ============================================================================
# Start with a random guess for the weight
w = 0.0  # Initial weight (our model will learn the correct value)

print(f"Initial weight w = {w:.3f}")
print(f"Initial prediction for x=5: y = {w * 5:.3f} (should be 10.0)\n")

# ============================================================================
# PART 4: Define Model Components
# ============================================================================

def model_forward(x, weight):
    """
    Forward pass: compute predictions
    
    Our model: y_pred = w * x
    
    Args:
        x: input data (array or scalar)
        weight: current weight parameter
    
    Returns:
        predictions: model output
    """
    return weight * x


def compute_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) Loss Function
    
    Loss measures how wrong our predictions are:
    L(w) = (1/N) * Σ(y_pred - y_true)²
    
    Args:
        y_true: actual target values
        y_pred: predicted values
    
    Returns:
        loss: average squared error (scalar)
    """
    # Compute squared differences
    squared_errors = (y_pred - y_true) ** 2
    
    # Return the mean (average) error
    return np.mean(squared_errors)


def compute_gradient(x, y_true, y_pred):
    """
    Manually compute gradient dL/dw
    
    Mathematical Derivation:
    ------------------------
    Loss:     L(w) = (1/N) * Σ(w*x - y)²
    
    Derivative:  dL/dw = (1/N) * Σ 2*(w*x - y)*x
                       = (2/N) * Σ x*(w*x - y)
                       = (2/N) * Σ x*(y_pred - y_true)
    
    In code: gradient = mean(2 * x * (y_pred - y_true))
    
    Args:
        x: input data
        y_true: actual targets
        y_pred: predicted values
    
    Returns:
        gradient: dL/dw (tells us which direction to update w)
    """
    # Compute gradient using the formula derived above
    gradient = np.mean(2 * x * (y_pred - y_true))
    return gradient


# ============================================================================
# PART 5: Training Configuration
# ============================================================================
learning_rate = 0.01  # How big of a step to take in each iteration
n_iterations = 20     # Number of training iterations (epochs)

print("="*80)
print("TRAINING PROCESS")
print("="*80)
print(f"Learning rate: {learning_rate}")
print(f"Iterations: {n_iterations}\n")

# Lists to store training history for visualization
weight_history = [w]
loss_history = []

# ============================================================================
# PART 6: The Training Loop (Gradient Descent Algorithm)
# ============================================================================
print("Epoch | Weight (w) | Loss      | Gradient")
print("-" * 50)

for epoch in range(n_iterations):
    # STEP 1: Forward Pass
    # --------------------
    # Use current weight to make predictions
    y_pred = model_forward(X_train, w)
    
    # STEP 2: Compute Loss
    # --------------------
    # Measure how wrong our predictions are
    loss = compute_loss(y_train, y_pred)
    
    # STEP 3: Compute Gradient
    # -------------------------
    # Calculate dL/dw - tells us which direction to move w
    gradient = compute_gradient(X_train, y_train, y_pred)
    
    # STEP 4: Update Parameter (Gradient Descent Step)
    # -------------------------------------------------
    # Move w in the opposite direction of the gradient
    # This reduces the loss (makes predictions better)
    w = w - learning_rate * gradient
    
    # Store history for later visualization
    weight_history.append(w)
    loss_history.append(loss)
    
    # Print progress
    if epoch % 2 == 0:  # Print every 2 epochs
        print(f"{epoch+1:4d}  | {w:10.6f} | {loss:9.6f} | {gradient:9.6f}")

print("-" * 50)
print(f"\nFinal weight w = {w:.6f}")
print(f"Target weight = 2.0")
print(f"Error: {abs(w - 2.0):.6f}\n")

# ============================================================================
# PART 7: Test the Trained Model
# ============================================================================
print("="*80)
print("TESTING")
print("="*80)

# Test on training data
print("\nPredictions on training data:")
for x, y_true in zip(X_train, y_train):
    y_pred = model_forward(x, w)
    print(f"  x = {x:.0f} → y_pred = {y_pred:.3f}, y_true = {y_true:.0f}, error = {abs(y_pred-y_true):.3f}")

# Test on new data
print("\nPredictions on new data:")
test_inputs = [6, 7, 8]
for x in test_inputs:
    y_pred = model_forward(x, w)
    y_expected = 2 * x
    print(f"  x = {x:.0f} → y_pred = {y_pred:.3f}, y_expected = {y_expected:.0f}")

# ============================================================================
# PART 8: Visualization
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Subplot 1: Loss over iterations
axes[0].plot(loss_history, 'b-', linewidth=2, marker='o', markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Loss Decreases Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')  # Log scale to see improvement better

# Subplot 2: Weight convergence
axes[1].plot(weight_history, 'g-', linewidth=2, marker='s', markersize=4)
axes[1].axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Optimal weight (2.0)')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Weight (w)', fontsize=12)
axes[1].set_title('Weight Converges to Optimal Value', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Subplot 3: Final fit
axes[2].scatter(X_train, y_train, color='blue', s=100, label='Training data', zorder=3)
x_line = np.linspace(0, 6, 100)
y_line = model_forward(x_line, w)
axes[2].plot(x_line, y_line, 'r-', linewidth=2, label=f'Learned line (w={w:.3f})')
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('y', fontsize=12)
axes[2].set_title('Fitted Line vs Training Data', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/manual_gradient_descent.png', dpi=150)
print("\n✓ Plot saved as 'manual_gradient_descent.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# PART 9: Key Takeaways
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. GRADIENT DESCENT is an iterative optimization algorithm
   - Start with random parameters
   - Compute gradient (derivative of loss w.r.t. parameters)
   - Update parameters in opposite direction of gradient
   - Repeat until convergence

2. LEARNING RATE controls step size
   - Too large: might overshoot and diverge
   - Too small: slow convergence
   - Typical values: 0.001 to 0.1

3. LOSS FUNCTION measures prediction quality
   - MSE for regression problems
   - Different losses for different tasks

4. GRADIENT points in the direction of steepest increase
   - We move in opposite direction (-gradient) to decrease loss

5. CONVERGENCE happens when gradients become very small
   - Loss stops decreasing significantly
   - Parameters stabilize around optimal values
""")

print("="*80)
print("EXPERIMENT IDEAS:")
print("="*80)
print("""
Try modifying the code to explore:
1. Different learning rates (0.001, 0.1, 0.5) - what happens?
2. Different initial weights (5.0, -3.0) - does it still converge?
3. More training data points - does it improve accuracy?
4. Noisy data (add random noise to y) - how robust is gradient descent?
""")
print("="*80)
