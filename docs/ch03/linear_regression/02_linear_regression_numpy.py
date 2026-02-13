"""
==============================================================================
02_linear_regression_numpy.py
==============================================================================
DIFFICULTY: ⭐ (Beginner)

DESCRIPTION:
    Implement linear regression from scratch using only NumPy.
    This helps understand the mathematics before using PyTorch abstractions.

TOPICS COVERED:
    - Linear regression mathematical foundation
    - Gradient descent algorithm
    - Loss functions (Mean Squared Error)
    - Manual gradient computation
    - Training loop structure

PREREQUISITES:
    - Basic linear algebra (vectors, matrices)
    - Basic calculus (derivatives)
    - Tutorial 01 (helpful but not required)

LEARNING OBJECTIVES:
    - Understand the linear model: y = wx + b
    - Compute loss and gradients manually
    - Implement gradient descent from scratch
    - Visualize training progress

TIME: ~20 minutes
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH NUMPY")
print("=" * 70)

# ============================================================================
# PART 1: GENERATE SYNTHETIC DATA
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE SYNTHETIC DATA")
print("=" * 70)

# Set random seed for reproducibility
np.random.seed(42)

# True parameters we want to learn
TRUE_W = 2.0  # True slope
TRUE_B = 1.0  # True intercept

# Generate synthetic data
# We'll create points that roughly follow y = 2x + 1, with some noise
n_samples = 100
X = np.random.uniform(-10, 10, n_samples)  # Random x values between -10 and 10
noise = np.random.normal(0, 2, n_samples)  # Gaussian noise with std=2
y = TRUE_W * X + TRUE_B + noise           # y = 2x + 1 + noise

print(f"Generated {n_samples} data points")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")
print(f"\nFirst 5 samples:")
print(f"X[:5] = {X[:5]}")
print(f"y[:5] = {y[:5]}")

# ============================================================================
# PART 2: VISUALIZE THE DATA
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: VISUALIZE THE DATA")
print("=" * 70)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data points')
plt.plot(X, TRUE_W * X + TRUE_B, 'r-', linewidth=2, label=f'True line: y={TRUE_W}x+{TRUE_B}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data with True Relationship')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/02_data_visualization.png', dpi=100)
print("Saved data visualization to: 02_data_visualization.png")

# ============================================================================
# PART 3: DEFINE THE MODEL AND LOSS FUNCTION
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DEFINE THE MODEL AND LOSS FUNCTION")
print("=" * 70)

def predict(X, w, b):
    """
    Linear model: y_pred = w*X + b
    
    Args:
        X: Input features (n_samples,)
        w: Weight/slope parameter
        b: Bias/intercept parameter
    
    Returns:
        y_pred: Predictions (n_samples,)
    """
    return w * X + b

def compute_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss
    
    Loss = (1/n) * sum((y_true - y_pred)^2)
    
    This measures the average squared difference between
    true values and predictions.
    
    Args:
        y_true: True target values (n_samples,)
        y_pred: Predicted values (n_samples,)
    
    Returns:
        loss: Scalar MSE value
    """
    n = len(y_true)
    loss = (1 / n) * np.sum((y_true - y_pred) ** 2)
    return loss

# Test the functions
w_init = 0.0  # Initialize weight to 0
b_init = 0.0  # Initialize bias to 0

y_pred_init = predict(X, w_init, b_init)
initial_loss = compute_loss(y, y_pred_init)

print(f"Initial parameters: w={w_init}, b={b_init}")
print(f"Initial loss: {initial_loss:.4f}")

# ============================================================================
# PART 4: COMPUTE GRADIENTS
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: COMPUTE GRADIENTS")
print("=" * 70)

def compute_gradients(X, y_true, y_pred):
    """
    Compute gradients of MSE loss with respect to w and b
    
    Loss = (1/n) * sum((y_true - y_pred)^2)
         = (1/n) * sum((y_true - (w*X + b))^2)
    
    Partial derivatives:
    ∂Loss/∂w = (2/n) * sum((y_pred - y_true) * X)
    ∂Loss/∂b = (2/n) * sum(y_pred - y_true)
    
    Args:
        X: Input features (n_samples,)
        y_true: True target values (n_samples,)
        y_pred: Predicted values (n_samples,)
    
    Returns:
        grad_w: Gradient with respect to weight
        grad_b: Gradient with respect to bias
    """
    n = len(X)
    error = y_pred - y_true
    
    # Gradient of loss with respect to w
    grad_w = (2 / n) * np.sum(error * X)
    
    # Gradient of loss with respect to b
    grad_b = (2 / n) * np.sum(error)
    
    return grad_w, grad_b

# Test gradient computation
grad_w, grad_b = compute_gradients(X, y, y_pred_init)
print(f"Initial gradients:")
print(f"  grad_w = {grad_w:.4f}")
print(f"  grad_b = {grad_b:.4f}")
print("\nThe gradients tell us the direction to adjust parameters")
print("to decrease the loss.")

# ============================================================================
# PART 5: TRAINING LOOP (GRADIENT DESCENT)
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: TRAINING LOOP (GRADIENT DESCENT)")
print("=" * 70)

# Initialize parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01  # How big steps we take
n_epochs = 100        # Number of training iterations

# Storage for tracking progress
loss_history = []
w_history = [w]
b_history = [b]

print(f"Training Parameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Number of epochs: {n_epochs}")
print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12}")
print("-" * 50)

for epoch in range(n_epochs):
    # 1. Forward pass: compute predictions
    y_pred = predict(X, w, b)
    
    # 2. Compute loss
    loss = compute_loss(y, y_pred)
    loss_history.append(loss)
    
    # 3. Compute gradients
    grad_w, grad_b = compute_gradients(X, y, y_pred)
    
    # 4. Update parameters using gradient descent
    # New_param = Old_param - learning_rate * gradient
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    
    # Store history
    w_history.append(w)
    b_history.append(b)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss:<12.4f} {w:<12.4f} {b:<12.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Learned w: {w:.4f} (True: {TRUE_W})")
print(f"  Learned b: {b:.4f} (True: {TRUE_B})")
print(f"  Final loss: {loss_history[-1]:.4f}")
print(f"  Initial loss: {loss_history[0]:.4f}")
print(f"  Loss reduction: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%")

# ============================================================================
# PART 6: VISUALIZE TRAINING PROGRESS
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: VISUALIZE TRAINING PROGRESS")
print("=" * 70)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss over time
axes[0, 0].plot(loss_history, linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training Loss Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Parameter evolution
axes[0, 1].plot(w_history, label='w (slope)', linewidth=2)
axes[0, 1].axhline(y=TRUE_W, color='r', linestyle='--', label=f'True w={TRUE_W}')
axes[0, 1].plot(b_history, label='b (intercept)', linewidth=2)
axes[0, 1].axhline(y=TRUE_B, color='g', linestyle='--', label=f'True b={TRUE_B}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Parameter Value')
axes[0, 1].set_title('Parameter Evolution During Training')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Data with learned fit
axes[1, 0].scatter(X, y, alpha=0.5, label='Data')
axes[1, 0].plot(X, TRUE_W * X + TRUE_B, 'r-', linewidth=2, 
                label=f'True: y={TRUE_W}x+{TRUE_B}')
X_sorted = np.sort(X)
y_pred_final = predict(X_sorted, w, b)
axes[1, 0].plot(X_sorted, y_pred_final, 'g-', linewidth=2, 
                label=f'Learned: y={w:.2f}x+{b:.2f}')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Data with Learned Model')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Residuals (prediction errors)
y_pred_all = predict(X, w, b)
residuals = y - y_pred_all
axes[1, 1].scatter(y_pred_all, residuals, alpha=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals (y_true - y_pred)')
axes[1, 1].set_title('Residual Plot (should be randomly scattered)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/02_training_progress.png', dpi=100)
print("Saved training visualization to: 02_training_progress.png")
plt.show()

# ============================================================================
# PART 7: MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("PART 7: MAKE PREDICTIONS")
print("=" * 70)

# Make predictions on new data
X_new = np.array([-5, 0, 5])
y_pred_new = predict(X_new, w, b)

print("Making predictions on new data:")
for x_val, y_val in zip(X_new, y_pred_new):
    y_true_val = TRUE_W * x_val + TRUE_B  # True value (without noise)
    print(f"  X={x_val:6.1f} -> Predicted: {y_val:7.2f}, True: {y_true_val:7.2f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key Concepts:
1. Linear Model: y = wx + b
   - w (weight/slope): controls the steepness of the line
   - b (bias/intercept): controls where the line crosses y-axis

2. Loss Function: MSE = (1/n) * sum((y_true - y_pred)^2)
   - Measures how well our model fits the data
   - We want to minimize this value

3. Gradients: Partial derivatives of loss w.r.t. parameters
   - Tell us which direction to adjust parameters
   - Computed using calculus (chain rule)

4. Gradient Descent: Iterative optimization algorithm
   - Update rule: param = param - learning_rate * gradient
   - Learning rate controls step size

5. Training Loop:
   a) Forward pass: compute predictions
   b) Compute loss
   c) Compute gradients
   d) Update parameters
   e) Repeat

Why This Matters:
- This is the foundation of all neural network training
- PyTorch automates gradient computation, but the concepts remain
- Understanding the math helps debug and improve models

Next Steps:
- Proceed to 03_linear_regression_manual_pytorch.py
- See how PyTorch tensors simplify this code
- Still compute gradients manually to build intuition
""")
