"""
==============================================================================
Tutorial 01: Linear Regression with Pure NumPy
==============================================================================
DIFFICULTY: ⭐ Beginner

WHAT YOU'LL LEARN:
- The mathematics behind neural networks (forward pass, loss, gradients)
- Manual implementation of gradient descent
- How weights and biases are updated

PREREQUISITES:
- Basic Python
- Understanding of linear functions (y = wx + b)
- Basic calculus (derivatives)

KEY CONCEPTS:
- Forward propagation
- Mean Squared Error (MSE) loss
- Backpropagation (manual gradient computation)
- Gradient descent optimization
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# This ensures you get the same results every time you run the code
np.random.seed(42)

# ==============================================================================
# STEP 1: Generate Synthetic Data
# ==============================================================================
# We'll create a simple linear relationship: y = 2x + 1 (with some noise)
print("=" * 70)
print("STEP 1: Generating Synthetic Data")
print("=" * 70)

# Number of training examples
n_samples = 100

# Generate random x values between 0 and 10
X = np.random.rand(n_samples, 1) * 10

# Generate y values: y = 2x + 1 + noise
# The "true" relationship has weight=2 and bias=1
true_w = 2.0
true_b = 1.0
noise = np.random.randn(n_samples, 1) * 0.5  # Add some random noise
y = true_w * X + true_b + noise

print(f"Generated {n_samples} data points")
print(f"True weight: {true_w}, True bias: {true_b}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Sample data - X[0]: {X[0][0]:.2f}, y[0]: {y[0][0]:.2f}")

# ==============================================================================
# STEP 2: Initialize Model Parameters
# ==============================================================================
# In linear regression, our model is: y_pred = w * x + b
# We need to learn the optimal values for w (weight) and b (bias)
print("\n" + "=" * 70)
print("STEP 2: Initializing Model Parameters")
print("=" * 70)

# Initialize weight and bias randomly
# In practice, good initialization is crucial for training
w = np.random.randn(1, 1)  # Random small value for weight
b = np.zeros((1, 1))        # Start bias at zero

print(f"Initial weight: {w[0][0]:.4f}")
print(f"Initial bias: {b[0][0]:.4f}")

# ==============================================================================
# STEP 3: Define Forward Pass, Loss Function, and Gradients
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Defining Model Functions")
print("=" * 70)

def forward(X, w, b):
    """
    Forward pass: Compute predictions
    
    Formula: y_pred = w * X + b
    
    Args:
        X: Input data, shape (n_samples, 1)
        w: Weight parameter, shape (1, 1)
        b: Bias parameter, shape (1, 1)
    
    Returns:
        y_pred: Predictions, shape (n_samples, 1)
    """
    return X @ w + b  # Matrix multiplication (same as X * w for 1D)

def compute_loss(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE) loss
    
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    
    This measures how far our predictions are from the true values.
    Lower loss = better predictions.
    
    Args:
        y_true: True labels, shape (n_samples, 1)
        y_pred: Predicted values, shape (n_samples, 1)
    
    Returns:
        loss: Scalar value representing the average squared error
    """
    n = len(y_true)
    loss = (1 / n) * np.sum((y_true - y_pred) ** 2)
    return loss

def compute_gradients(X, y_true, y_pred):
    """
    Compute gradients using calculus (backpropagation)
    
    For MSE loss L = (1/n) * Σ(y_true - y_pred)²
    where y_pred = w * X + b
    
    The gradients are:
    - dL/dw = (2/n) * Σ X * (y_pred - y_true)
    - dL/db = (2/n) * Σ (y_pred - y_true)
    
    These tell us how to adjust w and b to reduce the loss.
    
    Args:
        X: Input data, shape (n_samples, 1)
        y_true: True labels, shape (n_samples, 1)
        y_pred: Predicted values, shape (n_samples, 1)
    
    Returns:
        dw: Gradient with respect to weight
        db: Gradient with respect to bias
    """
    n = len(y_true)
    
    # Gradient of loss with respect to weight
    # dL/dw = (2/n) * X^T * (y_pred - y_true)
    dw = (2 / n) * X.T @ (y_pred - y_true)
    
    # Gradient of loss with respect to bias
    # dL/db = (2/n) * sum(y_pred - y_true)
    db = (2 / n) * np.sum(y_pred - y_true)
    
    return dw, db

print("Forward pass, loss, and gradient functions defined")

# ==============================================================================
# STEP 4: Training Loop (Gradient Descent)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Training the Model")
print("=" * 70)

# Hyperparameters
learning_rate = 0.01  # How big of a step we take in gradient descent
n_epochs = 100        # Number of times we iterate over the entire dataset

# Store loss history for visualization
loss_history = []

print(f"Training for {n_epochs} epochs with learning rate {learning_rate}\n")

# Training loop
for epoch in range(n_epochs):
    # 1. Forward pass: Compute predictions
    y_pred = forward(X, w, b)
    
    # 2. Compute loss
    loss = compute_loss(y, y_pred)
    loss_history.append(loss)
    
    # 3. Compute gradients (backpropagation)
    dw, db = compute_gradients(X, y, y_pred)
    
    # 4. Update parameters using gradient descent
    # New parameter = Old parameter - learning_rate * gradient
    # We move in the opposite direction of the gradient to minimize loss
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss:.4f}, "
              f"w: {w[0][0]:.4f}, b: {b[0][0]:.4f}")

# ==============================================================================
# STEP 5: Evaluate Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Final Results")
print("=" * 70)

print(f"\nTrue values:    w = {true_w:.4f}, b = {true_b:.4f}")
print(f"Learned values: w = {w[0][0]:.4f}, b = {b[0][0]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")

# Calculate percentage error
w_error = abs(w[0][0] - true_w) / true_w * 100
b_error = abs(b[0][0] - true_b) / true_b * 100
print(f"Weight error: {w_error:.2f}%")
print(f"Bias error: {b_error:.2f}%")

# ==============================================================================
# STEP 6: Visualize Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Visualizing Results")
print("=" * 70)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Data and fitted line
ax1.scatter(X, y, alpha=0.5, label='Training data')
X_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = forward(X_line, w, b)
ax1.plot(X_line, y_line, 'r-', linewidth=2, label=f'Fitted line: y = {w[0][0]:.2f}x + {b[0][0]:.2f}')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression: Data and Fitted Line')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: Loss over epochs
ax2.plot(loss_history, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (MSE)')
ax2.set_title('Training Loss Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/01_results.png', dpi=100, bbox_inches='tight')
print("Visualization saved as '01_results.png'")
print("\nTraining completed successfully! ✓")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
# 1. Neural networks (even simple linear regression) work by:
#    - Making predictions (forward pass)
#    - Measuring how wrong they are (loss function)
#    - Computing how to improve (gradients via backpropagation)
#    - Updating parameters (gradient descent)
#
# 2. The learning rate controls how fast we learn:
#    - Too small: slow convergence
#    - Too large: might overshoot the optimal solution
#
# 3. More epochs generally mean better fit, but:
#    - Eventually, the improvement becomes negligible
#    - On real data, too many epochs can lead to overfitting
#
# NEXT STEPS:
# - Move on to Tutorial 02 to see how PyTorch simplifies this process!
# ==============================================================================
