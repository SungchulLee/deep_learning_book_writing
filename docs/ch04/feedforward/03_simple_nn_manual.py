"""
==============================================================================
Tutorial 03: Simple Neural Network with Manual Backpropagation
==============================================================================
DIFFICULTY: ⭐⭐ Beginner-Intermediate

WHAT YOU'LL LEARN:
- Building a multi-layer neural network from scratch
- Forward propagation through multiple layers
- Backpropagation with non-linear activations
- The role of activation functions (ReLU, Sigmoid)

PREREQUISITES:
- Tutorials 01 & 02 (linear regression basics)
- Understanding of matrix multiplication
- Basic calculus (chain rule)

KEY CONCEPTS:
- Multi-layer perceptron (MLP)
- Activation functions (ReLU)
- Backpropagation through layers
- Non-linear transformations
==============================================================================
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ==============================================================================
# STEP 1: Generate Non-Linear Data
# ==============================================================================
print("=" * 70)
print("STEP 1: Generating Non-Linear Data")
print("=" * 70)

# This time we need a non-linear dataset
# Linear models can't solve non-linear problems!
# Let's create a simple classification problem: XOR-like data

n_samples = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data that requires non-linear decision boundary
# We'll use a circular pattern
def generate_circle_data(n_samples):
    """
    Generate data in concentric circles
    Inner circle: class 0, Outer circle: class 1
    """
    # Generate random angles and radii
    angles = torch.rand(n_samples) * 2 * np.pi
    
    # Half samples in inner circle, half in outer circle
    radii = torch.zeros(n_samples)
    radii[:n_samples//2] = torch.rand(n_samples//2) * 2  # Inner: radius 0-2
    radii[n_samples//2:] = torch.rand(n_samples//2) * 2 + 3  # Outer: radius 3-5
    
    # Convert polar to Cartesian coordinates
    X = torch.stack([radii * torch.cos(angles), 
                     radii * torch.sin(angles)], dim=1)
    
    # Labels: 0 for inner circle, 1 for outer circle
    y = torch.zeros(n_samples, 1)
    y[n_samples//2:] = 1
    
    # Add some noise
    X += torch.randn_like(X) * 0.3
    
    return X, y

X, y = generate_circle_data(n_samples)
X = X.to(device)
y = y.to(device)

print(f"Generated {n_samples} data points")
print(f"X shape: {X.shape} (features: x-coordinate, y-coordinate)")
print(f"y shape: {y.shape} (labels: 0 or 1)")
print(f"Class 0 samples: {(y == 0).sum().item()}")
print(f"Class 1 samples: {(y == 1).sum().item()}")

# ==============================================================================
# STEP 2: Initialize Neural Network Parameters
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Initializing Neural Network Parameters")
print("=" * 70)

# Network architecture:
# Input layer: 2 features (x, y coordinates)
# Hidden layer: 8 neurons
# Output layer: 1 neuron (binary classification)

input_size = 2
hidden_size = 8
output_size = 1

# Initialize weights with small random values
# Good initialization is important for training!
# We use Xavier/Glorot initialization: scale by sqrt(n_inputs)
w1 = torch.randn(input_size, hidden_size, device=device) * np.sqrt(2.0 / input_size)
b1 = torch.zeros(1, hidden_size, device=device)

w2 = torch.randn(hidden_size, output_size, device=device) * np.sqrt(2.0 / hidden_size)
b2 = torch.zeros(1, output_size, device=device)

print(f"Network architecture:")
print(f"  Input layer:  {input_size} neurons")
print(f"  Hidden layer: {hidden_size} neurons")
print(f"  Output layer: {output_size} neuron")
print(f"\nParameter shapes:")
print(f"  w1: {w1.shape}, b1: {b1.shape}")
print(f"  w2: {w2.shape}, b2: {b2.shape}")
print(f"  Total parameters: {w1.numel() + b1.numel() + w2.numel() + b2.numel()}")

# ==============================================================================
# STEP 3: Define Activation Functions
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Understanding Activation Functions")
print("=" * 70)

def relu(x):
    """
    ReLU (Rectified Linear Unit) activation
    
    Formula: ReLU(x) = max(0, x)
    
    Why use ReLU?
    - Introduces non-linearity (allows network to learn complex patterns)
    - Computationally efficient
    - Helps with vanishing gradient problem
    - Most popular activation for hidden layers
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with ReLU applied element-wise
    """
    return torch.maximum(x, torch.tensor(0.0, device=x.device))

def relu_derivative(x):
    """
    Derivative of ReLU
    
    Formula: d(ReLU)/dx = 1 if x > 0, else 0
    
    This is needed for backpropagation!
    
    Args:
        x: Input tensor (the pre-activation values)
    
    Returns:
        Gradient tensor
    """
    return (x > 0).float()

def sigmoid(x):
    """
    Sigmoid activation
    
    Formula: σ(x) = 1 / (1 + e^(-x))
    
    Why use Sigmoid for output?
    - Squashes output to range (0, 1)
    - Interpretable as probability for binary classification
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor in range (0, 1)
    """
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of Sigmoid
    
    Formula: dσ/dx = σ(x) * (1 - σ(x))
    
    Args:
        x: Input tensor (sigmoid outputs)
    
    Returns:
        Gradient tensor
    """
    return x * (1 - x)

print("Activation functions defined:")
print("  - ReLU: For hidden layers (non-linearity)")
print("  - Sigmoid: For output layer (probability)")
print("\nWhy activation functions matter:")
print("  Without them, multiple layers collapse to a single linear transformation!")
print("  Activation functions allow the network to learn non-linear patterns.")

# ==============================================================================
# STEP 4: Forward Propagation
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Forward Propagation")
print("=" * 70)

def forward(X, w1, b1, w2, b2):
    """
    Forward pass through the network
    
    Flow:
    Input (2) -> Linear -> ReLU -> Hidden (8) -> Linear -> Sigmoid -> Output (1)
    
    Args:
        X: Input data, shape (n_samples, 2)
        w1, b1: First layer parameters
        w2, b2: Second layer parameters
    
    Returns:
        y_pred: Final predictions, shape (n_samples, 1)
        cache: Intermediate values needed for backprop
    """
    # Layer 1: Input -> Hidden
    z1 = X @ w1 + b1  # Linear transformation (n_samples, hidden_size)
    a1 = relu(z1)      # Activation (n_samples, hidden_size)
    
    # Layer 2: Hidden -> Output
    z2 = a1 @ w2 + b2  # Linear transformation (n_samples, output_size)
    a2 = sigmoid(z2)   # Activation (n_samples, output_size)
    
    # Store intermediate values for backpropagation
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    
    return a2, cache

print("Forward propagation defined")
print("Network flow: Input -> [Linear+ReLU] -> [Linear+Sigmoid] -> Output")

# ==============================================================================
# STEP 5: Define Loss Function
# ==============================================================================

def compute_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss
    
    Formula: L = -(y*log(y_pred) + (1-y)*log(1-y_pred))
    
    Better than MSE for classification!
    Measures the difference between predicted probabilities and true labels.
    
    Args:
        y_true: True labels (0 or 1), shape (n_samples, 1)
        y_pred: Predicted probabilities, shape (n_samples, 1)
    
    Returns:
        loss: Scalar tensor
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    loss = -torch.mean(y_true * torch.log(y_pred) + 
                       (1 - y_true) * torch.log(1 - y_pred))
    return loss

# ==============================================================================
# STEP 6: Backpropagation
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Backpropagation (Chain Rule in Action)")
print("=" * 70)

def backward(X, y_true, y_pred, cache, w2):
    """
    Compute gradients using backpropagation (chain rule)
    
    We work backwards from the output to the input, computing
    how each parameter contributed to the loss.
    
    Chain rule example for w2:
    dL/dw2 = dL/da2 * da2/dz2 * dz2/dw2
    
    Args:
        X: Input data
        y_true: True labels
        y_pred: Predictions
        cache: Intermediate values from forward pass
        w2: Second layer weights (needed for backprop to first layer)
    
    Returns:
        Gradients for all parameters
    """
    n = X.shape[0]
    
    # Retrieve cached values
    z1 = cache['z1']
    a1 = cache['a1']
    a2 = cache['a2']
    
    # ===== Backpropagate through output layer =====
    # Gradient of loss w.r.t. output (cross-entropy + sigmoid)
    # Convenient property: dL/dz2 = (a2 - y_true) for this combination
    dz2 = a2 - y_true  # Shape: (n_samples, 1)
    
    # Gradients for w2 and b2
    dw2 = (1 / n) * a1.T @ dz2  # Shape: (hidden_size, 1)
    db2 = (1 / n) * torch.sum(dz2, dim=0, keepdim=True)  # Shape: (1, 1)
    
    # ===== Backpropagate through hidden layer =====
    # Gradient flowing back to hidden layer
    da1 = dz2 @ w2.T  # Shape: (n_samples, hidden_size)
    
    # Apply ReLU derivative
    dz1 = da1 * relu_derivative(z1)  # Shape: (n_samples, hidden_size)
    
    # Gradients for w1 and b1
    dw1 = (1 / n) * X.T @ dz1  # Shape: (input_size, hidden_size)
    db1 = (1 / n) * torch.sum(dz1, dim=0, keepdim=True)  # Shape: (1, hidden_size)
    
    return dw1, db1, dw2, db2

print("Backpropagation defined")
print("Key insight: Chain rule allows us to compute gradients efficiently")
print("We propagate error backwards through the network, layer by layer")

# ==============================================================================
# STEP 7: Training Loop
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Training the Neural Network")
print("=" * 70)

learning_rate = 0.1
n_epochs = 1000
loss_history = []

print(f"Training for {n_epochs} epochs with learning rate {learning_rate}\n")

for epoch in range(n_epochs):
    # Forward pass
    y_pred, cache = forward(X, w1, b1, w2, b2)
    
    # Compute loss
    loss = compute_loss(y, y_pred)
    loss_history.append(loss.item())
    
    # Backward pass
    dw1, db1, dw2, db2 = backward(X, y, y_pred, cache, w2)
    
    # Update parameters
    with torch.no_grad():
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        # Calculate accuracy
        predictions = (y_pred > 0.5).float()
        accuracy = (predictions == y).float().mean().item() * 100
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, "
              f"Accuracy: {accuracy:.2f}%")

# ==============================================================================
# STEP 8: Evaluate and Visualize
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Final Evaluation and Visualization")
print("=" * 70)

# Final predictions
y_pred_final, _ = forward(X, w1, b1, w2, b2)
predictions = (y_pred_final > 0.5).float()
final_accuracy = (predictions == y).float().mean().item() * 100

print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
print(f"Final Loss: {loss_history[-1]:.4f}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Convert to numpy for plotting
X_cpu = X.cpu().numpy()
y_cpu = y.cpu().numpy()
pred_cpu = predictions.cpu().numpy()

# Plot 1: Training data
ax1 = axes[0]
scatter1 = ax1.scatter(X_cpu[y_cpu.flatten() == 0, 0], X_cpu[y_cpu.flatten() == 0, 1],
                       c='blue', label='Class 0', alpha=0.6, edgecolors='k')
scatter2 = ax1.scatter(X_cpu[y_cpu.flatten() == 1, 0], X_cpu[y_cpu.flatten() == 1, 1],
                       c='red', label='Class 1', alpha=0.6, edgecolors='k')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Training Data (True Labels)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Predictions
ax2 = axes[1]
ax2.scatter(X_cpu[pred_cpu.flatten() == 0, 0], X_cpu[pred_cpu.flatten() == 0, 1],
            c='blue', label='Predicted Class 0', alpha=0.6, edgecolors='k')
ax2.scatter(X_cpu[pred_cpu.flatten() == 1, 0], X_cpu[pred_cpu.flatten() == 1, 1],
            c='red', label='Predicted Class 1', alpha=0.6, edgecolors='k')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title(f'Predictions (Accuracy: {final_accuracy:.1f}%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Loss curve
ax3 = axes[2]
ax3.plot(loss_history, linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (Binary Cross-Entropy)')
ax3.set_title('Training Loss Over Time')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/03_results.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved as '03_results.png'")
print("Training completed successfully! ✓")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
# 1. Neural networks with multiple layers can solve non-linear problems
#    that linear models cannot (like the circular pattern we used).
#
# 2. Activation functions are CRITICAL:
#    - Without them, multiple layers = one linear layer
#    - ReLU is popular for hidden layers (simple, effective)
#    - Sigmoid/Softmax for output layers in classification
#
# 3. Backpropagation uses the chain rule:
#    - Compute gradients layer by layer, going backwards
#    - Each layer's gradient depends on the next layer's gradient
#    - This is why we "cache" intermediate values during forward pass
#
# 4. Binary Cross-Entropy is better than MSE for classification:
#    - Provides better gradients for probability predictions
#    - More stable training
#
# 5. This manual backpropagation is educational but tedious!
#    Next tutorial: Let PyTorch compute gradients automatically!
#
# NEXT STEPS:
# - Tutorial 04: Use PyTorch's autograd (automatic differentiation)
# ==============================================================================
