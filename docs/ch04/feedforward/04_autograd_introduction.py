"""
==============================================================================
Tutorial 04: PyTorch Autograd - Automatic Differentiation
==============================================================================
DIFFICULTY: ⭐⭐ Intermediate

WHAT YOU'LL LEARN:
- PyTorch's automatic differentiation system
- How requires_grad works
- Using .backward() to compute gradients automatically
- The computational graph

PREREQUISITES:
- Tutorial 03 (understanding of backpropagation)
- Comfort with PyTorch tensors

KEY CONCEPTS:
- Autograd (automatic differentiation)
- Computational graph
- requires_grad parameter
- .backward() and .grad
- torch.no_grad() context
==============================================================================
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

print("=" * 70)
print("Welcome to PyTorch Autograd!")
print("=" * 70)
print("\nNo more manual gradient computation! 🎉")
print("PyTorch will do it all for us automatically.\n")

# ==============================================================================
# PART 1: Introduction to Autograd
# ==============================================================================
print("=" * 70)
print("PART 1: Understanding Autograd Basics")
print("=" * 70)

# Simple example: compute gradient of y = x^2 at x = 3
print("\nExample: Finding derivative of y = x² at x = 3")
print("-" * 50)

# Create a tensor with gradient tracking enabled
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x.item()}")
print(f"x.requires_grad = {x.requires_grad}")

# Compute y = x^2
y = x ** 2
print(f"\ny = x² = {y.item()}")
print(f"y.requires_grad = {y.requires_grad}")

# Compute gradient: dy/dx
y.backward()  # This computes the gradient!
print(f"\ndy/dx at x=3: {x.grad.item()}")
print(f"Mathematical answer: 2x = 2(3) = 6 ✓")

print("\n" + "-" * 50)
print("What just happened?")
print("-" * 50)
print("1. PyTorch built a computational graph: x -> (square) -> y")
print("2. .backward() traversed the graph backwards")
print("3. Gradients were computed automatically using chain rule")
print("4. The gradient was stored in x.grad")

# ==============================================================================
# PART 2: More Complex Example
# ==============================================================================
print("\n" + "=" * 70)
print("PART 2: Multi-Variable Gradients")
print("=" * 70)

# Create two variables
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Compute: z = 3a² + 2b
z = 3 * a**2 + 2 * b
print(f"z = 3a² + 2b where a=2, b=3")
print(f"z = {z.item()}")

# Compute gradients
z.backward()
print(f"\n∂z/∂a = {a.grad.item()} (mathematical answer: 6a = 12) ✓")
print(f"∂z/∂b = {b.grad.item()} (mathematical answer: 2) ✓")

# ==============================================================================
# IMPORTANT: Gradient Accumulation
# ==============================================================================
print("\n" + "=" * 70)
print("PART 3: Understanding Gradient Accumulation")
print("=" * 70)

print("\n⚠️  IMPORTANT: Gradients accumulate by default!")
print("-" * 50)

x = torch.tensor(2.0, requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(f"After first backward: x.grad = {x.grad.item()}")

# Second computation WITHOUT clearing gradients
y2 = x ** 3
y2.backward()
print(f"After second backward: x.grad = {x.grad.item()}")
print("^ Notice: gradients accumulated (4 + 12 = 16)")

print("\nTo prevent this, use: x.grad.zero_() or optimizer.zero_grad()")

# ==============================================================================
# STEP 1: Generate Data (Same as Tutorial 03)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Generating Data")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_samples = 200

def generate_circle_data(n_samples):
    angles = torch.rand(n_samples) * 2 * np.pi
    radii = torch.zeros(n_samples)
    radii[:n_samples//2] = torch.rand(n_samples//2) * 2
    radii[n_samples//2:] = torch.rand(n_samples//2) * 2 + 3
    
    X = torch.stack([radii * torch.cos(angles), 
                     radii * torch.sin(angles)], dim=1)
    y = torch.zeros(n_samples, 1)
    y[n_samples//2:] = 1
    X += torch.randn_like(X) * 0.3
    
    return X, y

X, y = generate_circle_data(n_samples)
X = X.to(device)
y = y.to(device)

print(f"Generated {n_samples} data points on {device}")

# ==============================================================================
# STEP 2: Initialize Parameters WITH requires_grad=True
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Initializing Parameters with Autograd")
print("=" * 70)

input_size = 2
hidden_size = 8
output_size = 1

# KEY DIFFERENCE: requires_grad=True enables gradient tracking!
w1 = torch.randn(input_size, hidden_size, device=device, requires_grad=True)
b1 = torch.zeros(1, hidden_size, device=device, requires_grad=True)
w2 = torch.randn(hidden_size, output_size, device=device, requires_grad=True)
b2 = torch.zeros(1, output_size, device=device, requires_grad=True)

print(f"w1.requires_grad = {w1.requires_grad} ✓")
print(f"b1.requires_grad = {b1.requires_grad} ✓")
print(f"w2.requires_grad = {w2.requires_grad} ✓")
print(f"b2.requires_grad = {b2.requires_grad} ✓")
print("\nPyTorch will now track all operations on these tensors!")

# ==============================================================================
# STEP 3: Define Forward Pass (Simplified!)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Forward Pass (Much Simpler Now)")
print("=" * 70)

def forward(X, w1, b1, w2, b2):
    """
    Forward pass - PyTorch tracks everything automatically!
    
    No need to manually cache intermediate values for backprop.
    PyTorch builds the computational graph automatically.
    
    Args:
        X: Input data
        w1, b1: First layer parameters
        w2, b2: Second layer parameters
    
    Returns:
        y_pred: Final predictions
    """
    # Layer 1
    z1 = X @ w1 + b1
    a1 = torch.relu(z1)  # Using PyTorch's built-in ReLU!
    
    # Layer 2
    z2 = a1 @ w2 + b2
    a2 = torch.sigmoid(z2)  # Using PyTorch's built-in Sigmoid!
    
    return a2

print("Forward pass defined - no manual caching needed!")
print("PyTorch automatically:")
print("  - Tracks all operations")
print("  - Builds computational graph")
print("  - Stores values needed for backprop")

# ==============================================================================
# STEP 4: Define Loss Function
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Loss Function")
print("=" * 70)

def compute_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss
    
    Note: We could use torch.nn.BCELoss() instead,
    but implementing it helps understanding.
    """
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -torch.mean(y_true * torch.log(y_pred) + 
                       (1 - y_true) * torch.log(1 - y_pred))
    return loss

print("Loss function defined (Binary Cross-Entropy)")

# ==============================================================================
# STEP 5: Training Loop with Autograd
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Training with Autograd")
print("=" * 70)

learning_rate = 0.1
n_epochs = 1000
loss_history = []

print(f"Training for {n_epochs} epochs with learning rate {learning_rate}")
print("\nCompare this to Tutorial 03:")
print("  Tutorial 03: Manual backpropagation (many lines of code)")
print("  Tutorial 04: loss.backward() (one line!) ✨\n")

for epoch in range(n_epochs):
    # ===========================================================
    # KEY STEP: Zero gradients from previous iteration
    # ===========================================================
    # Gradients accumulate by default, so we must zero them
    if w1.grad is not None:
        w1.grad.zero_()
        b1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()
    
    # ===========================================================
    # Forward pass - PyTorch tracks operations automatically
    # ===========================================================
    y_pred = forward(X, w1, b1, w2, b2)
    
    # Compute loss
    loss = compute_loss(y, y_pred)
    loss_history.append(loss.item())
    
    # ===========================================================
    # MAGIC HAPPENS HERE: Backward pass in ONE line!
    # ===========================================================
    loss.backward()  # Computes all gradients automatically!
    
    # Now all parameters have their gradients:
    # w1.grad, b1.grad, w2.grad, b2.grad are all populated
    
    # ===========================================================
    # Update parameters
    # ===========================================================
    # Use torch.no_grad() to prevent tracking these operations
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        predictions = (y_pred > 0.5).float()
        accuracy = (predictions == y).float().mean().item() * 100
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, "
              f"Accuracy: {accuracy:.2f}%")

# ==============================================================================
# STEP 6: Evaluate Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Final Evaluation")
print("=" * 70)

# Make predictions (no gradient tracking needed for inference)
with torch.no_grad():
    y_pred_final = forward(X, w1, b1, w2, b2)
    predictions = (y_pred_final > 0.5).float()
    final_accuracy = (predictions == y).float().mean().item() * 100

print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
print(f"Final Loss: {loss_history[-1]:.4f}")

# ==============================================================================
# STEP 7: Visualize Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Visualization")
print("=" * 70)

X_cpu = X.cpu().numpy()
y_cpu = y.cpu().numpy()
pred_cpu = predictions.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training data
ax1 = axes[0]
ax1.scatter(X_cpu[y_cpu.flatten() == 0, 0], X_cpu[y_cpu.flatten() == 0, 1],
           c='blue', label='Class 0', alpha=0.6, edgecolors='k')
ax1.scatter(X_cpu[y_cpu.flatten() == 1, 0], X_cpu[y_cpu.flatten() == 1, 1],
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
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss Over Time')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/04_results.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved as '04_results.png'")
print("Training completed successfully! ✓")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
# 1. PyTorch Autograd eliminates manual gradient computation:
#    - Set requires_grad=True for parameters you want to optimize
#    - PyTorch automatically tracks all operations
#    - .backward() computes all gradients via chain rule
#
# 2. Key autograd concepts:
#    - Computational graph: PyTorch builds this automatically
#    - .backward(): Computes gradients by traversing graph backwards
#    - .grad: Where PyTorch stores computed gradients
#    - .zero_(): Clear gradients (they accumulate otherwise!)
#
# 3. torch.no_grad() context:
#    - Use when you don't need gradients (e.g., parameter updates, inference)
#    - Saves memory and computation
#    - Critical for parameter updates to prevent tracking
#
# 4. Comparison with Tutorial 03:
#    Tutorial 03 (Manual):      ~30 lines of backprop code
#    Tutorial 04 (Autograd):    loss.backward()  (1 line!)
#
# 5. This is powerful but we're still manually:
#    - Managing parameters (w1, b1, w2, b2)
#    - Zeroing gradients
#    - Updating parameters
#    Next tutorial: Use nn.Module and optimizers!
#
# NEXT STEPS:
# - Tutorial 05: Use torch.nn.Module for cleaner code
# - Tutorial 06: Use optimizers instead of manual updates
# - Tutorial 07: Train on real datasets (MNIST)
# ==============================================================================
