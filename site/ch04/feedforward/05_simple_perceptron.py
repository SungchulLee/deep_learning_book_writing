"""
================================================================================
01_simple_perceptron.py - Understanding a Single Neuron
================================================================================

This example demonstrates the most basic unit of a neural network: a perceptron.
A perceptron is just a single neuron that performs a linear transformation
followed by an activation function.

MATHEMATICAL CONCEPT:
    output = activation(weight * input + bias)
    
In this example, we'll train a perceptron to learn a simple linear relationship.

LEARNING OBJECTIVES:
    1. Understand forward pass computation
    2. See how gradients are computed automatically (autograd)
    3. Learn basic tensor operations
    4. Understand the training loop structure

DIFFICULTY: ⭐☆☆☆☆ (Easiest)
TIME: 10-15 minutes
================================================================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ================================================================================
# PART 1: GENERATE SYNTHETIC DATA
# ================================================================================
# We'll create a simple dataset where y = 2x + 3 (linear relationship)
# Our goal: train a perceptron to learn these weights (2 and 3)

print("=" * 80)
print("STEP 1: Creating Synthetic Data")
print("=" * 80)

# Set random seed for reproducibility
# This ensures you get the same results every time you run the code
torch.manual_seed(42)

# Generate 100 random input values between 0 and 10
X = torch.randn(100, 1) * 10  # Shape: (100, 1) - 100 samples, 1 feature

# True relationship: y = 2x + 3 (what we want our model to learn)
# We add small noise to make it realistic
y = 2 * X + 3 + torch.randn(100, 1) * 0.5  # Small Gaussian noise added

print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")
print(f"First 5 samples:")
print(f"X: {X[:5].squeeze()}")
print(f"y: {y[:5].squeeze()}")

# ================================================================================
# PART 2: DEFINE THE PERCEPTRON (SINGLE NEURON)
# ================================================================================
print("\n" + "=" * 80)
print("STEP 2: Defining the Perceptron")
print("=" * 80)

# A perceptron is simply a linear layer: y = wx + b
# nn.Linear(input_features, output_features) creates this transformation
# input_features=1: we have 1 input dimension (x)
# output_features=1: we want 1 output dimension (y)
model = nn.Linear(1, 1)

# Let's examine the initial random weights
# The model has two parameters: weight (w) and bias (b)
print(f"Initial weight: {model.weight.item():.4f}")  # Random initialization
print(f"Initial bias: {model.bias.item():.4f}")      # Random initialization
print(f"Goal: Learn weight ≈ 2.0 and bias ≈ 3.0")

# ================================================================================
# PART 3: DEFINE LOSS FUNCTION AND OPTIMIZER
# ================================================================================
print("\n" + "=" * 80)
print("STEP 3: Setting Up Training Components")
print("=" * 80)

# Loss function: Measures how wrong our predictions are
# MSE (Mean Squared Error) = average of (predicted - actual)^2
# Good for regression problems like ours
loss_fn = nn.MSELoss()

# Optimizer: Updates weights to minimize loss
# SGD (Stochastic Gradient Descent) is the simplest optimizer
# lr (learning rate): Controls how big our update steps are
#   - Too large: Model might overshoot and diverge
#   - Too small: Training takes forever
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f"Loss function: Mean Squared Error")
print(f"Optimizer: Stochastic Gradient Descent (SGD)")
print(f"Learning rate: {learning_rate}")

# ================================================================================
# PART 4: TRAINING LOOP
# ================================================================================
print("\n" + "=" * 80)
print("STEP 4: Training the Perceptron")
print("=" * 80)

# Track loss over time for visualization
loss_history = []

# Number of times we'll iterate over the entire dataset
num_epochs = 100

for epoch in range(num_epochs):
    # ----------------------------------------
    # 4.1 FORWARD PASS
    # ----------------------------------------
    # Compute predictions using current weights
    # model(X) is equivalent to: weight * X + bias
    predictions = model(X)
    
    # ----------------------------------------
    # 4.2 COMPUTE LOSS
    # ----------------------------------------
    # How far off are our predictions from actual values?
    loss = loss_fn(predictions, y)
    
    # ----------------------------------------
    # 4.3 BACKWARD PASS (BACKPROPAGATION)
    # ----------------------------------------
    # First, clear any gradients from previous iteration
    # Without this, gradients would accumulate (add up) over iterations
    optimizer.zero_grad()
    
    # Compute gradients of loss with respect to all model parameters
    # This is the "magic" of autograd - PyTorch computes derivatives automatically!
    loss.backward()
    
    # ----------------------------------------
    # 4.4 UPDATE WEIGHTS
    # ----------------------------------------
    # Adjust weights in the direction that reduces loss
    # New weight = Old weight - learning_rate * gradient
    optimizer.step()
    
    # Save loss for plotting
    loss_history.append(loss.item())
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Loss: {loss.item():.4f} | "
              f"Weight: {model.weight.item():.4f} | "
              f"Bias: {model.bias.item():.4f}")

# ================================================================================
# PART 5: EVALUATE THE RESULTS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 5: Final Results")
print("=" * 80)

print(f"Learned weight: {model.weight.item():.4f} (target: 2.0)")
print(f"Learned bias: {model.bias.item():.4f} (target: 3.0)")
print(f"Final loss: {loss.item():.4f}")

# How close are we to the true parameters?
weight_error = abs(model.weight.item() - 2.0)
bias_error = abs(model.bias.item() - 3.0)
print(f"\nWeight error: {weight_error:.4f}")
print(f"Bias error: {bias_error:.4f}")

# ================================================================================
# PART 6: VISUALIZATION
# ================================================================================
print("\n" + "=" * 80)
print("STEP 6: Visualizing Results")
print("=" * 80)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ----------------------------------------
# Left plot: Data and learned function
# ----------------------------------------
# Convert tensors to numpy for plotting
X_np = X.numpy()
y_np = y.numpy()

# Generate predictions on a smooth range for a nice line
X_test = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
with torch.no_grad():  # Don't compute gradients during inference
    y_pred = model(X_test).numpy()

ax1.scatter(X_np, y_np, alpha=0.5, label='Training data', s=20)
ax1.plot(X_test.numpy(), y_pred, 'r-', linewidth=2, 
         label=f'Learned: y = {model.weight.item():.2f}x + {model.bias.item():.2f}')
ax1.plot(X_test.numpy(), 2*X_test.numpy() + 3, 'g--', linewidth=2, 
         label='True: y = 2.0x + 3.0', alpha=0.7)
ax1.set_xlabel('Input (x)', fontsize=12)
ax1.set_ylabel('Output (y)', fontsize=12)
ax1.set_title('Perceptron: Data and Learned Function', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ----------------------------------------
# Right plot: Training loss over time
# ----------------------------------------
ax2.plot(loss_history, linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (MSE)', fontsize=12)
ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # Log scale helps visualize convergence

plt.tight_layout()
plt.savefig('01_perceptron_results.png', dpi=150, bbox_inches='tight')
print("Plot saved as '01_perceptron_results.png'")
plt.show()

# ================================================================================
# KEY TAKEAWAYS
# ================================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. A perceptron performs linear transformation: y = wx + b

2. Training has 4 steps (repeated many times):
   - Forward pass: compute predictions
   - Compute loss: measure error
   - Backward pass: compute gradients
   - Update weights: improve model

3. Autograd (automatic differentiation) computes gradients for us
   - loss.backward() does all the calculus!
   - No need to manually compute derivatives

4. Learning rate controls step size
   - Too large: unstable, might diverge
   - Too small: slow convergence
   
5. Even a single neuron can learn linear relationships perfectly!

NEXT: Try changing the learning rate and see how it affects convergence speed.
""")

# ================================================================================
# EXERCISES FOR STUDENTS
# ================================================================================
print("=" * 80)
print("EXERCISES TO TRY")
print("=" * 80)
print("""
1. Change learning rate to 0.001 and 0.1 - what happens?
2. Increase noise in data generation - how does it affect learning?
3. Try 500 epochs instead of 100 - does the model improve?
4. Change the true relationship to y = 5x - 2 and retrain
5. What happens if you don't call optimizer.zero_grad()?
""")
