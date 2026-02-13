"""
================================================================================
BEGINNER 03: Introduction to Optimizers in PyTorch
================================================================================

WHAT YOU'LL LEARN:
- What is an optimizer and why we need it
- How optimizers use loss to update model parameters
- Understanding learning rate
- Basic SGD optimizer
- Your first complete training loop

PREREQUISITES:
- Complete 01_intro_to_loss_functions.py
- Understand basic neural networks

TIME TO COMPLETE: ~20 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 80)
print("INTRODUCTION TO OPTIMIZERS")
print("=" * 80)

# ============================================================================
# SECTION 1: What is an Optimizer?
# ============================================================================
print("\n" + "-" * 80)
print("WHAT IS AN OPTIMIZER?")
print("-" * 80)

print("""
An optimizer is the ALGORITHM that updates your model's parameters
to minimize the loss function.

Think of it like this:
1. Your model makes predictions (forward pass)
2. Loss function says "you're this wrong"
3. PyTorch computes gradients (backward pass)
4. Optimizer uses gradients to UPDATE the parameters
5. Repeat until loss is low enough

Without an optimizer, your model can't learn!
""")

# ============================================================================
# SECTION 2: Simple Example - Learning a Straight Line
# ============================================================================
print("\n" + "-" * 80)
print("EXAMPLE: Learning to Fit a Line")
print("-" * 80)

# Generate some data points on a line: y = 2x + 1
torch.manual_seed(42)  # For reproducibility
X = torch.linspace(0, 10, 50).reshape(-1, 1)  # 50 points from 0 to 10
y_true = 2 * X + 1 + torch.randn(50, 1) * 0.5  # y = 2x + 1 + noise

print(f"Data shape: X={X.shape}, y={y_true.shape}")
print(f"First 5 points:")
for i in range(5):
    print(f"  X={X[i].item():.2f}, y={y_true[i].item():.2f}")

# ============================================================================
# SECTION 3: Create a Simple Model
# ============================================================================
print("\n" + "-" * 80)
print("CREATE A SIMPLE MODEL")
print("-" * 80)

# We want to learn: y = w*x + b
# This is called Linear Regression
model = nn.Linear(1, 1)  # 1 input feature, 1 output

print("Initial model parameters:")
print(f"  Weight (w): {model.weight.item():.4f}")
print(f"  Bias (b): {model.bias.item():.4f}")

# Make initial predictions (will be bad!)
with torch.no_grad():  # Don't track gradients for this test
    y_pred_initial = model(X)

print(f"\nInitial prediction for X=5: {model(torch.tensor([[5.0]])).item():.2f}")
print(f"True value should be around: {2 * 5 + 1:.2f} = 11.0")

# ============================================================================
# SECTION 4: Setup Loss and Optimizer
# ============================================================================
print("\n" + "-" * 80)
print("SETUP LOSS FUNCTION AND OPTIMIZER")
print("-" * 80)

# Loss function - measures how wrong we are
criterion = nn.MSELoss()

# Optimizer - updates parameters to reduce loss
# Learning rate (lr) controls how big each update step is
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: {learning_rate}")

print("\nWHAT IS LEARNING RATE?")
print("- Too small: Training is very slow")
print("- Too large: Training is unstable, might not converge")
print("- Just right: Steady improvement")
print(f"- We're using: {learning_rate} (a common starting point)")

# ============================================================================
# SECTION 5: Training Loop - Step by Step
# ============================================================================
print("\n" + "-" * 80)
print("TRAINING LOOP - DETAILED WALKTHROUGH")
print("-" * 80)

print("\nLet's train for just ONE epoch (iteration) first:\n")

# Save initial parameters
initial_weight = model.weight.item()
initial_bias = model.bias.item()

# Step 1: Forward pass
y_pred = model(X)
print("Step 1: Forward pass (make predictions)")
print(f"  Predictions shape: {y_pred.shape}")

# Step 2: Compute loss
loss = criterion(y_pred, y_true)
print(f"\nStep 2: Compute loss")
print(f"  Loss: {loss.item():.4f}")

# Step 3: Zero gradients (important!)
optimizer.zero_grad()
print(f"\nStep 3: Zero gradients")
print(f"  Why? Gradients accumulate by default in PyTorch")
print(f"  Weight gradient before zero_grad: {model.weight.grad}")

# Step 4: Backward pass (compute gradients)
loss.backward()
print(f"\nStep 4: Backward pass (compute gradients)")
print(f"  Weight gradient: {model.weight.grad.item():.4f}")
print(f"  Bias gradient: {model.bias.grad.item():.4f}")
print(f"  Negative gradient = increase parameter, Positive = decrease")

# Step 5: Update parameters
optimizer.step()
print(f"\nStep 5: Update parameters using optimizer")
print(f"  Old weight: {initial_weight:.4f}")
print(f"  New weight: {model.weight.item():.4f}")
print(f"  Change: {model.weight.item() - initial_weight:.4f}")
print(f"\n  Old bias: {initial_bias:.4f}")
print(f"  New bias: {model.bias.item():.4f}")
print(f"  Change: {model.bias.item() - initial_bias:.4f}")

# ============================================================================
# SECTION 6: Complete Training Loop
# ============================================================================
print("\n" + "-" * 80)
print("COMPLETE TRAINING - 1000 EPOCHS")
print("-" * 80)

# Reset the model
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
print_every = 100  # Print progress every 100 epochs

print(f"Training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = criterion(y_pred, y_true)
    
    # Backward pass
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()         # Compute new gradients
    optimizer.step()        # Update parameters
    
    # Print progress
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ============================================================================
# SECTION 7: Results After Training
# ============================================================================
print("\n" + "-" * 80)
print("RESULTS AFTER TRAINING")
print("-" * 80)

print(f"Final loss: {loss.item():.4f}")
print(f"\nLearned parameters:")
print(f"  Weight (w): {model.weight.item():.4f} (target was ~2.0)")
print(f"  Bias (b): {model.bias.item():.4f} (target was ~1.0)")

# Test predictions
test_inputs = torch.tensor([[0.0], [5.0], [10.0]])
with torch.no_grad():
    predictions = model(test_inputs)

print(f"\nTest predictions:")
for i, (x, pred) in enumerate(zip(test_inputs, predictions)):
    expected = 2 * x.item() + 1
    print(f"  X={x.item():.1f}: Predicted={pred.item():.2f}, Expected≈{expected:.2f}")

# ============================================================================
# SECTION 8: Understanding the Training Process
# ============================================================================
print("\n" + "-" * 80)
print("UNDERSTANDING WHAT HAPPENED")
print("-" * 80)

print("""
1. INITIALIZATION:
   Model started with random weights and bias
   Initial predictions were far from correct

2. ITERATION (repeated 1000 times):
   a) Forward: Model makes predictions
   b) Loss: Measure how wrong predictions are
   c) Backward: Compute gradients (direction to improve)
   d) Step: Update parameters in the right direction

3. RESULT:
   Loss decreased from high to low
   Parameters converged to the true values (w≈2, b≈1)
   Model now makes accurate predictions!

4. KEY COMPONENTS:
   • Loss function: Tells us how wrong we are
   • Optimizer: Updates parameters to reduce loss
   • Learning rate: Controls update step size
   • Epochs: Number of times we go through the process
""")

# ============================================================================
# SECTION 9: The Optimizer Step Formula
# ============================================================================
print("\n" + "-" * 80)
print("HOW THE OPTIMIZER UPDATES PARAMETERS")
print("-" * 80)

print("""
For Stochastic Gradient Descent (SGD):

    new_parameter = old_parameter - learning_rate × gradient

Example from our training:
    If weight gradient = 2.5 and learning_rate = 0.01
    new_weight = old_weight - 0.01 × 2.5
    new_weight = old_weight - 0.025

The optimizer does this for EVERY parameter in the model!
""")

# ============================================================================
# SECTION 10: Common Pitfalls
# ============================================================================
print("\n" + "-" * 80)
print("COMMON PITFALLS (And How to Avoid Them)")
print("-" * 80)

print("""
❌ MISTAKE 1: Forgetting optimizer.zero_grad()
   → Gradients accumulate, causing incorrect updates
   ✓ Always call optimizer.zero_grad() before loss.backward()

❌ MISTAKE 2: Wrong order of operations
   → Must be: forward → loss → zero_grad → backward → step
   ✓ Follow this exact order in your training loop

❌ MISTAKE 3: Using torch.no_grad() during training
   → Prevents gradient computation, model can't learn
   ✓ Only use no_grad() for evaluation/testing

❌ MISTAKE 4: Not tracking loss
   → Can't tell if model is learning
   ✓ Print/log loss regularly to monitor training

❌ MISTAKE 5: Learning rate too high or too low
   → Model diverges or learns too slowly
   ✓ Start with 0.01 or 0.001, adjust based on loss curve
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Optimizers update model parameters to minimize loss
   
2. Basic training loop structure:
   for epoch in range(num_epochs):
       predictions = model(inputs)       # Forward
       loss = criterion(predictions, targets)  # Loss
       optimizer.zero_grad()             # Reset gradients
       loss.backward()                   # Compute gradients
       optimizer.step()                  # Update parameters

3. Learning rate is crucial:
   • Controls how much parameters change each step
   • Too high = unstable, too low = slow learning
   • Typically start with 0.01 or 0.001

4. SGD (Stochastic Gradient Descent) is the simplest optimizer
   • Updates: parameter -= learning_rate × gradient
   • Works well for many problems
   • Other optimizers (Adam, RMSprop) add improvements

NEXT STEPS:
→ Try different learning rates and see what happens
→ Experiment with more complex models
→ Learn about advanced optimizers (Adam, AdamW)
""")
print("=" * 80)
