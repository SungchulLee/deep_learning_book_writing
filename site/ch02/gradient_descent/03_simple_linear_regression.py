"""
================================================================================
Level 1 - Example 3: Simple Linear Regression with PyTorch
================================================================================

LEARNING OBJECTIVES:
- Apply gradient descent to a real regression problem
- Use PyTorch's nn.Module for better code organization
- Implement training and evaluation loops
- Visualize model predictions

DIFFICULTY: ⭐ Beginner

TIME: 30-40 minutes

PREREQUISITES:
- Completed Examples 01 and 02
- Basic understanding of linear regression

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("LINEAR REGRESSION WITH PYTORCH")
print("="*80)

# ============================================================================
# PART 1: Create a Realistic Dataset
# ============================================================================
print("\n" + "="*80)
print("PART 1: DATASET CREATION")
print("="*80)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data: y = 3x + 2 + noise
n_samples = 100
X_numpy = np.random.rand(n_samples, 1) * 10  # Random x values from 0 to 10
y_numpy = 3 * X_numpy + 2 + np.random.randn(n_samples, 1) * 2  # Add noise

# Convert to PyTorch tensors
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"Dataset size: {n_samples} samples")
print(f"X shape: {X.shape}  (n_samples, n_features)")
print(f"y shape: {y.shape}  (n_samples, 1)")
print(f"\nTrue relationship: y = 3*x + 2")
print("Goal: Learn this relationship from noisy data")

# ============================================================================
# PART 2: Define Model Using nn.Module (Professional Way)
# ============================================================================
print("\n" + "="*80)
print("PART 2: MODEL DEFINITION")
print("="*80)

class LinearRegressionModel(nn.Module):
    """
    Linear Regression Model: y = wx + b
    
    This is the PyTorch way of defining models.
    Benefits:
    - Automatic parameter management
    - Easy to extend
    - Works with PyTorch optimizers
    - Clean and organized code
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize model parameters
        
        Args:
            input_dim: number of input features
            output_dim: number of output values
        """
        super(LinearRegressionModel, self).__init__()
        
        # Define a linear layer: y = wx + b
        # This creates weight (w) and bias (b) with requires_grad=True
        self.linear = nn.Linear(input_dim, output_dim)
        
        print(f"Created linear layer: {input_dim} inputs → {output_dim} outputs")
    
    def forward(self, x):
        """
        Forward pass: compute predictions
        
        Args:
            x: input tensor of shape (batch_size, input_dim)
        
        Returns:
            predictions: output tensor of shape (batch_size, output_dim)
        """
        return self.linear(x)


# Create model instance
input_dim = 1   # Single feature (x)
output_dim = 1  # Single output (y)
model = LinearRegressionModel(input_dim, output_dim)

print(f"\nModel architecture:")
print(model)

# Print initial parameters
print(f"\nInitial parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data.numpy().flatten()}")

# ============================================================================
# PART 3: Define Loss Function and Optimizer
# ============================================================================
print("\n" + "="*80)
print("PART 3: LOSS FUNCTION & OPTIMIZER")
print("="*80)

# Loss function: Mean Squared Error
criterion = nn.MSELoss()
print("Loss function: MSE (Mean Squared Error)")
print("  L = (1/N) * Σ(y_pred - y_true)²")

# Optimizer: Stochastic Gradient Descent
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f"\nOptimizer: SGD (Stochastic Gradient Descent)")
print(f"  Learning rate: {learning_rate}")
print(f"  Optimizing: {sum(p.numel() for p in model.parameters())} parameters")

# ============================================================================
# PART 4: Training Loop
# ============================================================================
print("\n" + "="*80)
print("PART 4: TRAINING")
print("="*80)

n_epochs = 100
loss_history = []

print("Epoch |   Loss    | Weight  | Bias")
print("-" * 45)

for epoch in range(n_epochs):
    # -------------------------
    # Standard Training Steps:
    # -------------------------
    
    # 1. Forward pass
    y_pred = model(X)
    
    # 2. Compute loss
    loss = criterion(y_pred, y)
    
    # 3. Zero gradients (IMPORTANT!)
    optimizer.zero_grad()
    
    # 4. Backward pass (compute gradients)
    loss.backward()
    
    # 5. Update parameters
    optimizer.step()
    
    # Store loss for plotting
    loss_history.append(loss.item())
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        print(f"{epoch+1:4d}  | {loss.item():9.4f} | {weight:7.4f} | {bias:6.4f}")

print("-" * 45)

# ============================================================================
# PART 5: Evaluate Trained Model
# ============================================================================
print("\n" + "="*80)
print("PART 5: EVALUATION")
print("="*80)

# Get final parameters
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()

print(f"Learned equation: y = {final_weight:.4f}*x + {final_bias:.4f}")
print(f"True equation:    y = 3.0000*x + 2.0000")
print(f"\nParameter errors:")
print(f"  Weight error: {abs(final_weight - 3.0):.4f}")
print(f"  Bias error:   {abs(final_bias - 2.0):.4f}")

# Make predictions on test data
model.eval()  # Set model to evaluation mode (important for some layers)
with torch.no_grad():  # Don't track gradients during evaluation
    # Test on a few points
    x_test = torch.tensor([[2.0], [5.0], [8.0]])
    y_test_pred = model(x_test)
    
    print("\nTest predictions:")
    for i, x_val in enumerate(x_test):
        pred = y_test_pred[i].item()
        true = 3 * x_val.item() + 2
        print(f"  x = {x_val.item():.1f} → predicted: {pred:.2f}, true: {true:.2f}")

# ============================================================================
# PART 6: Visualization
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Training loss
axes[0].plot(loss_history, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Data and fitted line
axes[1].scatter(X.numpy(), y.numpy(), alpha=0.5, label='Training data')
x_line = torch.linspace(0, 10, 100).reshape(-1, 1)
with torch.no_grad():
    y_line = model(x_line)
axes[1].plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, 
             label=f'Fitted line (y={final_weight:.2f}x+{final_bias:.2f})')
axes[1].plot(x_line.numpy(), 3*x_line.numpy()+2, 'g--', linewidth=2, 
             label='True line (y=3x+2)', alpha=0.7)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('Linear Regression Fit', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Residuals (errors)
with torch.no_grad():
    y_pred_all = model(X)
    residuals = (y - y_pred_all).numpy()
axes[2].scatter(X.numpy(), residuals, alpha=0.5)
axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('Residual (y_true - y_pred)', fontsize=12)
axes[2].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/linear_regression.png', dpi=150)
print("\n✓ Plot saved as 'linear_regression.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# PART 7: Computing Metrics
# ============================================================================
print("\n" + "="*80)
print("PART 7: PERFORMANCE METRICS")
print("="*80)

with torch.no_grad():
    y_pred_all = model(X)
    
    # Mean Squared Error (MSE)
    mse = criterion(y_pred_all, y).item()
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(y_pred_all - y)).item()
    
    # R² Score (coefficient of determination)
    ss_res = torch.sum((y - y_pred_all) ** 2).item()
    ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()
    r2_score = 1 - (ss_res / ss_tot)

print(f"MSE (Mean Squared Error):      {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error):     {mae:.4f}")
print(f"R² Score:                      {r2_score:.4f}")
print("\nInterpretation:")
print(f"  - RMSE of {rmse:.2f} means predictions are off by ±{rmse:.2f} on average")
print(f"  - R² of {r2_score:.4f} means model explains {r2_score*100:.2f}% of variance")

# ============================================================================
# PART 8: Key Takeaways
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. nn.Module is the standard way to define models
   - Inherit from nn.Module
   - Define layers in __init__()
   - Implement forward() method

2. Training loop has 5 standard steps:
   1. Forward pass:      y_pred = model(X)
   2. Compute loss:      loss = criterion(y_pred, y)
   3. Zero gradients:    optimizer.zero_grad()
   4. Backward pass:     loss.backward()
   5. Update parameters: optimizer.step()

3. Use model.eval() and torch.no_grad() for evaluation
   - Disables certain layers (dropout, batch norm)
   - Saves memory by not tracking gradients

4. Multiple metrics provide better evaluation
   - MSE: penalizes large errors more
   - MAE: treats all errors equally
   - R²: measures fraction of variance explained

5. Residual plots help diagnose issues
   - Random scatter: good fit
   - Patterns: might need non-linear model
""")

# ============================================================================
# PART 9: Experiments to Try
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENTS TO TRY")
print("="*80)
print("""
1. Different learning rates:
   - Try lr = 0.001, 0.1, 0.5
   - Observe convergence speed and stability

2. More training data:
   - Increase n_samples to 1000
   - Does the model fit better?

3. More noise:
   - Increase noise level in data generation
   - How does it affect R² score?

4. Different optimizers:
   - Replace SGD with Adam: torch.optim.Adam(...)
   - Compare convergence

5. Save and load model:
   - torch.save(model.state_dict(), 'model.pth')
   - model.load_state_dict(torch.load('model.pth'))
""")
print("="*80)
