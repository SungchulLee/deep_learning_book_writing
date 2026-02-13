"""
==============================================================================
05_linear_regression_nn_module.py
==============================================================================
DIFFICULTY: ⭐⭐ (Intermediate)

DESCRIPTION:
    Linear regression using PyTorch's nn.Module and nn.Linear.
    This is the "proper" PyTorch way to build models.

TOPICS COVERED:
    - nn.Module class for models
    - nn.Linear layer
    - Optimizers (torch.optim.SGD)
    - Cleaner, more scalable code

PREREQUISITES:
    - Tutorial 04 (Autograd)

LEARNING OBJECTIVES:
    - Create custom models with nn.Module
    - Use built-in layers (nn.Linear)
    - Use optimizers for parameter updates
    - Follow PyTorch best practices

TIME: ~20 minutes
==============================================================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH NN.MODULE")
print("=" * 70)

# ============================================================================
# PART 1: GENERATE DATA
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE DATA")
print("=" * 70)

torch.manual_seed(42)

TRUE_W = 2.5
TRUE_B = 3.0
n_samples = 100

X = torch.rand(n_samples, 1) * 20 - 10  # Shape: (100, 1)
noise = torch.randn(n_samples, 1) * 2
y = TRUE_W * X + TRUE_B + noise

print(f"Data shapes: X={X.shape}, y={y.shape}")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")

# ============================================================================
# PART 2: DEFINE MODEL USING NN.MODULE
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DEFINE MODEL CLASS")
print("=" * 70)

class LinearRegressionModel(nn.Module):
    """
    Linear Regression Model using nn.Module
    
    This is the standard way to define models in PyTorch.
    All models should inherit from nn.Module.
    """
    
    def __init__(self, input_dim=1, output_dim=1):
        """
        Initialize the model
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        # Always call parent constructor first
        super(LinearRegressionModel, self).__init__()
        
        # Define layers
        # nn.Linear(in_features, out_features)
        # This creates: y = X @ W.T + b
        # where W is shape (out_features, in_features)
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass: define how data flows through the model
        
        Args:
            x: Input tensor
        
        Returns:
            Output predictions
        """
        return self.linear(x)

# Create model instance
model = LinearRegressionModel(input_dim=1, output_dim=1)

print("Model created:")
print(model)
print(f"\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# ============================================================================
# PART 3: DEFINE LOSS AND OPTIMIZER
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DEFINE LOSS AND OPTIMIZER")
print("=" * 70)

# Loss function
criterion = nn.MSELoss()  # Mean Squared Error
print(f"Loss function: {criterion}")

# Optimizer - handles parameter updates automatically!
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f"Optimizer: {optimizer}")
print(f"Learning rate: {learning_rate}")

print("""
Key advantages of using an Optimizer:
1. Automatically updates all model parameters
2. No need for manual parameter updates
3. Easy to switch optimizers (SGD, Adam, RMSprop, etc.)
4. Handles gradient zeroing with optimizer.zero_grad()
""")

# ============================================================================
# PART 4: TRAINING LOOP - THE PYTORCH WAY
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING LOOP")
print("=" * 70)

n_epochs = 100
loss_history = []
w_history = []
b_history = []

print(f"Training for {n_epochs} epochs...")
print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12}")
print("-" * 50)

for epoch in range(n_epochs):
    # 1. Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # 2. Backward pass
    optimizer.zero_grad()  # Zero gradients (replaces w.grad.zero_())
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters (replaces manual update)
    
    # Track history
    loss_history.append(loss.item())
    w_history.append(model.linear.weight.item())
    b_history.append(model.linear.bias.item())
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss.item():<12.4f} "
              f"{model.linear.weight.item():<12.4f} "
              f"{model.linear.bias.item():<12.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

final_w = model.linear.weight.item()
final_b = model.linear.bias.item()

print(f"\nFinal Results:")
print(f"  Learned w: {final_w:.4f} (True: {TRUE_W}, Error: {abs(final_w-TRUE_W):.4f})")
print(f"  Learned b: {final_b:.4f} (True: {TRUE_B}, Error: {abs(final_b-TRUE_B):.4f})")
print(f"  Final loss: {loss_history[-1]:.6f}")

# ============================================================================
# PART 5: MODEL EVALUATION MODE
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: MODEL EVALUATION MODE")
print("=" * 70)

print("""
Models have two modes:
1. Training mode (model.train()): Default, enables dropout, batch norm, etc.
2. Evaluation mode (model.eval()): Disables dropout, batch norm, etc.

For linear regression, this doesn't matter, but it's good practice!
""")

# Set model to evaluation mode
model.eval()
print("Model set to evaluation mode")

# Make predictions (no gradient tracking needed)
with torch.no_grad():
    X_test = torch.tensor([[5.0], [-3.0], [0.0]])
    y_pred_test = model(X_test)
    
print(f"\nTest predictions:")
for i in range(len(X_test)):
    x_val = X_test[i].item()
    y_true_val = TRUE_W * x_val + TRUE_B
    y_pred_val = y_pred_test[i].item()
    print(f"  X={x_val:6.1f} -> Pred: {y_pred_val:7.2f}, True: {y_true_val:7.2f}")

# ============================================================================
# PART 6: SAVING AND LOADING MODELS
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: SAVING AND LOADING MODELS")
print("=" * 70)

# Save model
model_path = '/home/claude/pytorch_linear_regression_tutorial/linear_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Load model
new_model = LinearRegressionModel(input_dim=1, output_dim=1)
new_model.load_state_dict(torch.load(model_path))
new_model.eval()
print("Model loaded successfully")

# Verify loaded model
with torch.no_grad():
    y_pred_loaded = new_model(X_test)
    print("\nVerifying loaded model (should match above):")
    for i in range(len(X_test)):
        print(f"  X={X_test[i].item():6.1f} -> Pred: {y_pred_loaded[i].item():7.2f}")

# ============================================================================
# PART 7: VISUALIZE RESULTS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curve
axes[0, 0].plot(loss_history, linewidth=2, color='blue')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss (nn.Module)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Parameter evolution
axes[0, 1].plot(w_history, label='w', linewidth=2)
axes[0, 1].axhline(y=TRUE_W, color='r', linestyle='--', label=f'True w={TRUE_W}')
axes[0, 1].plot(b_history, label='b', linewidth=2)
axes[0, 1].axhline(y=TRUE_B, color='g', linestyle='--', label=f'True b={TRUE_B}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_title('Parameter Evolution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Final fit
model.eval()
with torch.no_grad():
    X_sorted, _ = torch.sort(X, dim=0)
    y_pred_sorted = model(X_sorted)

axes[1, 0].scatter(X.numpy(), y.numpy(), alpha=0.5, s=20)
axes[1, 0].plot(X_sorted.numpy(), y_pred_sorted.numpy(), 'r-', linewidth=2)
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Final Model Fit')
axes[1, 0].grid(True, alpha=0.3)

# Code comparison
comparison = """
TRAINING LOOP EVOLUTION:

Tutorial 02 (NumPy):
  - Manual gradient formulas
  - Manual parameter updates
  - ~40 lines of code

Tutorial 03 (PyTorch Manual):
  - Tensor operations
  - Manual gradients
  - Manual updates

Tutorial 04 (Autograd):
  - loss.backward()
  - Manual updates
  - grad.zero_()

Tutorial 05 (nn.Module):
  - model(X)
  - optimizer.zero_grad()
  - loss.backward()
  - optimizer.step()
  
Much cleaner and more maintainable!
"""
axes[1, 1].text(0.05, 0.95, comparison, transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/05_nn_module_results.png', dpi=100)
print("\nSaved visualization to: 05_nn_module_results.png")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Standard PyTorch Training Loop:

model = MyModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    # Forward
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Key Components:
1. nn.Module: Base class for all models
2. nn.Linear: Built-in linear layer
3. optimizer: Handles parameter updates
4. criterion: Loss function

Advantages:
✓ Clean, readable code
✓ Easy to extend to complex models
✓ Automatic parameter management
✓ Easy to save/load models
✓ GPU support (just add .to('cuda'))

Next: Tutorial 06 - Multiple input features!
""")
