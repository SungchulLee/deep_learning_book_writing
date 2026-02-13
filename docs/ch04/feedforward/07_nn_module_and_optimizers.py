"""
==============================================================================
Tutorial 05: PyTorch nn.Module and Optimizers
==============================================================================
DIFFICULTY: ⭐⭐ Intermediate

WHAT YOU'LL LEARN:
- Building models with nn.Module
- Using nn.Sequential for simple architectures
- Built-in loss functions and optimizers
- Professional PyTorch code structure

PREREQUISITES:
- Tutorial 04 (autograd understanding)

KEY CONCEPTS:
- nn.Module (base class for neural networks)
- nn.Linear (fully connected layers)
- nn.Sequential (container for layers)
- torch.optim (optimization algorithms)
- Model training best practices
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ==============================================================================
# INTRODUCTION: The PyTorch Way
# ==============================================================================
print("=" * 70)
print("Welcome to Professional PyTorch!")
print("=" * 70)
print("\nWhat we've learned so far:")
print("  Tutorial 01-02: Basic tensor operations")
print("  Tutorial 03:    Manual backpropagation")
print("  Tutorial 04:    Autograd (automatic gradients)")
print("\nWhat we'll learn today:")
print("  ✓ nn.Module: Build models properly")
print("  ✓ nn.Sequential: Quick model building")
print("  ✓ Optimizers: No more manual parameter updates")
print("  ✓ Professional code structure")

# ==============================================================================
# STEP 1: Understanding nn.Module
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Understanding nn.Module")
print("=" * 70)

print("\nnn.Module is the base class for ALL neural networks in PyTorch.")
print("Benefits:")
print("  - Automatic parameter management")
print("  - Easy to save/load models")
print("  - Clean, organized code")
print("  - Integrates seamlessly with PyTorch ecosystem")

# Simple example: custom model
class SimpleNet(nn.Module):
    """
    A simple neural network using nn.Module
    
    All PyTorch models inherit from nn.Module and must implement:
    1. __init__: Define layers
    2. forward: Define forward pass
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Always call parent constructor first
        super(SimpleNet, self).__init__()
        
        # Define layers
        # nn.Linear performs: y = xW^T + b (fully connected layer)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Define the forward pass
        
        PyTorch automatically computes backward pass for us!
        
        Args:
            x: Input tensor, shape (batch_size, input_size)
        
        Returns:
            out: Output tensor, shape (batch_size, output_size)
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

print("\nExample: SimpleNet class defined")
print("Key methods:")
print("  - __init__: Initialize layers")
print("  - forward: Define computation")

# ==============================================================================
# STEP 2: Understanding nn.Sequential
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Understanding nn.Sequential")
print("=" * 70)

print("\nFor simple sequential models, nn.Sequential is even easier!")
print("It automatically chains layers together.\n")

# Same network using nn.Sequential
class SimpleNetSequential(nn.Module):
    """
    Same network using nn.Sequential
    
    nn.Sequential chains operations in order.
    More concise for simple architectures!
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetSequential, self).__init__()
        
        # Define entire network in one go
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

print("Comparison:")
print("  Regular nn.Module:  More flexible, explicit")
print("  nn.Sequential:      More concise, for simple chains")
print("\nBoth are valid! Use what fits your needs.")

# ==============================================================================
# STEP 3: Generate Data
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Generating Data")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

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

print(f"Generated {n_samples} samples")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ==============================================================================
# STEP 4: Create Model
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Creating the Model")
print("=" * 70)

input_size = 2
hidden_size = 16  # Increased from 8 for better performance
output_size = 1

# Create model and move to device
model = SimpleNet(input_size, hidden_size, output_size).to(device)

print(f"Model created: {model.__class__.__name__}")
print("\nModel architecture:")
print(model)

# Useful model methods
print("\nUseful model information:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# ==============================================================================
# STEP 5: Define Loss and Optimizer
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Loss Function and Optimizer")
print("=" * 70)

# Use PyTorch's built-in loss function
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
print("Loss function: nn.BCELoss()")
print("  - Equivalent to our manual implementation")
print("  - Optimized and numerically stable")

# Create optimizer
# Optimizer handles parameter updates automatically!
optimizer = optim.Adam(model.parameters(), lr=0.01)
print("\nOptimizer: Adam")
print("  - Adaptive learning rates")
print("  - Works well for most problems")
print("  - model.parameters() automatically gets all trainable params")

print("\nOther popular optimizers:")
print("  - SGD: Stochastic Gradient Descent (classic)")
print("  - RMSprop: Good for RNNs")
print("  - AdamW: Adam with weight decay")

# ==============================================================================
# STEP 6: Training Loop (Professional Style)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Training Loop")
print("=" * 70)

n_epochs = 1000
loss_history = []

print(f"Training for {n_epochs} epochs\n")
print("Notice how clean the training loop is now!")
print("Compare to Tutorial 04:")
print("  Old: Manual grad zeroing, manual parameter updates")
print("  New: optimizer.zero_grad(), optimizer.step()")
print()

for epoch in range(n_epochs):
    # ===========================================================
    # 1. Zero gradients - optimizer handles this!
    # ===========================================================
    optimizer.zero_grad()
    
    # ===========================================================
    # 2. Forward pass
    # ===========================================================
    outputs = model(X)
    
    # ===========================================================
    # 3. Compute loss
    # ===========================================================
    loss = criterion(outputs, y)
    loss_history.append(loss.item())
    
    # ===========================================================
    # 4. Backward pass (compute gradients)
    # ===========================================================
    loss.backward()
    
    # ===========================================================
    # 5. Update parameters - optimizer handles this!
    # ===========================================================
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y).float().mean().item() * 100
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy:.2f}%")

# ==============================================================================
# STEP 7: Evaluation Mode
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Model Evaluation")
print("=" * 70)

# Switch model to evaluation mode
# Important for layers like Dropout and BatchNorm (we'll learn these later)
model.eval()
print("Model switched to evaluation mode: model.eval()")
print("  - Disables dropout (if present)")
print("  - Uses running stats for batch norm (if present)")
print("  - Good practice even if not strictly necessary here")

# Make predictions
with torch.no_grad():  # No gradient computation needed for inference
    y_pred = model(X)
    predictions = (y_pred > 0.5).float()
    final_accuracy = (predictions == y).float().mean().item() * 100

print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
print(f"Final Loss: {loss_history[-1]:.4f}")

# ==============================================================================
# STEP 8: Saving and Loading Models
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Saving and Loading Models")
print("=" * 70)

# Save model
model_path = '/home/claude/pytorch_feedforward_tutorial/simple_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
print("\n  - state_dict(): Contains all model parameters")
print("  - torch.save(): Saves to disk")

# Load model
loaded_model = SimpleNet(input_size, hidden_size, output_size).to(device)
loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
loaded_model.eval()
print("\nModel loaded successfully!")
print("  1. Create new model instance with same architecture")
print("  2. Load saved parameters with load_state_dict()")
print("  3. Set to eval mode")

# Verify loaded model works
with torch.no_grad():
    loaded_predictions = loaded_model(X)
    match = torch.allclose(y_pred, loaded_predictions)
    print(f"\nPredictions match original model: {match} ✓")

# ==============================================================================
# STEP 9: Visualization
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 9: Visualization")
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
ax1.set_title('Training Data')
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
ax3.plot(loss_history, linewidth=2, color='darkblue')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/05_results.png', dpi=100, bbox_inches='tight')
print("Visualization saved as '05_results.png'")
print("\nTraining completed successfully! ✓")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
# 1. nn.Module benefits:
#    - Cleaner code organization
#    - Automatic parameter management
#    - Easy model saving/loading
#    - Integration with PyTorch ecosystem
#
# 2. Professional training loop (5 steps):
#    a) optimizer.zero_grad()  - Clear old gradients
#    b) outputs = model(input) - Forward pass
#    c) loss = criterion(...)  - Compute loss
#    d) loss.backward()        - Backward pass
#    e) optimizer.step()       - Update parameters
#
# 3. Model modes:
#    - model.train(): Training mode (default)
#    - model.eval(): Evaluation mode (for inference)
#
# 4. Saving/Loading:
#    - torch.save(model.state_dict(), path)
#    - model.load_state_dict(torch.load(path))
#
# 5. Code evolution:
#    Tutorial 01-02: Pure tensors, manual everything
#    Tutorial 03:    Manual backprop (educational)
#    Tutorial 04:    Autograd (automatic gradients)
#    Tutorial 05:    nn.Module + Optimizers (professional!)
#
# NEXT STEPS:
# - Tutorial 06: Train on real datasets (MNIST)
# - Tutorial 07: Advanced techniques (regularization, normalization)
# ==============================================================================
