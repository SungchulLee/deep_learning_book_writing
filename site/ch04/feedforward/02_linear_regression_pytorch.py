"""
==============================================================================
Tutorial 02: Linear Regression with PyTorch Tensors
==============================================================================
DIFFICULTY: ⭐ Beginner

WHAT YOU'LL LEARN:
- PyTorch tensors (GPU-accelerated arrays)
- Converting NumPy code to PyTorch
- Benefits of using PyTorch tensors
- Device management (CPU vs GPU)

PREREQUISITES:
- Tutorial 01 (understanding of gradient descent)
- Basic Python

KEY CONCEPTS:
- PyTorch tensors vs NumPy arrays
- Tensor operations
- Device placement (CPU/GPU)
- torch.no_grad() context manager
==============================================================================
"""

import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# ==============================================================================
# STEP 0: Device Configuration
# ==============================================================================
# PyTorch can run on CPU or GPU. Using GPU can speed up training significantly!
print("=" * 70)
print("STEP 0: Device Configuration")
print("=" * 70)

# Check if CUDA (NVIDIA GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("GPU not available, using CPU")

# ==============================================================================
# STEP 1: Generate Synthetic Data (now with PyTorch!)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Generating Synthetic Data with PyTorch Tensors")
print("=" * 70)

n_samples = 100

# Create tensors directly (similar to NumPy, but can run on GPU)
# torch.randn() creates random numbers from normal distribution
# torch.rand() creates random numbers from uniform distribution [0, 1)
X = torch.rand(n_samples, 1, device=device) * 10  # Random x values [0, 10)

# Generate y values: y = 2x + 1 + noise
true_w = 2.0
true_b = 1.0
noise = torch.randn(n_samples, 1, device=device) * 0.5
y = true_w * X + true_b + noise

print(f"Generated {n_samples} data points on {device}")
print(f"True weight: {true_w}, True bias: {true_b}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X dtype: {X.dtype}")  # Usually torch.float32
print(f"Sample data - X[0]: {X[0].item():.2f}, y[0]: {y[0].item():.2f}")

# ==============================================================================
# COMPARISON: NumPy vs PyTorch Tensors
# ==============================================================================
print("\n" + "=" * 70)
print("NumPy vs PyTorch Comparison")
print("=" * 70)
print("NumPy Array:")
print("  - Lives on CPU only")
print("  - import numpy as np")
print("  - Create: np.array([1, 2, 3])")
print("\nPyTorch Tensor:")
print("  - Can live on CPU or GPU")
print("  - import torch")
print("  - Create: torch.tensor([1, 2, 3])")
print("  - Move to GPU: tensor.to('cuda')")
print("  - Get value: tensor.item() (for scalar tensors)")

# ==============================================================================
# STEP 2: Initialize Model Parameters
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Initializing Model Parameters")
print("=" * 70)

# Create parameters as tensors
# requires_grad=True tells PyTorch to track operations for automatic differentiation
# We'll use manual gradients here, but this prepares you for Tutorial 04
w = torch.randn(1, 1, device=device)
b = torch.zeros(1, 1, device=device)

print(f"Initial weight: {w.item():.4f}")
print(f"Initial bias: {b.item():.4f}")
print(f"Parameters created on: {w.device}")

# ==============================================================================
# STEP 3: Define Model Functions (PyTorch version)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Defining Model Functions")
print("=" * 70)

def forward(X, w, b):
    """
    Forward pass using PyTorch tensors
    
    Note: The @ operator works the same as in NumPy
    PyTorch also supports: torch.matmul(X, w)
    
    Args:
        X: Input tensor, shape (n_samples, 1)
        w: Weight tensor, shape (1, 1)
        b: Bias tensor, shape (1, 1)
    
    Returns:
        y_pred: Prediction tensor, shape (n_samples, 1)
    """
    return X @ w + b

def compute_loss(y_true, y_pred):
    """
    Compute MSE loss using PyTorch operations
    
    PyTorch has built-in loss functions, but we implement
    it manually here to show the math.
    
    Note: torch.mean() is equivalent to np.mean()
          torch.sum() is equivalent to np.sum()
    
    Args:
        y_true: True labels, shape (n_samples, 1)
        y_pred: Predictions, shape (n_samples, 1)
    
    Returns:
        loss: Scalar tensor
    """
    n = y_true.shape[0]
    return (1 / n) * torch.sum((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):
    """
    Compute gradients manually (same math as Tutorial 01)
    
    In Tutorial 04, we'll let PyTorch compute these automatically!
    
    Args:
        X: Input tensor, shape (n_samples, 1)
        y_true: True labels, shape (n_samples, 1)
        y_pred: Predictions, shape (n_samples, 1)
    
    Returns:
        dw: Gradient w.r.t. weight
        db: Gradient w.r.t. bias
    """
    n = y_true.shape[0]
    
    # Note: .T is the transpose operation (same as in NumPy)
    dw = (2 / n) * X.T @ (y_pred - y_true)
    db = (2 / n) * torch.sum(y_pred - y_true)
    
    return dw, db

print("Model functions defined (now using PyTorch operations)")

# ==============================================================================
# STEP 4: Training Loop
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Training the Model")
print("=" * 70)

learning_rate = 0.01
n_epochs = 100
loss_history = []

print(f"Training for {n_epochs} epochs with learning rate {learning_rate}\n")

for epoch in range(n_epochs):
    # Forward pass
    y_pred = forward(X, w, b)
    
    # Compute loss
    loss = compute_loss(y, y_pred)
    loss_history.append(loss.item())  # .item() converts tensor to Python number
    
    # Compute gradients
    dw, db = compute_gradients(X, y, y_pred)
    
    # Update parameters
    # Note: We use torch.no_grad() to tell PyTorch not to track these operations
    # This is important for memory efficiency (we'll explain more in Tutorial 04)
    with torch.no_grad():
        w -= learning_rate * dw
        b -= learning_rate * db
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, "
              f"w: {w.item():.4f}, b: {b.item():.4f}")

# ==============================================================================
# STEP 5: Evaluate Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Final Results")
print("=" * 70)

print(f"\nTrue values:    w = {true_w:.4f}, b = {true_b:.4f}")
print(f"Learned values: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")

# ==============================================================================
# STEP 6: Visualize Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Visualizing Results")
print("=" * 70)

# For plotting, we need to move data back to CPU and convert to NumPy
X_cpu = X.cpu().numpy()
y_cpu = y.cpu().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Data and fitted line
ax1.scatter(X_cpu, y_cpu, alpha=0.5, label='Training data')
X_line = torch.linspace(0, 10, 100, device=device).reshape(-1, 1)
y_line = forward(X_line, w, b)
# Convert predictions to NumPy for plotting
ax1.plot(X_line.cpu().numpy(), y_line.cpu().numpy(), 'r-', 
         linewidth=2, label=f'Fitted line: y = {w.item():.2f}x + {b.item():.2f}')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('PyTorch Linear Regression')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: Loss over epochs
ax2.plot(loss_history, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (MSE)')
ax2.set_title('Training Loss Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/02_results.png', dpi=100, bbox_inches='tight')
print("Visualization saved as '02_results.png'")
print("\nTraining completed successfully! ✓")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
# 1. PyTorch tensors are similar to NumPy arrays but:
#    - Can run on GPU for faster computation
#    - Support automatic differentiation (we'll use this in Tutorial 04)
#    - Are designed for deep learning
#
# 2. Common PyTorch tensor operations:
#    - Creation: torch.randn(), torch.zeros(), torch.ones()
#    - Math: +, -, *, /, @, torch.sum(), torch.mean()
#    - Device: .to(device), .cpu(), .cuda()
#    - Conversion: .numpy() (to NumPy), .item() (to Python number)
#
# 3. torch.no_grad() context:
#    - Disables gradient tracking for operations inside the block
#    - Saves memory and speeds up computation
#    - Use during inference or manual parameter updates
#
# 4. Device management:
#    - Create tensors on device directly: torch.randn(..., device=device)
#    - Or move existing tensors: tensor.to(device)
#    - All tensors in an operation must be on the same device!
#
# NEXT STEPS:
# - Tutorial 03: Build a multi-layer neural network
# - Tutorial 04: Use PyTorch's automatic differentiation (autograd)
# ==============================================================================
