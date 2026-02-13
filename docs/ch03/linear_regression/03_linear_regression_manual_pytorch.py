"""
==============================================================================
03_linear_regression_manual_pytorch.py
==============================================================================
DIFFICULTY: ⭐⭐ (Beginner-Intermediate)

DESCRIPTION:
    Linear regression using PyTorch tensors but manually computing gradients.
    This bridges NumPy and full PyTorch, showing how tensors work while
    maintaining control over gradient computation.

TOPICS COVERED:
    - Converting NumPy code to PyTorch tensors
    - Manual gradient computation with tensors
    - Understanding tensor operations
    - Comparing NumPy vs PyTorch approaches

PREREQUISITES:
    - Tutorial 01 (PyTorch basics)
    - Tutorial 02 (Linear regression with NumPy)

LEARNING OBJECTIVES:
    - Use PyTorch tensors for computation
    - Manually compute gradients (still no autograd)
    - Understand tensor operations vs NumPy operations
    - Appreciate what autograd will automate

TIME: ~20 minutes
==============================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH PYTORCH TENSORS (MANUAL GRADIENTS)")
print("=" * 70)

# ============================================================================
# PART 1: GENERATE SYNTHETIC DATA
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE SYNTHETIC DATA")
print("=" * 70)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# True parameters
TRUE_W = 2.0
TRUE_B = 1.0

# Generate data using NumPy first
n_samples = 100
X_numpy = np.random.uniform(-10, 10, n_samples)
noise = np.random.normal(0, 2, n_samples)
y_numpy = TRUE_W * X_numpy + TRUE_B + noise

# Convert to PyTorch tensors
# dtype=torch.float32 is the standard for most PyTorch operations
X = torch.from_numpy(X_numpy).float()  # Convert to float32 tensor
y = torch.from_numpy(y_numpy).float()

print(f"Generated {n_samples} data points")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")
print(f"\nData types:")
print(f"  X: {X.dtype}, shape: {X.shape}")
print(f"  y: {y.dtype}, shape: {y.shape}")
print(f"\nFirst 5 samples:")
print(f"X[:5] = {X[:5]}")
print(f"y[:5] = {y[:5]}")

# ============================================================================
# PART 2: DEFINE THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DEFINE THE MODEL")
print("=" * 70)

def predict(X, w, b):
    """
    Linear model: y_pred = w * X + b
    
    Args:
        X: Input features (tensor)
        w: Weight parameter (tensor)
        b: Bias parameter (tensor)
    
    Returns:
        y_pred: Predictions (tensor)
    """
    return w * X + b

def compute_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss
    
    Args:
        y_true: True values (tensor)
        y_pred: Predictions (tensor)
    
    Returns:
        loss: MSE value (scalar tensor)
    """
    n = y_true.shape[0]  # Number of samples
    loss = (1 / n) * torch.sum((y_true - y_pred) ** 2)
    return loss

# Initialize parameters as tensors
w = torch.tensor([0.0], dtype=torch.float32)
b = torch.tensor([0.0], dtype=torch.float32)

print(f"Initialized parameters:")
print(f"  w: {w}, shape: {w.shape}, dtype: {w.dtype}")
print(f"  b: {b}, shape: {b.shape}, dtype: {b.dtype}")

# Test initial predictions
y_pred_init = predict(X, w, b)
initial_loss = compute_loss(y, y_pred_init)

print(f"\nInitial predictions:")
print(f"  y_pred[:5]: {y_pred_init[:5]}")
print(f"  Initial loss: {initial_loss.item():.4f}")

# ============================================================================
# PART 3: MANUAL GRADIENT COMPUTATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: MANUAL GRADIENT COMPUTATION")
print("=" * 70)

def compute_gradients(X, y_true, y_pred):
    """
    Manually compute gradients of MSE with respect to w and b
    
    This is identical to the NumPy version, but uses PyTorch tensors
    
    ∂Loss/∂w = (2/n) * sum((y_pred - y_true) * X)
    ∂Loss/∂b = (2/n) * sum(y_pred - y_true)
    
    Args:
        X: Input features (tensor)
        y_true: True values (tensor)
        y_pred: Predictions (tensor)
    
    Returns:
        grad_w: Gradient w.r.t. weight (tensor)
        grad_b: Gradient w.r.t. bias (tensor)
    """
    n = X.shape[0]
    error = y_pred - y_true
    
    # Compute gradients using PyTorch operations
    grad_w = (2.0 / n) * torch.sum(error * X)
    grad_b = (2.0 / n) * torch.sum(error)
    
    return grad_w, grad_b

# Test gradient computation
grad_w, grad_b = compute_gradients(X, y, y_pred_init)
print(f"Initial gradients:")
print(f"  grad_w: {grad_w.item():.4f}")
print(f"  grad_b: {grad_b.item():.4f}")

# ============================================================================
# PART 4: TRAINING LOOP
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING LOOP")
print("=" * 70)

# Reset parameters
w = torch.tensor([0.0], dtype=torch.float32)
b = torch.tensor([0.0], dtype=torch.float32)

# Hyperparameters
learning_rate = 0.01
n_epochs = 100

# Tracking
loss_history = []
w_history = [w.item()]
b_history = [b.item()]

print(f"Training Configuration:")
print(f"  Learning rate: {learning_rate}")
print(f"  Number of epochs: {n_epochs}")
print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12}")
print("-" * 50)

for epoch in range(n_epochs):
    # 1. Forward pass
    y_pred = predict(X, w, b)
    
    # 2. Compute loss
    loss = compute_loss(y, y_pred)
    loss_history.append(loss.item())  # .item() converts tensor to Python number
    
    # 3. Compute gradients manually
    grad_w, grad_b = compute_gradients(X, y, y_pred)
    
    # 4. Update parameters
    # Note: We need to detach from any potential computational graph
    # Even though we're not using autograd yet, it's good practice
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    
    # Store history
    w_history.append(w.item())
    b_history.append(b.item())
    
    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss.item():<12.4f} {w.item():<12.4f} {b.item():<12.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Learned w: {w.item():.4f} (True: {TRUE_W})")
print(f"  Learned b: {b.item():.4f} (True: {TRUE_B})")
print(f"  Final loss: {loss_history[-1]:.4f}")
print(f"  Initial loss: {loss_history[0]:.4f}")

# ============================================================================
# PART 5: COMPARISON - PYTORCH VS NUMPY OPERATIONS
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: PYTORCH OPERATIONS - KEY DIFFERENCES FROM NUMPY")
print("=" * 70)

print("""
PyTorch Tensors vs NumPy Arrays:

Similarities:
1. Similar API: Most operations have the same names
2. Same mathematical operations
3. Indexing and slicing work similarly

Key Differences:

1. Device Support:
   - PyTorch: Can run on CPU or GPU
   - NumPy: CPU only
   Example:
     tensor_cpu = torch.tensor([1, 2, 3])
     tensor_gpu = tensor_cpu.to('cuda')  # Move to GPU if available

2. Automatic Differentiation:
   - PyTorch: Built-in autograd (we'll use in next tutorial)
   - NumPy: Must compute gradients manually
   
3. Data Type:
   - PyTorch: Default float is float32
   - NumPy: Default float is float64
   
4. In-place Operations:
   - PyTorch: Operations ending with _ are in-place
   - NumPy: Usually requires explicit assignment
   Example:
     x.add_(1)  # Adds 1 in-place (PyTorch)
     x = x + 1  # Creates new array (NumPy)

5. Item Extraction:
   - PyTorch: Use .item() to get Python scalar
   - NumPy: Direct access or .item()
   Example:
     scalar = tensor.item()  # PyTorch
     scalar = array[0]       # NumPy
""")

# ============================================================================
# PART 6: VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: VISUALIZE RESULTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curve
axes[0, 0].plot(loss_history, linewidth=2, color='blue')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training Loss (PyTorch Implementation)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Parameter evolution
axes[0, 1].plot(w_history, label='w (slope)', linewidth=2)
axes[0, 1].axhline(y=TRUE_W, color='r', linestyle='--', label=f'True w={TRUE_W}')
axes[0, 1].plot(b_history, label='b (intercept)', linewidth=2)
axes[0, 1].axhline(y=TRUE_B, color='g', linestyle='--', label=f'True b={TRUE_B}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Parameter Value')
axes[0, 1].set_title('Parameter Convergence')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Final fit
X_sorted, indices = torch.sort(X)
y_sorted = y[indices]
y_pred_final = predict(X_sorted, w, b)

# Convert to numpy for plotting
X_sorted_np = X_sorted.numpy()
y_sorted_np = y_sorted.numpy()
y_pred_final_np = y_pred_final.detach().numpy()

axes[1, 0].scatter(X.numpy(), y.numpy(), alpha=0.5, label='Data')
axes[1, 0].plot(X_sorted_np, TRUE_W * X_sorted_np + TRUE_B, 'r-', 
                linewidth=2, label=f'True: y={TRUE_W}x+{TRUE_B}')
axes[1, 0].plot(X_sorted_np, y_pred_final_np, 'g-', 
                linewidth=2, label=f'Learned: y={w.item():.2f}x+{b.item():.2f}')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Data with Learned Model')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Learning rate sensitivity
axes[1, 1].text(0.5, 0.6, 'Key Insights:', ha='center', fontsize=14, weight='bold')
axes[1, 1].text(0.5, 0.5, f'Final w: {w.item():.4f} (error: {abs(w.item()-TRUE_W):.4f})', 
                ha='center', fontsize=11)
axes[1, 1].text(0.5, 0.4, f'Final b: {b.item():.4f} (error: {abs(b.item()-TRUE_B):.4f})', 
                ha='center', fontsize=11)
axes[1, 1].text(0.5, 0.3, f'Loss reduction: {((loss_history[0]-loss_history[-1])/loss_history[0]*100):.1f}%', 
                ha='center', fontsize=11)
axes[1, 1].text(0.5, 0.1, 'Next: Autograd will compute\ngradients automatically!', 
                ha='center', fontsize=10, style='italic')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/03_pytorch_manual.png', dpi=100)
print("Saved visualization to: 03_pytorch_manual.png")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
What We Learned:
1. PyTorch tensors are similar to NumPy arrays but with extra features
2. Same mathematical operations work on tensors
3. Gradient computation is still manual (for now!)
4. Code structure is identical to NumPy version

Advantages of PyTorch Tensors:
✓ GPU acceleration (we didn't use it here, but it's available)
✓ Automatic differentiation (coming in next tutorial!)
✓ Part of a larger ecosystem for neural networks
✓ Optimized C++/CUDA backends

Why This Approach:
- We computed gradients manually to understand the math
- In the next tutorial, autograd will do this for us
- Understanding manual computation helps debug models
- Shows that PyTorch doesn't change the underlying math

Next Steps:
- Tutorial 04: Use autograd for automatic gradient computation
- No more manual gradient formulas!
- Focus on model architecture and training
""")
