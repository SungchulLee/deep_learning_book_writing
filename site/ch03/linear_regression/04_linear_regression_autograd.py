"""
==============================================================================
04_linear_regression_autograd.py
==============================================================================
DIFFICULTY: ⭐⭐ (Intermediate)

DESCRIPTION:
    Linear regression using PyTorch's automatic differentiation (autograd).
    No more manual gradient computation! Let PyTorch do the calculus.

TOPICS COVERED:
    - Using requires_grad=True for automatic differentiation
    - .backward() for gradient computation
    - Gradient accumulation and zeroing
    - torch.no_grad() context

PREREQUISITES:
    - Tutorial 01 (PyTorch basics with autograd)
    - Tutorial 03 (Manual PyTorch gradients)

LEARNING OBJECTIVES:
    - Use autograd for gradient computation
    - Understand when to zero gradients
    - Use no_grad() context for efficiency
    - Compare with manual gradient code

TIME: ~15 minutes
==============================================================================
"""

import torch
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH AUTOGRAD")
print("=" * 70)

# ============================================================================
# PART 1: GENERATE DATA
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE DATA")
print("=" * 70)

torch.manual_seed(42)

TRUE_W = 2.0
TRUE_B = 1.0
n_samples = 100

# Generate data
X = torch.rand(n_samples) * 20 - 10  # Random values between -10 and 10
noise = torch.randn(n_samples) * 2    # Gaussian noise
y = TRUE_W * X + TRUE_B + noise

print(f"Generated {n_samples} samples")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")

# ============================================================================
# PART 2: INITIALIZE PARAMETERS WITH requires_grad=True
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: INITIALIZE PARAMETERS WITH AUTOGRAD")
print("=" * 70)

# KEY DIFFERENCE: Set requires_grad=True
# This tells PyTorch to track operations on these tensors
# so it can automatically compute gradients
w = torch.tensor([0.0], requires_grad=True)  # ← requires_grad=True!
b = torch.tensor([0.0], requires_grad=True)  # ← requires_grad=True!

print(f"Parameters initialized:")
print(f"  w: {w.item():.4f}, requires_grad={w.requires_grad}")
print(f"  b: {b.item():.4f}, requires_grad={b.requires_grad}")

# ============================================================================
# PART 3: DEFINE MODEL AND LOSS
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DEFINE MODEL AND LOSS")
print("=" * 70)

def model(X, w, b):
    """Linear model: y = w*X + b"""
    return w * X + b

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return torch.mean((y_true - y_pred) ** 2)

print("Model and loss functions defined")
print("Note: Same as before, but now PyTorch tracks operations")

# ============================================================================
# PART 4: TRAINING LOOP WITH AUTOGRAD
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING LOOP WITH AUTOGRAD")
print("=" * 70)

learning_rate = 0.01
n_epochs = 100

loss_history = []
w_history = [w.item()]
b_history = [b.item()]

print(f"Training Configuration:")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: {n_epochs}")
print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12} {'grad_w':<12} {'grad_b':<12}")
print("-" * 75)

for epoch in range(n_epochs):
    # 1. Forward pass: Compute predictions and loss
    #    PyTorch builds a computational graph automatically
    y_pred = model(X, w, b)
    loss = mse_loss(y, y_pred)
    
    # 2. Backward pass: Compute gradients automatically!
    #    This is where the magic happens - no manual gradient formulas!
    loss.backward()  # Computes gradients via backpropagation
    
    # Now w.grad and b.grad contain the gradients
    # (Same values we computed manually before!)
    
    # 3. Update parameters
    #    We use torch.no_grad() because we don't want to track
    #    the parameter update operations in the computational graph
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Store history
    loss_history.append(loss.item())
    w_history.append(w.item())
    b_history.append(b.item())
    
    # 4. Zero gradients for next iteration
    #    CRITICAL: Gradients accumulate by default!
    #    We must zero them before the next backward pass
    w.grad.zero_()
    b.grad.zero_()
    
    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss.item():<12.4f} {w.item():<12.4f} "
              f"{b.item():<12.4f} {w.grad.item():<12.4f} {b.grad.item():<12.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Learned w: {w.item():.4f} (True: {TRUE_W}, Error: {abs(w.item()-TRUE_W):.4f})")
print(f"  Learned b: {b.item():.4f} (True: {TRUE_B}, Error: {abs(b.item()-TRUE_B):.4f})")
print(f"  Final loss: {loss_history[-1]:.4f}")

# ============================================================================
# PART 5: UNDERSTANDING GRADIENT ACCUMULATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: UNDERSTANDING GRADIENT ACCUMULATION")
print("=" * 70)

print("""
Why do we need to zero gradients?

PyTorch accumulates gradients by default. This is useful for some
advanced scenarios (like gradient accumulation for large batches),
but for standard training, we want fresh gradients each iteration.

Example of what happens WITHOUT zeroing:
""")

# Demonstration
x_demo = torch.tensor([2.0], requires_grad=True)

# First backward pass
y = x_demo ** 2
y.backward()
print(f"After first backward: x_demo.grad = {x_demo.grad.item()}")  # Should be 4

# Second backward pass WITHOUT zeroing
y = x_demo ** 2
y.backward()
print(f"After second backward (accumulated): x_demo.grad = {x_demo.grad.item()}")  # 4 + 4 = 8

# Now let's zero and do it again
x_demo.grad.zero_()
y = x_demo ** 2
y.backward()
print(f"After zeroing and third backward: x_demo.grad = {x_demo.grad.item()}")  # Back to 4

print("\nThis is why we call w.grad.zero_() in the training loop!")

# ============================================================================
# PART 6: USING torch.no_grad() CONTEXT
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: UNDERSTANDING torch.no_grad()")
print("=" * 70)

print("""
torch.no_grad() disables gradient tracking temporarily.
Use it when:
1. Making predictions (inference)
2. Updating parameters (as we did in the training loop)
3. Any operation where you don't need gradients

Benefits:
- Saves memory (no computational graph)
- Faster computation
- Prevents accidental gradient computation

Example:
""")

x = torch.tensor([1.0], requires_grad=True)

# With gradient tracking
y = x ** 2
print(f"With gradients: y.requires_grad = {y.requires_grad}")

# Without gradient tracking
with torch.no_grad():
    y_no_grad = x ** 2
    print(f"Inside no_grad: y_no_grad.requires_grad = {y_no_grad.requires_grad}")

print("\nThis is essential for efficient inference and parameter updates!")

# ============================================================================
# PART 7: VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("PART 7: VISUALIZE RESULTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curve
axes[0, 0].plot(loss_history, linewidth=2, color='purple')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training Loss with Autograd')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot 2: Parameter evolution
axes[0, 1].plot(w_history, label='w (slope)', linewidth=2, color='blue')
axes[0, 1].axhline(y=TRUE_W, color='r', linestyle='--', linewidth=2, label=f'True w={TRUE_W}')
axes[0, 1].plot(b_history, label='b (intercept)', linewidth=2, color='green')
axes[0, 1].axhline(y=TRUE_B, color='orange', linestyle='--', linewidth=2, label=f'True b={TRUE_B}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Parameter Value')
axes[0, 1].set_title('Parameter Convergence (Autograd)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Final fit
with torch.no_grad():  # No gradients needed for visualization
    X_sorted, _ = torch.sort(X)
    y_pred_sorted = model(X_sorted, w, b)

axes[1, 0].scatter(X.numpy(), y.numpy(), alpha=0.5, s=20, label='Data')
axes[1, 0].plot(X_sorted.numpy(), (TRUE_W * X_sorted + TRUE_B).numpy(), 
                'r--', linewidth=2, label=f'True: y={TRUE_W}x+{TRUE_B}')
axes[1, 0].plot(X_sorted.numpy(), y_pred_sorted.numpy(), 
                'g-', linewidth=2, label=f'Learned: y={w.item():.2f}x+{b.item():.2f}')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Data with Learned Model')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Comparison table
comparison_text = f"""
AUTOGRAD VS MANUAL GRADIENTS

Code Complexity:
  Manual:   5+ lines for gradient formulas
  Autograd: 1 line (loss.backward())

Flexibility:
  Manual:   Hard to extend
  Autograd: Works for any function

Errors:
  Manual:   Easy to make mistakes
  Autograd: Automatic, no mistakes

Performance:
  Manual:   Similar
  Autograd: Highly optimized

Results:
  Final w: {w.item():.4f} (Error: {abs(w.item()-TRUE_W):.4f})
  Final b: {b.item():.4f} (Error: {abs(b.item()-TRUE_B):.4f})
"""
axes[1, 1].text(0.1, 0.95, comparison_text, 
                transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/04_autograd_results.png', dpi=100)
print("Saved visualization to: 04_autograd_results.png")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key Takeaways:

1. AUTOGRAD BASICS:
   - Set requires_grad=True on parameters
   - PyTorch builds computational graph automatically
   - Call .backward() to compute all gradients

2. TRAINING LOOP STRUCTURE:
   for epoch in range(n_epochs):
       # Forward pass
       y_pred = model(X, w, b)
       loss = loss_function(y, y_pred)
       
       # Backward pass
       loss.backward()  # ← Computes gradients automatically
       
       # Update parameters
       with torch.no_grad():
           w -= learning_rate * w.grad
           b -= learning_rate * b.grad
       
       # Zero gradients
       w.grad.zero_()
       b.grad.zero_()

3. IMPORTANT POINTS:
   ✓ Always zero gradients before backward()
   ✓ Use torch.no_grad() for parameter updates
   ✓ Gradients accumulate by default
   ✓ Same results as manual computation

4. ADVANTAGES:
   ✓ No manual gradient formulas
   ✓ Less error-prone
   ✓ Works for any differentiable function
   ✓ Scales to complex models

Next Steps:
- Tutorial 05: Use nn.Module for cleaner code
- Tutorial 06: Multiple input features
- Tutorial 07: Polynomial regression
""")
