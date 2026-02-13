#!/usr/bin/env python3
"""
==============================================================================
BEGINNER TUTORIAL 05: Complete Linear Regression Training Loop
==============================================================================

LEARNING OBJECTIVES:
-------------------
1. Build a complete training loop from scratch
2. Understand the standard training cycle: forward → loss → backward → update
3. Learn proper gradient management in training
4. Visualize training progress
5. Understand parameter updates without autograd

KEY CONCEPTS:
------------
- Training Loop: forward, loss, backward, update (repeat)
- Gradient Zeroing: Essential before each backward pass
- torch.no_grad(): Context for parameter updates
- Loss Tracking: Monitor convergence
- Synthetic Data: Generate clean test data

REAL-WORLD APPLICATION:
-----------------------
This is the foundation of ALL deep learning training loops!
Understanding this simple example prepares you for complex neural networks.

==============================================================================
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic_data(n_samples=100, true_w=2.0, true_b=1.0, noise_std=0.2):
    """
    Generate synthetic linear regression data: y = w*x + b + noise
    
    Args:
        n_samples: Number of data points
        true_w: True slope (weight)
        true_b: True intercept (bias)
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        x, y: Input and target tensors
    """
    x = torch.randn(n_samples, 1)  # Random inputs from N(0,1)
    noise = noise_std * torch.randn(n_samples, 1)  # Gaussian noise
    y = true_w * x + true_b + noise  # True relationship + noise
    return x, y


def main():
    """
    Complete example of training a linear regression model from scratch.
    We'll predict y from x, learning parameters w (weight) and b (bias).
    """
    
    print("="*70)
    print("LINEAR REGRESSION TRAINING FROM SCRATCH")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ==========================
    # STEP 1: Generate Data
    # ==========================
    print("\nSTEP 1: Generating Synthetic Data")
    print("-" * 40)
    
    # True parameters we're trying to learn
    TRUE_W = 2.0  # True slope
    TRUE_B = 1.0  # True intercept
    N_SAMPLES = 100
    
    x, y = generate_synthetic_data(N_SAMPLES, TRUE_W, TRUE_B, noise_std=0.2)
    
    print(f"Generated {N_SAMPLES} samples")
    print(f"True model: y = {TRUE_W}*x + {TRUE_B} + noise")
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # ==========================
    # STEP 2: Initialize Parameters
    # ==========================
    print("\nSTEP 2: Initializing Model Parameters")
    print("-" * 40)
    
    # Initialize parameters randomly
    # These are LEAF tensors with gradient tracking enabled
    w = torch.randn(1, requires_grad=True)  # Weight (slope)
    b = torch.zeros(1, requires_grad=True)  # Bias (intercept)
    
    print(f"Initial w: {w.item():.4f} (target: {TRUE_W})")
    print(f"Initial b: {b.item():.4f} (target: {TRUE_B})")
    
    # ==========================
    # STEP 3: Set Hyperparameters
    # ==========================
    print("\nSTEP 3: Setting Hyperparameters")
    print("-" * 40)
    
    learning_rate = 0.1
    num_epochs = 200
    
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    
    # ==========================
    # STEP 4: Training Loop
    # ==========================
    print("\nSTEP 4: Training Loop")
    print("-" * 40)
    print("Starting training...\n")
    
    # Track loss for visualization
    loss_history = []
    
    for epoch in range(num_epochs):
        # ========================================
        # FORWARD PASS: Compute predictions
        # ========================================
        # Our model: y_hat = w*x + b
        y_pred = x * w + b  # Broadcasting handles the shapes
        
        # Compute loss: Mean Squared Error (MSE)
        # MSE = mean((predictions - targets)^2)
        loss = torch.mean((y_pred - y) ** 2)
        
        # Store loss for plotting
        loss_history.append(loss.item())
        
        # ========================================
        # BACKWARD PASS: Compute gradients
        # ========================================
        # CRITICAL: Zero gradients before backward!
        # Gradients accumulate by default, so we must clear them
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        
        # Compute gradients via backpropagation
        # This populates w.grad and b.grad
        loss.backward()
        
        # ========================================
        # UPDATE STEP: Gradient descent
        # ========================================
        # Update rule: θ_new = θ_old - learning_rate * gradient
        # We use torch.no_grad() because:
        # 1. We don't want these operations in the computation graph
        # 2. It's faster and uses less memory
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        # ========================================
        # LOGGING: Print progress
        # ========================================
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs}: "
                  f"Loss = {loss.item():.6f} | "
                  f"w = {w.item():.4f} | "
                  f"b = {b.item():.4f}")
    
    # ==========================
    # STEP 5: Final Results
    # ==========================
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    print(f"\nTrue parameters:     w = {TRUE_W:.4f}, b = {TRUE_B:.4f}")
    print(f"Learned parameters:  w = {w.item():.4f}, b = {b.item():.4f}")
    print(f"Error in w: {abs(w.item() - TRUE_W):.4f}")
    print(f"Error in b: {abs(b.item() - TRUE_B):.4f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    # ==========================
    # STEP 6: Visualization
    # ==========================
    print("\nGenerating visualization...")
    
    # Convert to numpy for plotting
    x_np = x.detach().numpy().flatten()
    y_np = y.detach().numpy().flatten()
    
    # Generate fitted line
    x_sorted = np.sort(x_np)
    y_fitted = w.item() * x_sorted + b.item()
    y_true = TRUE_W * x_sorted + TRUE_B
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ----- Plot 1: Data and Fitted Line -----
    ax1 = axes[0]
    ax1.scatter(x_np, y_np, alpha=0.6, s=30, label='Training Data', color='blue')
    ax1.plot(x_sorted, y_fitted, 'r-', linewidth=3, 
             label=f'Learned: y={w.item():.2f}x+{b.item():.2f}')
    ax1.plot(x_sorted, y_true, 'g--', linewidth=2, 
             label=f'True: y={TRUE_W}x+{TRUE_B}')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Linear Regression: Data and Fitted Line', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ----- Plot 2: Training Loss Curve -----
    ax2 = axes[1]
    ax2.plot(range(num_epochs), loss_history, linewidth=2, color='purple')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale shows convergence better
    
    plt.tight_layout()
    plt.savefig('/home/claude/linear_regression_training.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'linear_regression_training.png'")
    
    # ==========================
    # UNDERSTANDING SECTION
    # ==========================
    print("\n" + "="*70)
    print("UNDERSTANDING THE TRAINING LOOP")
    print("="*70)
    print("""
    The Training Cycle (repeated for each epoch):
    
    1. FORWARD PASS
       └─ Compute predictions: y_pred = w*x + b
       └─ Compute loss: MSE = mean((y_pred - y)^2)
    
    2. BACKWARD PASS  
       └─ Zero old gradients: w.grad.zero_(), b.grad.zero_()
       └─ Compute new gradients: loss.backward()
    
    3. UPDATE PARAMETERS
       └─ Gradient descent: w = w - lr * w.grad
       └─                   b = b - lr * b.grad
    
    Why torch.no_grad() for updates?
    • Parameter updates shouldn't be part of the computation graph
    • Saves memory and computational overhead
    • Updates are in-place operations on the parameter values
    
    Why zero gradients?
    • Gradients accumulate by default in PyTorch
    • Old gradients must be cleared before computing new ones
    • Forgetting this leads to incorrect parameter updates!
    
    Learning Rate:
    • Too large: Unstable training, might diverge
    • Too small: Slow convergence
    • 0.1 works well for this simple problem
    """)
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ Training loop structure: forward → loss → backward → update
    ✓ Always zero gradients before backward pass
    ✓ Use torch.no_grad() for parameter updates
    ✓ Track loss to monitor convergence
    ✓ Visualization helps understand model quality
    ✓ This pattern extends to ALL neural network training!
    """)


if __name__ == "__main__":
    main()
