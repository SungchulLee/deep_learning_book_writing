"""
07_loss_functions.py - Choosing the Right Loss Function

Learn about different loss functions and when to use them.
Understanding loss functions is crucial for successful training!

TIME: 25-30 minutes | DIFFICULTY: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("Loss Functions Guide")
print("="*70)

# Create sample predictions and targets
y_true = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
y_pred = torch.tensor([0.1, 1.5, 1.8, 3.2, 3.9])

print("\nREGRESSION LOSSES")
print("-"*70)

# MSE Loss (L2)
mse = nn.MSELoss()
print(f"MSE Loss: {mse(y_pred, y_true):.4f}")
print("  Formula: mean((pred - true)^2)")
print("  Use: Regression, penalizes large errors heavily")

# MAE Loss (L1)
mae = nn.L1Loss()
print(f"\nMAE Loss: {mae(y_pred, y_true):.4f}")
print("  Formula: mean(|pred - true|)")
print("  Use: Regression with outliers, more robust than MSE")

# Smooth L1 Loss (Huber)
smooth_l1 = nn.SmoothL1Loss()
print(f"\nSmooth L1 Loss: {smooth_l1(y_pred, y_true):.4f}")
print("  Formula: Combines L1 and L2")
print("  Use: Robust to outliers, used in object detection")

print("\n" + "="*70)
print("CLASSIFICATION LOSSES")
print("-"*70)

# Binary Classification
logits_binary = torch.tensor([[0.8], [-0.5], [1.2], [-2.0]])
targets_binary = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

# BCE with Logits (recommended for binary classification)
bce_with_logits = nn.BCEWithLogitsLoss()
print(f"BCE with Logits: {bce_with_logits(logits_binary, targets_binary):.4f}")
print("  Use: Binary classification (includes sigmoid)")
print("  More numerically stable than BCE alone")

# Multi-class Classification
logits_multi = torch.randn(4, 3)  # 4 samples, 3 classes
targets_multi = torch.tensor([0, 1, 2, 1])  # Class indices

# Cross Entropy (recommended for multi-class)
ce = nn.CrossEntropyLoss()
print(f"\nCross Entropy: {ce(logits_multi, targets_multi):.4f}")
print("  Use: Multi-class classification (includes log_softmax)")
print("  Most common for classification tasks")

print("\n" + "="*70)
print("LOSS FUNCTION SELECTION GUIDE")
print("="*70)
print("""
PROBLEM TYPE          | RECOMMENDED LOSS      | OUTPUT ACTIVATION
----------------------|----------------------|-------------------
Regression            | MSELoss              | None (linear)
Regression (outliers) | L1Loss / SmoothL1    | None
Binary Classification | BCEWithLogitsLoss    | None (logits)
Multi-class           | CrossEntropyLoss     | None (logits)
Multi-label           | BCEWithLogitsLoss    | None (logits)

KEY POINTS:
✓ Use *WithLogits versions - they're more stable
✓ Don't apply activation before these losses
✓ CrossEntropyLoss expects class indices, not one-hot
✓ MSE good for regression, CrossEntropy for classification
✓ L1 more robust to outliers than MSE

COMMON MISTAKES:
✗ Applying sigmoid before BCEWithLogitsLoss
✗ Applying softmax before CrossEntropyLoss  
✗ Using MSE for classification
✗ Using CrossEntropy for regression
""")

# Visualize loss behaviors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# L1 vs L2 behavior
errors = np.linspace(-3, 3, 100)
l1_loss = np.abs(errors)
l2_loss = errors**2

ax1.plot(errors, l1_loss, label='L1 (MAE)', linewidth=2)
ax1.plot(errors, l2_loss, label='L2 (MSE)', linewidth=2)
ax1.set_xlabel('Prediction Error', fontsize=12)
ax1.set_ylabel('Loss Value', fontsize=12)
ax1.set_title('L1 vs L2 Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cross Entropy visualization
probs = np.linspace(0.01, 0.99, 100)
ce_loss = -np.log(probs)

ax2.plot(probs, ce_loss, linewidth=2, color='red')
ax2.set_xlabel('Predicted Probability (for true class)', fontsize=12)
ax2.set_ylabel('Cross Entropy Loss', fontsize=12)
ax2.set_title('Cross Entropy Loss Behavior', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_loss_functions.png', dpi=150)
print("\nLoss visualizations saved!")

print("\nEXERCISES:")
print("1. Implement custom loss function for imbalanced data")
print("2. Compare MSE vs MAE on dataset with outliers")
print("3. Create weighted cross entropy for class imbalance")
print("4. Visualize gradient magnitude for different losses")
plt.show()
