"""
================================================================================
BEGINNER 02: Common Regression Loss Functions
================================================================================

WHAT YOU'LL LEARN:
- Different loss functions for regression tasks
- When to use MSE vs MAE vs Huber Loss
- How different losses handle outliers
- Visualizing the difference between loss functions

PREREQUISITES:
- Complete 01_intro_to_loss_functions.py
- Understand basic regression concepts

TIME TO COMPLETE: ~15 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("COMMON REGRESSION LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: Sample Data with an Outlier
# ============================================================================
print("\n" + "-" * 80)
print("SAMPLE DATA: Predicting Test Scores")
print("-" * 80)

# Actual test scores
actual_scores = torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0])  # Note: 15 is an outlier!
print(f"Actual scores: {actual_scores}")

# Model predictions (pretty close except for the outlier)
predicted_scores = torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])
print(f"Predicted scores: {predicted_scores}")

# Show errors
errors = actual_scores - predicted_scores
print(f"Errors: {errors}")
print("\nNote: The 5th student has an error of -72 (a huge outlier!)")
print("Let's see how different loss functions handle this...")

# ============================================================================
# SECTION 2: Mean Squared Error (MSE) - L2 Loss
# ============================================================================
print("\n" + "-" * 80)
print("1. MEAN SQUARED ERROR (MSE) - L2 Loss")
print("-" * 80)

mse_criterion = nn.MSELoss()
mse_loss = mse_criterion(predicted_scores, actual_scores)

print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"RMSE (Root MSE): {torch.sqrt(mse_loss).item():.4f}")

print("\nCHARACTERISTICS:")
print("✓ Most commonly used for regression")
print("✓ Differentiable everywhere (smooth gradients)")
print("✓ Sensitive to outliers (squares the error!)")
print(f"✓ Formula: (1/n) × Σ(predicted - actual)²")

# Let's see the individual squared errors
squared_errors = (predicted_scores - actual_scores) ** 2
print(f"\nSquared errors: {squared_errors}")
print(f"Notice how the outlier error {squared_errors[4].item():.0f} dominates!")
print(f"It's {(squared_errors[4] / squared_errors[:4].sum()).item():.1f}x larger than all others combined!")

# ============================================================================
# SECTION 3: Mean Absolute Error (MAE) - L1 Loss
# ============================================================================
print("\n" + "-" * 80)
print("2. MEAN ABSOLUTE ERROR (MAE) - L1 Loss")
print("-" * 80)

mae_criterion = nn.L1Loss()  # L1Loss is MAE in PyTorch
mae_loss = mae_criterion(predicted_scores, actual_scores)

print(f"MAE Loss: {mae_loss.item():.4f}")

print("\nCHARACTERISTICS:")
print("✓ More robust to outliers than MSE")
print("✓ Less sensitive to large errors (doesn't square them)")
print("✗ Not differentiable at zero (can cause optimization issues)")
print(f"✓ Formula: (1/n) × Σ|predicted - actual|")

# Let's see the individual absolute errors
absolute_errors = torch.abs(predicted_scores - actual_scores)
print(f"\nAbsolute errors: {absolute_errors}")
print(f"The outlier contributes {absolute_errors[4].item():.0f}, but not as dramatically as MSE")

# ============================================================================
# SECTION 4: Comparing MSE vs MAE on the Outlier
# ============================================================================
print("\n" + "-" * 80)
print("COMPARISON: MSE vs MAE with Outlier")
print("-" * 80)

print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"MAE Loss: {mae_loss.item():.4f}")

print("\nWithout the outlier:")
# Calculate without the last data point (the outlier)
mse_no_outlier = mse_criterion(predicted_scores[:4], actual_scores[:4])
mae_no_outlier = mae_criterion(predicted_scores[:4], actual_scores[:4])

print(f"MSE Loss (no outlier): {mse_no_outlier.item():.4f}")
print(f"MAE Loss (no outlier): {mae_no_outlier.item():.4f}")

print("\nImpact of the outlier:")
print(f"MSE increased by: {((mse_loss - mse_no_outlier) / mse_no_outlier * 100).item():.1f}%")
print(f"MAE increased by: {((mae_loss - mae_no_outlier) / mae_no_outlier * 100).item():.1f}%")
print("\n→ MSE is MUCH more sensitive to outliers!")

# ============================================================================
# SECTION 5: Smooth L1 Loss (Huber Loss)
# ============================================================================
print("\n" + "-" * 80)
print("3. SMOOTH L1 LOSS (Huber Loss) - Best of Both Worlds")
print("-" * 80)

smooth_l1_criterion = nn.SmoothL1Loss()
smooth_l1_loss = smooth_l1_criterion(predicted_scores, actual_scores)

print(f"Smooth L1 Loss: {smooth_l1_loss.item():.4f}")

print("\nCHARACTERISTICS:")
print("✓ Combines benefits of MSE and MAE")
print("✓ Quadratic for small errors (like MSE)")
print("✓ Linear for large errors (like MAE)")
print("✓ More robust to outliers than MSE")
print("✓ Smoother gradients than MAE")

print("\nHOW IT WORKS:")
print("If |error| < 1: loss = 0.5 × error²  (MSE behavior)")
print("If |error| ≥ 1: loss = |error| - 0.5  (MAE behavior)")

# Show which regime each error falls into
for i, error in enumerate(errors):
    abs_error = abs(error.item())
    regime = "MSE regime" if abs_error < 1 else "MAE regime"
    print(f"Error {i+1}: {error.item():6.1f} → {regime}")

# ============================================================================
# SECTION 6: Choosing the Right Loss Function
# ============================================================================
print("\n" + "-" * 80)
print("DECISION GUIDE: Which Loss Should You Use?")
print("-" * 80)

print("""
📊 USE MEAN SQUARED ERROR (MSE) when:
   ✓ You have clean data with few outliers
   ✓ Large errors should be penalized heavily
   ✓ You want smooth gradients for optimization
   ✓ Example: Predicting house prices in a stable market

📏 USE MEAN ABSOLUTE ERROR (MAE) when:
   ✓ Your data has outliers
   ✓ All errors should be treated more equally
   ✓ You want the error in the same units as your data
   ✓ Example: Predicting delivery times (traffic outliers common)

🎯 USE SMOOTH L1 LOSS (Huber) when:
   ✓ You want robustness to outliers
   ✓ But still want smooth optimization
   ✓ Best for real-world data with occasional anomalies
   ✓ Example: Object detection bounding box regression
""")

# ============================================================================
# SECTION 7: Practical Example - Training Impact
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL IMPACT: How Loss Choice Affects Training")
print("-" * 80)

# Simulate gradient magnitude (how much the model will update)
# This is simplified, but shows the concept

print("When we have an outlier with error = 72:")
outlier_error = torch.tensor(72.0, requires_grad=True)

# MSE gradient
mse_loss_example = outlier_error ** 2 / 2  # Simplified
mse_loss_example.backward()
print(f"MSE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")

# MAE gradient
outlier_error.grad = None  # Reset gradient
mae_loss_example = torch.abs(outlier_error)
mae_loss_example.backward()
print(f"MAE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")

print("\n→ MSE produces a gradient 72x larger for this outlier!")
print("→ This means the model will update much more aggressively")
print("→ Outliers can dominate training with MSE")

# ============================================================================
# SECTION 8: Testing with Different Scenarios
# ============================================================================
print("\n" + "-" * 80)
print("EXPERIMENT: Different Data Scenarios")
print("-" * 80)

# Scenario 1: Clean data (no outliers)
clean_actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0])
clean_pred = torch.tensor([84.0, 89.0, 87.0, 91.0, 86.0])

# Scenario 2: Data with moderate errors
moderate_actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0])
moderate_pred = torch.tensor([80.0, 85.0, 83.0, 87.0, 82.0])

# Scenario 3: Data with outlier
outlier_actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0])
outlier_pred = torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])

scenarios = [
    ("Clean data (small errors)", clean_pred, clean_actual),
    ("Moderate errors", moderate_pred, moderate_actual),
    ("With outlier", outlier_pred, outlier_actual)
]

for name, pred, actual in scenarios:
    mse = F.mse_loss(pred, actual)
    mae = F.l1_loss(pred, actual)
    smooth_l1 = F.smooth_l1_loss(pred, actual)
    
    print(f"\n{name}:")
    print(f"  MSE:       {mse.item():6.2f}")
    print(f"  MAE:       {mae.item():6.2f}")
    print(f"  Smooth L1: {smooth_l1.item():6.2f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Different loss functions handle errors differently:
   • MSE: Squares errors → very sensitive to outliers
   • MAE: Absolute errors → robust to outliers
   • Smooth L1: Hybrid → best of both worlds

2. Loss choice impacts training:
   • MSE pushes model to fit outliers aggressively
   • MAE treats all errors more equally
   • Smooth L1 balances between the two

3. Choose based on your data:
   • Clean data → MSE
   • Noisy/outliers → MAE or Smooth L1
   • General purpose → Smooth L1

4. All three losses are differentiable and work with PyTorch autograd

NEXT STEPS:
→ Try with your own data
→ Experiment with different outlier magnitudes
→ Learn about classification losses (CrossEntropy, etc.)
""")
print("=" * 80)


if __name__ == "__main__":
    pass
