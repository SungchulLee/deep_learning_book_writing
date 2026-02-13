"""
================================================================================
BEGINNER 01: Introduction to Loss Functions in PyTorch
================================================================================

WHAT YOU'LL LEARN:
- What is a loss function and why we need it
- Three ways to compute loss in PyTorch
- Basic regression loss (Mean Squared Error)
- How to interpret loss values

PREREQUISITES:
- Basic Python knowledge
- Basic understanding of PyTorch tensors

TIME TO COMPLETE: ~10 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("INTRODUCTION TO LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: What is a Loss Function?
# ============================================================================
"""
A loss function (also called cost function or objective function) measures
how wrong our model's predictions are compared to the actual values.

Think of it like this:
- Your model makes a prediction: "I think the price is $100"
- The actual price is: $150
- The loss function says: "You're off by $50!"

The goal of training is to MINIMIZE this loss.
"""

# ============================================================================
# SECTION 2: Sample Data - Predicting House Prices
# ============================================================================
print("\n" + "-" * 80)
print("SAMPLE DATA: House Size vs Price")
print("-" * 80)

# Actual house prices (ground truth / target values)
# Let's say we have 5 houses with their actual prices
actual_prices = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])
print(f"Actual prices (in $1000s): {actual_prices}")

# Our model's predictions (these are wrong initially!)
# This is what our untrained model thinks the prices are
predicted_prices = torch.tensor([140.0, 210.0, 245.0, 310.0, 360.0])
print(f"Predicted prices (in $1000s): {predicted_prices}")

# Let's see the differences
differences = actual_prices - predicted_prices
print(f"Differences (actual - predicted): {differences}")
print("Negative = model predicted too high, Positive = model predicted too low")

# ============================================================================
# SECTION 3: Method 1 - Computing Loss Manually
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 1: Manual Loss Calculation")
print("-" * 80)
print("We'll compute Mean Squared Error (MSE) step by step")

# Step 1: Calculate the difference (error) for each house
errors = actual_prices - predicted_prices
print(f"\nStep 1 - Errors: {errors}")

# Step 2: Square each error (makes all errors positive and penalizes large errors more)
squared_errors = errors ** 2
print(f"Step 2 - Squared errors: {squared_errors}")

# Step 3: Take the mean (average) of all squared errors
mse_manual = torch.mean(squared_errors)
print(f"Step 3 - Mean Squared Error: {mse_manual.item():.4f}")

print("\nWHAT THIS MEANS:")
print(f"On average, our predictions are off by about ${torch.sqrt(mse_manual).item():.2f}k")
print("(Square root of MSE gives us the Root Mean Squared Error)")

# ============================================================================
# SECTION 4: Method 2 - Using PyTorch Functional API
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 2: Using torch.nn.functional")
print("-" * 80)

# PyTorch provides a built-in function to calculate MSE
# This is more convenient and optimized
mse_functional = F.mse_loss(predicted_prices, actual_prices)
print(f"MSE using F.mse_loss: {mse_functional.item():.4f}")

# Verify they're the same
print(f"\nManual MSE == Functional MSE? {torch.allclose(mse_manual, mse_functional)}")

# ============================================================================
# SECTION 5: Method 3 - Using PyTorch Loss Class (Most Common in Training)
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 3: Using nn.MSELoss Class")
print("-" * 80)

# Create a loss function object
# This is the most common way used in training loops
criterion = nn.MSELoss()

# Use it to calculate loss
mse_class = criterion(predicted_prices, actual_prices)
print(f"MSE using nn.MSELoss: {mse_class.item():.4f}")

print("\nWHY USE A CLASS?")
print("- You can configure it once (e.g., different reduction methods)")
print("- Cleaner code in training loops")
print("- Can easily swap different loss functions")

# ============================================================================
# SECTION 6: Understanding Different Reduction Methods
# ============================================================================
print("\n" + "-" * 80)
print("BONUS: Understanding 'reduction' Parameter")
print("-" * 80)

# reduction='mean': Average of all errors (default)
criterion_mean = nn.MSELoss(reduction='mean')
loss_mean = criterion_mean(predicted_prices, actual_prices)
print(f"Reduction='mean': {loss_mean.item():.4f}")

# reduction='sum': Sum of all errors
criterion_sum = nn.MSELoss(reduction='sum')
loss_sum = criterion_sum(predicted_prices, actual_prices)
print(f"Reduction='sum': {loss_sum.item():.4f}")

# reduction='none': Individual errors for each sample
criterion_none = nn.MSELoss(reduction='none')
loss_none = criterion_none(predicted_prices, actual_prices)
print(f"Reduction='none': {loss_none}")

print("\nNote: sum = mean × number_of_samples")
print(f"Verification: {loss_mean.item():.4f} × {len(actual_prices)} = {loss_sum.item():.4f}")

# ============================================================================
# SECTION 7: What Makes a Good vs Bad Loss?
# ============================================================================
print("\n" + "-" * 80)
print("INTERPRETING LOSS VALUES")
print("-" * 80)

# Perfect predictions (loss should be 0)
perfect_predictions = actual_prices.clone()
loss_perfect = criterion(perfect_predictions, actual_prices)
print(f"Perfect predictions → Loss: {loss_perfect.item():.4f}")

# Slightly better predictions
better_predictions = torch.tensor([148.0, 202.0, 249.0, 301.0, 351.0])
loss_better = criterion(better_predictions, actual_prices)
print(f"Better predictions → Loss: {loss_better.item():.4f}")

# Worse predictions
worse_predictions = torch.tensor([120.0, 230.0, 220.0, 330.0, 380.0])
loss_worse = criterion(worse_predictions, actual_prices)
print(f"Worse predictions → Loss: {loss_worse.item():.4f}")

print("\nKEY INSIGHT: Lower loss = Better predictions!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Loss functions measure how wrong your predictions are
2. Lower loss = better predictions (loss = 0 is perfect)
3. Three ways to compute loss in PyTorch:
   - Manual calculation (for learning/custom losses)
   - F.mse_loss() (functional API, quick and simple)
   - nn.MSELoss() (class API, best for training loops)
4. MSE is good for regression problems (predicting continuous values)
5. The 'reduction' parameter controls how errors are aggregated

NEXT STEPS:
→ Try changing the predicted_prices and see how loss changes
→ Learn about other loss functions (MAE, Huber, etc.)
→ Understand how optimizers use loss to improve the model
""")
print("=" * 80)
