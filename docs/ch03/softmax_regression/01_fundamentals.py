"""
===============================================================================
LEVEL 1: Softmax Regression Fundamentals
===============================================================================
Difficulty: Beginner
Prerequisites: Basic Python, basic NumPy
Learning Goals:
  - Understand what softmax function does
  - Learn how cross-entropy loss works
  - Compare NumPy and PyTorch implementations
  - Understand the relationship between logits, probabilities, and loss

Time to complete: 20-30 minutes
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn

print("=" * 80)
print("LEVEL 1: SOFTMAX REGRESSION FUNDAMENTALS")
print("=" * 80)

# =============================================================================
# PART 1: Understanding Softmax Function
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Understanding Softmax")
print("=" * 80)

"""
What is Softmax?
----------------
Softmax converts a vector of real numbers (called logits) into a probability
distribution. Each output is between 0 and 1, and they all sum to 1.

Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

Why do we use it?
- Converts raw scores to interpretable probabilities
- Amplifies differences between scores (larger values get higher probabilities)
- Essential for multi-class classification
"""

def softmax_numpy(x):
    """
    Compute softmax values for a 1D array.
    
    Args:
        x (np.array): Input array of logits (raw scores)
    
    Returns:
        np.array: Probability distribution (sums to 1)
    
    Note: For numerical stability, we subtract the max value before exp.
          This doesn't change the result but prevents overflow.
    """
    # Subtract max for numerical stability (prevents overflow in exp)
    x_shifted = x - np.max(x)
    exp_values = np.exp(x_shifted)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


# Example 1: Basic softmax computation
print("\nExample 1: Converting logits to probabilities")
print("-" * 80)
logits = np.array([2.0, 1.0, 0.1])
print(f"Input logits:         {logits}")
print(f"  (These are raw, unnormalized scores from a model)")

probabilities = softmax_numpy(logits)
print(f"\nOutput probabilities: {probabilities}")
print(f"  (These are interpretable as class probabilities)")
print(f"Sum of probabilities: {np.sum(probabilities):.6f}")
print(f"  (Should always equal 1.0)")

# Example 2: Effect of changing logits
print("\n\nExample 2: How logits affect probabilities")
print("-" * 80)
logits_scenarios = [
    np.array([1.0, 1.0, 1.0]),    # All equal
    np.array([3.0, 1.0, 1.0]),    # One much larger
    np.array([10.0, 1.0, 1.0]),   # One extremely larger
]

for i, logits in enumerate(logits_scenarios, 1):
    probs = softmax_numpy(logits)
    print(f"Scenario {i}: logits = {logits}")
    print(f"            probs  = {probs}")
    print()

print("💡 Key Insight: Larger differences in logits lead to more confident predictions!")


# =============================================================================
# PART 2: Softmax in PyTorch
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Softmax in PyTorch")
print("=" * 80)

# Convert to PyTorch tensor
logits_torch = torch.tensor([2.0, 1.0, 0.1])
print(f"\nInput (PyTorch tensor): {logits_torch}")

# Apply softmax along dimension 0 (the only dimension for 1D tensor)
probs_torch = torch.softmax(logits_torch, dim=0)
print(f"Output probabilities:   {probs_torch}")

# For batched data (multiple samples at once)
print("\n\nBatched Example (3 samples, 3 classes each):")
print("-" * 80)
# Shape: (batch_size, num_classes) = (3, 3)
batch_logits = torch.tensor([
    [2.0, 1.0, 0.1],   # Sample 1
    [0.5, 2.5, 1.0],   # Sample 2
    [1.5, 1.5, 1.5],   # Sample 3
])
print("Logits (3 samples x 3 classes):")
print(batch_logits)

# Apply softmax along dim=1 (across classes for each sample)
batch_probs = torch.softmax(batch_logits, dim=1)
print("\nProbabilities (after softmax):")
print(batch_probs)
print(f"\nSum for each sample: {batch_probs.sum(dim=1)}")
print("  (Each row sums to 1.0)")


# =============================================================================
# PART 3: Understanding Cross-Entropy Loss
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Cross-Entropy Loss")
print("=" * 80)

"""
What is Cross-Entropy Loss?
----------------------------
Cross-entropy measures how different your predicted probability distribution
is from the true distribution. Lower loss = better predictions.

For a single sample with true class k:
  Loss = -log(p_k)
  
where p_k is the predicted probability for the true class.

Why negative log?
- If p_k = 1.0 (perfect prediction), loss = -log(1.0) = 0
- If p_k = 0.5 (uncertain), loss = -log(0.5) = 0.69
- If p_k = 0.1 (wrong), loss = -log(0.1) = 2.30
- If p_k → 0 (very wrong), loss → infinity
"""

def cross_entropy_numpy(true_class, predicted_probs):
    """
    Compute cross-entropy loss for a single sample.
    
    Args:
        true_class (int): Index of the true class (0, 1, 2, ...)
        predicted_probs (np.array): Predicted probabilities for each class
    
    Returns:
        float: Cross-entropy loss value
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    predicted_probs = np.clip(predicted_probs, eps, 1 - eps)
    
    # Loss is the negative log probability of the true class
    loss = -np.log(predicted_probs[true_class])
    return loss


print("\nExample: Comparing good vs bad predictions")
print("-" * 80)

# True class is 0 (first class)
true_class = 0

# Good prediction (high probability on correct class)
good_probs = np.array([0.8, 0.15, 0.05])
loss_good = cross_entropy_numpy(true_class, good_probs)

# Medium prediction (moderate probability on correct class)
medium_probs = np.array([0.5, 0.3, 0.2])
loss_medium = cross_entropy_numpy(true_class, medium_probs)

# Bad prediction (low probability on correct class)
bad_probs = np.array([0.1, 0.6, 0.3])
loss_bad = cross_entropy_numpy(true_class, bad_probs)

print(f"True class: {true_class}")
print(f"\nGood prediction:   probs = {good_probs}   → loss = {loss_good:.4f}")
print(f"Medium prediction: probs = {medium_probs} → loss = {loss_medium:.4f}")
print(f"Bad prediction:    probs = {bad_probs}   → loss = {loss_bad:.4f}")
print("\n💡 Key Insight: Lower loss = better prediction on the true class!")


# =============================================================================
# PART 4: PyTorch CrossEntropyLoss (The Right Way)
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: PyTorch CrossEntropyLoss")
print("=" * 80)

"""
CRITICAL CONCEPT: PyTorch's CrossEntropyLoss
--------------------------------------------
nn.CrossEntropyLoss combines:
  1. Softmax (converts logits to probabilities)
  2. Log (takes logarithm)
  3. Negative Log Likelihood (computes loss)

INPUT:
  - Predictions: raw logits (unnormalized scores), NOT probabilities
  - Targets: class indices (0, 1, 2, ...), NOT one-hot vectors

DO NOT apply softmax before CrossEntropyLoss - it does it internally!
"""

# Create the loss function
criterion = nn.CrossEntropyLoss()

print("\nExample 1: Single sample")
print("-" * 80)

# True class index
y_true = torch.tensor([0])  # Shape: (1,) - true class is 0

# Predictions (logits) - DO NOT apply softmax!
# Shape: (1, 3) - 1 sample, 3 classes
y_pred_good = torch.tensor([[3.0, 1.0, 0.5]])   # High score on class 0
y_pred_bad = torch.tensor([[0.5, 3.0, 2.0]])    # High score on class 1

loss_good = criterion(y_pred_good, y_true)
loss_bad = criterion(y_pred_bad, y_true)

print(f"True class: {y_true.item()}")
print(f"\nGood logits: {y_pred_good}")
print(f"  Loss: {loss_good.item():.4f}")
print(f"\nBad logits: {y_pred_bad}")
print(f"  Loss: {loss_bad.item():.4f}")

# To see the predicted class, use argmax
pred_class_good = torch.argmax(y_pred_good, dim=1)
pred_class_bad = torch.argmax(y_pred_bad, dim=1)
print(f"\nPredicted class (good): {pred_class_good.item()} ✓")
print(f"Predicted class (bad):  {pred_class_bad.item()} ✗")


print("\n\nExample 2: Batch of samples")
print("-" * 80)

# Batch of 4 samples, 3 classes each
y_true_batch = torch.tensor([2, 0, 1, 2])  # Shape: (4,)

# Logits for 4 samples
y_pred_batch = torch.tensor([
    [0.5, 1.0, 3.0],   # Sample 0: should predict class 2 ✓
    [2.5, 0.5, 0.3],   # Sample 1: should predict class 0 ✓
    [0.2, 2.8, 0.5],   # Sample 2: should predict class 1 ✓
    [1.5, 2.0, 0.8],   # Sample 3: predicts class 1, true is 2 ✗
])  # Shape: (4, 3)

loss_batch = criterion(y_pred_batch, y_true_batch)
print(f"Batch loss (average): {loss_batch.item():.4f}")

# Get predictions
pred_classes = torch.argmax(y_pred_batch, dim=1)
print(f"\nTrue classes:      {y_true_batch.numpy()}")
print(f"Predicted classes: {pred_classes.numpy()}")

# Calculate accuracy
accuracy = (pred_classes == y_true_batch).float().mean()
print(f"Accuracy: {accuracy.item():.2%}")


# =============================================================================
# PART 5: Complete Pipeline Visualization
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Complete Pipeline")
print("=" * 80)

print("""
The Complete Softmax Regression Pipeline:
------------------------------------------

1. Model Output (Logits)
   ↓
   [2.5, 1.0, 0.3]  ← Raw, unnormalized scores
   
2. Softmax (in CrossEntropyLoss)
   ↓
   [0.77, 0.17, 0.08]  ← Probabilities (sum to 1)
   
3. Cross-Entropy Loss
   ↓
   Compare with true class → Compute loss
   
4. Backpropagation
   ↓
   Update model weights to reduce loss

During TRAINING:
  - Use CrossEntropyLoss (it handles softmax internally)
  - Input: logits (raw scores)
  - Target: class indices

During INFERENCE (making predictions):
  - Get logits from model
  - Apply softmax to get probabilities (optional)
  - Use argmax to get predicted class
""")


# =============================================================================
# PART 6: Common Mistakes and Best Practices
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Common Mistakes and Best Practices")
print("=" * 80)

print("""
❌ MISTAKE 1: Applying softmax before CrossEntropyLoss
--------------------------------------------------
# WRONG:
probs = torch.softmax(logits, dim=1)
loss = criterion(probs, targets)  # Double softmax!

# RIGHT:
loss = criterion(logits, targets)  # CrossEntropyLoss applies softmax


❌ MISTAKE 2: Using one-hot encoded targets
--------------------------------------------------
# WRONG:
targets = torch.tensor([[1, 0, 0], [0, 1, 0]])  # One-hot encoded

# RIGHT:
targets = torch.tensor([0, 1])  # Class indices


❌ MISTAKE 3: Wrong tensor shapes
--------------------------------------------------
# For batch of 10 samples, 5 classes:
logits shape should be:  (10, 5)
targets shape should be: (10,)  NOT (10, 1) or (10, 5)


✅ BEST PRACTICES:
--------------------------------------------------
1. Return logits from your model (no softmax in forward())
2. Use CrossEntropyLoss for training
3. Apply softmax only at inference if you need probabilities
4. Use class indices for targets, not one-hot vectors
5. Check tensor shapes: logits (N, C), targets (N,)
""")


# =============================================================================
# PART 7: Practice Exercise
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: Quick Practice")
print("=" * 80)

print("""
Try to predict the outcome:
---------------------------
Given:
  - True class: 1
  - Logits: [1.0, 5.0, 2.0]

Questions:
1. Which class will the model predict? (Hint: argmax)
2. Will the loss be high or low? (Hint: is the prediction correct?)

Let's check:
""")

true_class_exercise = torch.tensor([1])
logits_exercise = torch.tensor([[1.0, 5.0, 2.0]])

predicted_class = torch.argmax(logits_exercise, dim=1)
loss_exercise = criterion(logits_exercise, true_class_exercise)

print(f"True class: {true_class_exercise.item()}")
print(f"Predicted class: {predicted_class.item()}")
print(f"Loss: {loss_exercise.item():.4f}")
print(f"Correct prediction? {predicted_class.item() == true_class_exercise.item()}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print("""
✅ Softmax converts logits into probabilities
✅ Cross-entropy measures prediction quality
✅ Lower loss = better predictions
✅ PyTorch's CrossEntropyLoss:
   - Takes logits as input (NOT probabilities)
   - Takes class indices as targets (NOT one-hot)
   - Combines softmax + log + NLL internally

Next Steps:
-----------
→ Level 2: Build a simple neural network for classification
→ Level 3: Train on real datasets (MNIST)
→ Level 4: Advanced techniques and optimizations

🎉 Congratulations! You've mastered the fundamentals!
""")
