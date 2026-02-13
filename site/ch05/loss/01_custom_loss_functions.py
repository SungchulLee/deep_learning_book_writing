"""
================================================================================
ADVANCED 01: Creating Custom Loss Functions
================================================================================

WHAT YOU'LL LEARN:
- When and why to create custom losses
- How to implement custom loss functions
- Combining multiple loss terms
- Weighted losses for imbalanced data
- Focal loss, Dice loss, and other advanced losses

PREREQUISITES:
- Complete beginner and intermediate tutorials
- Strong understanding of PyTorch autograd

TIME TO COMPLETE: ~30 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("CREATING CUSTOM LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: Why Create Custom Loss Functions?
# ============================================================================
print("\n" + "-" * 80)
print("WHY CREATE CUSTOM LOSS FUNCTIONS?")
print("-" * 80)

print("""
Standard losses (MSE, CrossEntropy) don't always match your goals:

1. DOMAIN-SPECIFIC OBJECTIVES:
   • Medical imaging: Dice loss for segmentation overlap
   • Object detection: IoU loss for bounding boxes
   • GANs: Adversarial losses

2. HANDLING DATA IMBALANCE:
   • Focal loss for hard examples
   • Weighted losses for rare classes

3. MULTI-TASK LEARNING:
   • Combine multiple losses
   • Balance different objectives

4. CUSTOM CONSTRAINTS:
   • Physics-informed losses
   • Enforce specific properties

5. RESEARCH & EXPERIMENTATION:
   • Test new ideas
   • Improve upon existing methods
""")

# ============================================================================
# SECTION 2: Basic Custom Loss - Function Approach
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 1: Custom Loss as a Function")
print("-" * 80)

def custom_mse_loss(predictions, targets):
    """
    Custom implementation of Mean Squared Error
    This is for demonstration - use nn.MSELoss() in practice!
    """
    # Calculate squared differences
    squared_diff = (predictions - targets) ** 2
    
    # Take the mean
    loss = torch.mean(squared_diff)
    
    return loss

# Test it
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])

loss = custom_mse_loss(pred, target)
print(f"Custom MSE Loss: {loss.item():.4f}")

# Compare with PyTorch's MSE
pytorch_loss = F.mse_loss(pred, target)
print(f"PyTorch MSE Loss: {pytorch_loss.item():.4f}")
print(f"Match: {torch.allclose(loss, pytorch_loss)}\n")

print("KEY POINTS:")
print("  ✓ Use torch operations (not numpy) for autograd")
print("  ✓ Make sure output is a scalar (for backpropagation)")
print("  ✓ All operations must be differentiable")

# ============================================================================
# SECTION 3: Custom Loss - Class Approach (Recommended)
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 2: Custom Loss as a Class (Recommended)")
print("-" * 80)

class WeightedMSELoss(nn.Module):
    """
    MSE Loss with per-sample weights
    Useful when some samples are more important than others
    """
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: Model predictions
            targets: Ground truth values
            weights: Optional per-sample weights (default: all 1s)
        """
        # Calculate squared error
        squared_error = (predictions - targets) ** 2
        
        # Apply weights if provided
        if weights is not None:
            squared_error = squared_error * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(squared_error)
        elif self.reduction == 'sum':
            return torch.sum(squared_error)
        else:  # 'none'
            return squared_error

# Test it
criterion = WeightedMSELoss()

# Example: Later samples are more important
weights = torch.tensor([0.5, 1.0, 2.0])  # Increase importance
weighted_loss = criterion(pred, target, weights)

print(f"Weighted MSE Loss: {weighted_loss.item():.4f}")
print(f"Unweighted MSE Loss: {loss.item():.4f}")
print(f"\nThe weighted loss is higher because we emphasized the later samples")

# ============================================================================
# SECTION 4: Focal Loss - For Imbalanced Classification
# ============================================================================
print("\n" + "-" * 80)
print("FOCAL LOSS: Handling Class Imbalance")
print("-" * 80)

print("""
PROBLEM: Imbalanced datasets (e.g., 95% negative, 5% positive)
  • Model can achieve 95% accuracy by always predicting negative
  • Hard-to-classify examples get ignored
  
SOLUTION: Focal Loss
  • Down-weights easy examples
  • Focuses on hard examples
  • Formula: FL = -α(1-p)^γ log(p)
    where γ controls focusing (typical: 2)
""")

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    
    Paper: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth (0 or 1)
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Calculate focal weight
        # For positive class: (1-p)^γ
        # For negative class: p^γ
        focal_weight = torch.where(
            targets == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma
        )
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return torch.mean(focal_loss)

# Demonstrate focal loss
print("\nExample: Imbalanced binary classification")
print("Dataset: 90% class 0, 10% class 1\n")

# Create focal loss
focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
bce_criterion = nn.BCEWithLogitsLoss()

# Sample predictions and targets
logits = torch.tensor([2.0, -1.5, -2.0, 3.0, -1.0])  # Raw model outputs
targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])     # True labels

# Convert logits to probabilities for interpretation
probs = torch.sigmoid(logits)

print("Sample predictions and difficulty:")
for i, (logit, prob, target) in enumerate(zip(logits, probs, targets)):
    correct = (prob > 0.5 and target == 1) or (prob < 0.5 and target == 0)
    confidence = prob if target == 1 else 1 - prob
    difficulty = "EASY" if confidence > 0.8 else "HARD"
    status = "✓" if correct else "✗"
    
    print(f"  Sample {i+1}: Target={int(target)}, Prob={prob:.3f}, "
          f"{difficulty} {status}")

# Calculate losses
focal_loss = focal_criterion(logits, targets)
bce_loss = bce_criterion(logits, targets)

print(f"\nStandard BCE Loss: {bce_loss.item():.4f}")
print(f"Focal Loss: {focal_loss.item():.4f}")
print("\nFocal loss emphasizes the hard examples more!")

# ============================================================================
# SECTION 5: Dice Loss - For Segmentation
# ============================================================================
print("\n" + "-" * 80)
print("DICE LOSS: For Segmentation Tasks")
print("-" * 80)

print("""
DICE COEFFICIENT: Measures overlap between predictions and ground truth
  • Used in medical image segmentation
  • Range: 0 (no overlap) to 1 (perfect overlap)
  • Formula: Dice = 2|A ∩ B| / (|A| + |B|)
  
DICE LOSS = 1 - Dice Coefficient
""")

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Measures overlap between prediction and target
    """
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions (after sigmoid, values 0-1)
            targets: Ground truth binary mask (0 or 1)
        """
        # Flatten the tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        return 1 - dice

# Demonstrate dice loss
print("\nExample: Binary segmentation")

# Create a simple 5x5 "image"
true_mask = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

# Good prediction (close to truth)
good_pred = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 0.9, 0.8, 0.9, 0],
    [0, 0.85, 0.95, 0.85, 0],
    [0, 0.9, 0.8, 0.9, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

# Bad prediction (poor overlap)
bad_pred = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0.8, 0.9, 0.85, 0, 0],
    [0.9, 0.8, 0.9, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

dice_criterion = DiceLoss()

good_loss = dice_criterion(good_pred, true_mask)
bad_loss = dice_criterion(bad_pred, true_mask)

print(f"Good prediction Dice Loss: {good_loss.item():.4f}")
print(f"Bad prediction Dice Loss: {bad_loss.item():.4f}")
print("\nLower loss = better overlap!")

# ============================================================================
# SECTION 6: Combining Multiple Losses
# ============================================================================
print("\n" + "-" * 80)
print("COMBINING MULTIPLE LOSS TERMS")
print("-" * 80)

print("""
Often you want to optimize multiple objectives simultaneously:
  • Reconstruction + Regularization
  • Task loss + Consistency loss
  • Multiple task losses (multi-task learning)
  
APPROACH: Weighted sum of losses
  Total Loss = α₁ × Loss₁ + α₂ × Loss₂ + ...
""")

class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with learnable or fixed weights
    """
    def __init__(self, loss_weights=None):
        """
        Args:
            loss_weights: Dict of loss names to weights
                         If None, uses equal weights
        """
        super(CombinedLoss, self).__init__()
        self.loss_weights = loss_weights or {}
        
        # Define individual losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions, targets):
        """Calculate weighted combination of losses"""
        # Get weights (default to 1.0)
        w_mse = self.loss_weights.get('mse', 1.0)
        w_l1 = self.loss_weights.get('l1', 1.0)
        
        # Calculate individual losses
        mse = self.mse_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)
        
        # Combine
        total_loss = w_mse * mse + w_l1 * l1
        
        # Return total and components (useful for logging)
        return total_loss, {'mse': mse.item(), 'l1': l1.item()}

# Test combined loss
combined_criterion = CombinedLoss(loss_weights={'mse': 0.7, 'l1': 0.3})

pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])

total_loss, components = combined_criterion(pred, target)

print("\nCombined Loss Example:")
print(f"  MSE component: {components['mse']:.4f} (weight: 0.7)")
print(f"  L1 component: {components['l1']:.4f} (weight: 0.3)")
print(f"  Total loss: {total_loss.item():.4f}")

print("\nWHY COMBINE LOSSES?")
print("  • MSE: Smooth gradients, penalizes large errors")
print("  • L1: Robust to outliers")
print("  • Combination: Balance both properties!")

# ============================================================================
# SECTION 7: Best Practices for Custom Losses
# ============================================================================
print("\n" + "-" * 80)
print("BEST PRACTICES FOR CUSTOM LOSSES")
print("-" * 80)

print("""
✓ DO:
  1. Use nn.Module base class for losses
  2. Keep all operations in PyTorch (not numpy)
  3. Test gradient flow with small examples
  4. Add numerical stability (smooth terms, clamps)
  5. Document your loss function well
  6. Provide default hyperparameters
  7. Return scalar for backpropagation
  8. Consider numerical stability (avoid log(0), div by 0)

✗ DON'T:
  1. Use .item() inside loss (breaks gradients)
  2. Use in-place operations carelessly
  3. Forget to handle edge cases
  4. Make loss dependent on non-tensor constants
  5. Use operations that don't have gradients

TESTING YOUR LOSS:
  1. Check it returns correct shape (scalar)
  2. Verify gradients flow: loss.backward()
  3. Test with known inputs/outputs
  4. Compare with reference implementation if available
  5. Check numerical stability with edge cases
""")

# Example: Testing a custom loss
def test_custom_loss():
    """Template for testing custom losses"""
    print("\nTesting Custom Loss:")
    
    # 1. Create loss
    loss_fn = WeightedMSELoss()
    
    # 2. Create test inputs with gradients
    pred = torch.randn(10, requires_grad=True)
    target = torch.randn(10)
    
    # 3. Compute loss
    loss = loss_fn(pred, target)
    
    # 4. Check shape
    assert loss.dim() == 0, "Loss should be scalar!"
    print(f"  ✓ Shape check passed: {loss.shape}")
    
    # 5. Test backward pass
    loss.backward()
    assert pred.grad is not None, "Gradients should flow!"
    print(f"  ✓ Gradient check passed")
    
    # 6. Test numerical stability
    edge_pred = torch.tensor([0.0, 1e-10, 1e10])
    edge_target = torch.tensor([0.0, 0.0, 1e10])
    edge_loss = loss_fn(edge_pred, edge_target)
    assert not torch.isnan(edge_loss), "Loss should handle edge cases!"
    print(f"  ✓ Numerical stability check passed")
    
    print("  All tests passed! ✓\n")

test_custom_loss()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Create custom losses when standard ones don't fit:
   • Domain-specific objectives
   • Handle data imbalance
   • Multi-task learning
   • Research experiments

2. Two approaches:
   • Function: Simple, for quick experiments
   • Class (nn.Module): Professional, configurable, recommended

3. Advanced loss examples:
   • Focal Loss: For imbalanced classification
   • Dice Loss: For segmentation overlap
   • Combined Losses: Multiple objectives

4. Implementation tips:
   • Use PyTorch operations only
   • Return scalar for backpropagation
   • Add numerical stability
   • Test thoroughly

5. Common patterns:
   • Weighted losses for importance
   • Combining multiple terms
   • Class-balanced weights
   • Hard example mining

NEXT STEPS:
→ Implement focal loss for your imbalanced dataset
→ Experiment with loss combinations
→ Create domain-specific losses for your problem
→ Study papers for new loss functions
""")
print("=" * 80)
