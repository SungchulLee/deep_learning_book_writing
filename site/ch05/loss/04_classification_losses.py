"""
================================================================================
BEGINNER 04: Classification Loss Functions
================================================================================

WHAT YOU'LL LEARN:
- Difference between regression and classification
- Binary Cross-Entropy (BCE) for binary classification
- Cross-Entropy Loss for multi-class classification
- Understanding logits and probabilities
- One-hot encoding and class labels

PREREQUISITES:
- Complete previous beginner tutorials
- Understand basic classification concepts

TIME TO COMPLETE: ~20 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("CLASSIFICATION LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: Classification vs Regression
# ============================================================================
print("\n" + "-" * 80)
print("CLASSIFICATION VS REGRESSION")
print("-" * 80)

print("""
REGRESSION (Covered in previous tutorials):
  • Predict continuous values
  • Example: House price ($150,000), Temperature (25.3°C)
  • Loss: MSE, MAE, Huber
  
CLASSIFICATION:
  • Predict discrete categories/classes
  • Example: Email is Spam/Not Spam, Image is Cat/Dog/Bird
  • Loss: Cross-Entropy, Binary Cross-Entropy
  
Key difference: Classification outputs probabilities for each class!
""")

# ============================================================================
# SECTION 2: Binary Classification - Email Spam Detection
# ============================================================================
print("\n" + "-" * 80)
print("BINARY CLASSIFICATION EXAMPLE: Spam Detection")
print("-" * 80)

print("Let's classify 5 emails as Spam (1) or Not Spam (0):\n")

# True labels (0 = Not Spam, 1 = Spam)
true_labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32)
print(f"True labels: {true_labels}")
print("  0 = Not Spam, 1 = Spam\n")

# Model predictions (probabilities between 0 and 1)
# After sigmoid activation
predicted_probs = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.6])
print(f"Predicted probabilities: {predicted_probs}")
print("  Higher = More likely to be spam\n")

# Interpret predictions
print("Interpretation:")
for i, (true_label, pred_prob) in enumerate(zip(true_labels, predicted_probs)):
    true_class = "Spam" if true_label == 1 else "Not Spam"
    pred_class = "Spam" if pred_prob > 0.5 else "Not Spam"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
    correct = "✓" if true_class == pred_class else "✗"
    
    print(f"  Email {i+1}: True={true_class:8s}, Predicted={pred_class:8s} "
          f"({confidence*100:.0f}% confident) {correct}")

# ============================================================================
# SECTION 3: Binary Cross-Entropy Loss (BCE)
# ============================================================================
print("\n" + "-" * 80)
print("BINARY CROSS-ENTROPY LOSS (BCE)")
print("-" * 80)

# Calculate BCE loss
bce_criterion = nn.BCELoss()
bce_loss = bce_criterion(predicted_probs, true_labels)

print(f"BCE Loss: {bce_loss.item():.4f}\n")

print("WHAT IS BCE?")
print("  Formula: -[y × log(p) + (1-y) × log(1-p)]")
print("  where y = true label (0 or 1), p = predicted probability")

print("\nWHY THIS FORMULA?")
print("  • When true label = 1 (Spam):")
print("    Loss = -log(p) → Low if p is high (correct!)")
print("  • When true label = 0 (Not Spam):")
print("    Loss = -log(1-p) → Low if p is low (correct!)")

# Calculate loss for each sample
print("\nPer-sample losses:")
for i in range(len(true_labels)):
    y = true_labels[i].item()
    p = predicted_probs[i].item()
    
    # Manual BCE calculation
    if y == 1:
        sample_loss = -torch.log(torch.tensor(p))
    else:
        sample_loss = -torch.log(torch.tensor(1 - p))
    
    print(f"  Email {i+1}: True={int(y)}, Pred={p:.2f} → Loss={sample_loss.item():.4f}")

# ============================================================================
# SECTION 4: Understanding Logits
# ============================================================================
print("\n" + "-" * 80)
print("UNDERSTANDING LOGITS (RAW OUTPUTS)")
print("-" * 80)

print("""
In practice, neural networks output "logits" (raw, unbounded values).
We convert logits to probabilities using the Sigmoid function.

Logit (raw) → Sigmoid → Probability (0 to 1)
""")

# Example logits (could be any value)
logits = torch.tensor([-2.0, 3.0, -1.5, 2.5, 0.5])
print(f"Raw logits: {logits}\n")

# Convert to probabilities using sigmoid
probabilities = torch.sigmoid(logits)
print(f"After sigmoid: {probabilities}")

print("\nSigmoid function properties:")
print("  • logit = 0 → probability = 0.5 (uncertain)")
print("  • logit > 0 → probability > 0.5 (likely class 1)")
print("  • logit < 0 → probability < 0.5 (likely class 0)")
print("  • More extreme logits = more confident predictions")

# ============================================================================
# SECTION 5: BCEWithLogitsLoss (More Stable!)
# ============================================================================
print("\n" + "-" * 80)
print("BCEWithLogitsLoss - RECOMMENDED FOR TRAINING")
print("-" * 80)

print("""
Instead of: Model → Sigmoid → BCE Loss
Use:        Model → BCEWithLogitsLoss (combines both!)

Benefits:
  ✓ More numerically stable
  ✓ Faster computation
  ✓ Prevents gradient problems
""")

# Using BCEWithLogitsLoss
bce_with_logits = nn.BCEWithLogitsLoss()
loss_from_logits = bce_with_logits(logits, true_labels)

print(f"Loss using BCEWithLogitsLoss: {loss_from_logits.item():.4f}")

# Compare with manual approach
manual_probs = torch.sigmoid(logits)
manual_loss = bce_criterion(manual_probs, true_labels)
print(f"Loss using BCE(sigmoid(logits)): {manual_loss.item():.4f}")
print("→ Same result! But BCEWithLogitsLoss is more stable\n")

# ============================================================================
# SECTION 6: Multi-Class Classification - Image Classification
# ============================================================================
print("\n" + "-" * 80)
print("MULTI-CLASS CLASSIFICATION: Image Classification")
print("-" * 80)

print("Classifying 4 images into 3 categories: Cat, Dog, Bird\n")

# True labels (class indices)
true_classes = torch.tensor([0, 2, 1, 0])  # 0=Cat, 1=Dog, 2=Bird
print(f"True classes: {true_classes}")
print("  Image 1: Cat (0)")
print("  Image 2: Bird (2)")
print("  Image 3: Dog (1)")
print("  Image 4: Cat (0)\n")

# Model outputs (logits for each class)
# Shape: (batch_size, num_classes) = (4, 3)
logits_multi = torch.tensor([
    [3.0, 1.0, 0.5],   # Image 1: High confidence for Cat
    [0.5, 0.8, 2.5],   # Image 2: High confidence for Bird
    [1.0, 2.0, 0.5],   # Image 3: High confidence for Dog
    [2.5, 1.5, 1.0],   # Image 4: High confidence for Cat
])

print(f"Model logits (raw outputs):")
print(logits_multi)
print(f"Shape: {logits_multi.shape} (4 images, 3 classes)\n")

# Convert logits to probabilities using softmax
probs_multi = F.softmax(logits_multi, dim=1)
print(f"Probabilities after softmax:")
print(probs_multi)
print("\nNote: Each row sums to 1.0 (100% probability distributed across classes)")

# Show predictions
class_names = ['Cat', 'Dog', 'Bird']
print("\nPredictions:")
for i in range(len(true_classes)):
    predicted_class = torch.argmax(probs_multi[i]).item()
    confidence = probs_multi[i, predicted_class].item()
    true_class_name = class_names[true_classes[i]]
    pred_class_name = class_names[predicted_class]
    correct = "✓" if predicted_class == true_classes[i] else "✗"
    
    print(f"  Image {i+1}: True={true_class_name:4s}, Predicted={pred_class_name:4s} "
          f"({confidence*100:.1f}% confident) {correct}")

# ============================================================================
# SECTION 7: Cross-Entropy Loss for Multi-Class
# ============================================================================
print("\n" + "-" * 80)
print("CROSS-ENTROPY LOSS (Multi-Class)")
print("-" * 80)

# IMPORTANT: CrossEntropyLoss expects raw logits, NOT probabilities!
ce_criterion = nn.CrossEntropyLoss()
ce_loss = ce_criterion(logits_multi, true_classes)

print(f"Cross-Entropy Loss: {ce_loss.item():.4f}\n")

print("KEY POINTS:")
print("  1. CrossEntropyLoss takes RAW LOGITS (not probabilities!)")
print("  2. It applies softmax internally (more stable)")
print("  3. True labels are class indices (not one-hot encoded)")
print("  4. Formula: -log(probability of true class)")

# Show per-sample losses
print("\nPer-sample losses:")
for i in range(len(true_classes)):
    true_class = true_classes[i].item()
    true_prob = probs_multi[i, true_class].item()
    sample_loss = -torch.log(torch.tensor(true_prob))
    
    print(f"  Image {i+1}: True class={class_names[true_class]}, "
          f"Probability={true_prob:.4f}, Loss={sample_loss:.4f}")

print("\n→ Lower probability for true class = Higher loss")

# ============================================================================
# SECTION 8: One-Hot Encoding vs Class Indices
# ============================================================================
print("\n" + "-" * 80)
print("ONE-HOT ENCODING VS CLASS INDICES")
print("-" * 80)

print("PyTorch CrossEntropyLoss uses CLASS INDICES (simpler!)")
print(f"Class indices: {true_classes}\n")

print("But you might see ONE-HOT ENCODING in other frameworks:")
one_hot = F.one_hot(true_classes, num_classes=3)
print("One-hot encoded:")
print(one_hot)
print("\nEach row has a 1 in the position of the true class, 0s elsewhere")

# If you have one-hot encoded labels, convert them:
classes_from_one_hot = torch.argmax(one_hot, dim=1)
print(f"\nConverting back: {classes_from_one_hot}")
print(f"Same as original: {torch.equal(classes_from_one_hot, true_classes)}")

# ============================================================================
# SECTION 9: Comparing Binary vs Multi-Class
# ============================================================================
print("\n" + "-" * 80)
print("COMPARISON: Binary vs Multi-Class Classification")
print("-" * 80)

print("""
╔═══════════════════╦════════════════════╦═════════════════════════╗
║                   ║ BINARY             ║ MULTI-CLASS             ║
╠═══════════════════╬════════════════════╬═════════════════════════╣
║ Classes           ║ 2 (e.g., Yes/No)   ║ 3+ (e.g., Cat/Dog/Bird) ║
║ Model Output      ║ 1 logit            ║ N logits (N = classes)  ║
║ Activation        ║ Sigmoid            ║ Softmax                 ║
║ Output Range      ║ [0, 1]             ║ [0, 1] (sum to 1)       ║
║ Loss Function     ║ BCEWithLogitsLoss  ║ CrossEntropyLoss        ║
║ True Label Format ║ 0 or 1             ║ Class index (0 to N-1)  ║
╚═══════════════════╩════════════════════╩═════════════════════════╝
""")

# ============================================================================
# SECTION 10: Practical Tips
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL TIPS")
print("-" * 80)

print("""
✓ DO:
  • Use BCEWithLogitsLoss for binary classification
  • Use CrossEntropyLoss for multi-class classification
  • Let loss functions handle activation (sigmoid/softmax) internally
  • Use class indices for labels in CrossEntropyLoss
  • Monitor loss during training to check convergence

✗ DON'T:
  • Apply sigmoid before BCEWithLogitsLoss (it does it internally!)
  • Apply softmax before CrossEntropyLoss (it does it internally!)
  • Use BCELoss with raw logits (use BCEWithLogitsLoss instead)
  • One-hot encode labels for CrossEntropyLoss (use class indices)

COMMON MODEL ARCHITECTURES:
  Binary:     [...layers...] → Linear(in_features, 1) → BCEWithLogitsLoss
  Multi-Class: [...layers...] → Linear(in_features, num_classes) → CrossEntropyLoss
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Classification predicts discrete categories, not continuous values

2. Binary Classification (2 classes):
   • Use BCEWithLogitsLoss
   • Model outputs 1 value (logit)
   • Sigmoid converts logit → probability
   • Labels are 0 or 1

3. Multi-Class Classification (3+ classes):
   • Use CrossEntropyLoss
   • Model outputs N values (logits), one per class
   • Softmax converts logits → probability distribution
   • Labels are class indices (0, 1, 2, ...)

4. Both loss functions handle activation internally
   • Don't manually apply sigmoid/softmax before loss!
   • More numerically stable this way

5. For inference (making predictions):
   • Binary: threshold at 0.5 after sigmoid
   • Multi-class: take argmax after softmax

NEXT STEPS:
→ Build a simple image classifier
→ Experiment with different numbers of classes
→ Learn about class imbalance and weighted losses
""")
print("=" * 80)
