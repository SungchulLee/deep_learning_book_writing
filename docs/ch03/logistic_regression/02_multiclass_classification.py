"""
==============================================================================
02_multiclass_classification.py - Softmax and Multi-Class Problems
================================================================================

LEARNING OBJECTIVES:
- Extend logistic regression to multiple classes
- Understand softmax activation
- Use CrossEntropyLoss
- Handle one-hot encoding
- Evaluate multi-class models

TIME TO COMPLETE: ~1.5 hours
DIFFICULTY: ⭐⭐⭐⭐☆ (Advanced)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("MULTI-CLASS CLASSIFICATION")
print("="*80)

# =============================================================================
# PART 1: BINARY VS MULTI-CLASS
# =============================================================================

print("\n" + "="*80)
print("PART 1: UNDERSTANDING MULTI-CLASS CLASSIFICATION")
print("="*80)

print("""
Binary Classification:
  2 classes: 0 or 1
  Output: Single probability
  Activation: Sigmoid
  Loss: Binary Cross-Entropy (BCE)

Multi-Class Classification:
  K classes: 0, 1, 2, ..., K-1
  Output: K probabilities (sum to 1)
  Activation: Softmax
  Loss: Cross-Entropy

Softmax Function:
  For K classes, softmax converts logits to probabilities:
  
  P(class=k) = exp(logit_k) / sum(exp(logit_j) for all j)
  
  Properties:
  ✓ All probabilities are positive
  ✓ Sum of probabilities = 1
  ✓ Differentiable
""")

# =============================================================================
# PART 2: GENERATE MULTI-CLASS DATA
# =============================================================================

print("\n" + "="*80)
print("PART 2: PREPARING MULTI-CLASS DATA")
print("="*80)

# Generate 3-class dataset
n_classes = 3
n_samples = 1500

torch.manual_seed(42)
np.random.seed(42)

X, y = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=n_classes,
    n_clusters_per_class=1,
    random_state=42
)

print(f"Dataset: {n_samples} samples, {n_classes} classes")
print(f"Class distribution:")
for i in range(n_classes):
    count = (y == i).sum()
    pct = 100 * count / len(y)
    print(f"  Class {i}: {count:4d} samples ({pct:.1f}%)")

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))

# IMPORTANT: For CrossEntropyLoss, targets should be class indices (Long)
# NOT one-hot encoded!
y_train = torch.LongTensor(y_train)  # Shape: (N,) with values 0, 1, 2
y_test = torch.LongTensor(y_test)

print(f"\nData shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape} (class indices)")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Create DataLoaders
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)

# =============================================================================
# PART 3: MULTI-CLASS MODEL
# =============================================================================

print("\n" + "="*80)
print("PART 3: MULTI-CLASS LOGISTIC REGRESSION MODEL")
print("="*80)

class MultiClassLogisticRegression(nn.Module):
    """
    Multi-class Logistic Regression (Softmax Regression)
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # Linear layer: (input_dim) -> (num_classes)
        # Each class gets its own linear combination of features
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            logits: Raw scores of shape (batch_size, num_classes)
                   NOTE: We return logits, not probabilities!
                   CrossEntropyLoss applies softmax internally
        """
        logits = self.linear(x)  # Shape: (batch_size, num_classes)
        return logits
    
    def predict_proba(self, x):
        """Get class probabilities (for inference)"""
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities


# Create model
input_dim = X_train.shape[1]
model = MultiClassLogisticRegression(input_dim, n_classes)

print(f"Model created:")
print(f"  Input dimension: {input_dim}")
print(f"  Number of classes: {n_classes}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"\nModel architecture:")
print(model)

# =============================================================================
# PART 4: LOSS FUNCTION AND TRAINING
# =============================================================================

print("\n" + "="*80)
print("PART 4: TRAINING MULTI-CLASS MODEL")
print("="*80)

print("""
CrossEntropyLoss:
  - Combines LogSoftmax and NLLLoss
  - Expects:
    * Predictions: (batch_size, num_classes) - raw logits
    * Targets: (batch_size,) - class indices (Long tensor)
  - Automatically applies softmax
  - More numerically stable than manual softmax + log
""")

# Setup
criterion = nn.CrossEntropyLoss()  # Combines softmax + log + NLL
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

print(f"\nTraining for {num_epochs} epochs...")
print("-" * 60)

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        logits = model(batch_X)  # (batch_size, num_classes)
        loss = criterion(logits, batch_y)  # Expects logits and class indices
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accuracy calculation
        _, predicted = torch.max(logits, 1)  # Get class with highest logit
        train_loss += loss.item() * len(batch_X)
        correct += (predicted == batch_y).sum().item()
        total += len(batch_X)
    
    avg_train_loss = train_loss / total
    train_acc = correct / total
    
    # Validation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            
            _, predicted = torch.max(logits, 1)
            test_loss += loss.item() * len(batch_X)
            correct += (predicted == batch_y).sum().item()
            total += len(batch_X)
    
    avg_test_loss = test_loss / total
    test_acc = correct / total
    
    # Store history
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(avg_test_loss)
    history['test_acc'].append(test_acc)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.4f}")

print("\n✓ Training completed!")

# =============================================================================
# PART 5: EVALUATION
# =============================================================================

print("\n" + "="*80)
print("PART 5: COMPREHENSIVE EVALUATION")
print("="*80)

# Get predictions
model.eval()
all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        logits = model(batch_X)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_probabilities = np.array(all_probabilities)

# Classification report
print("\nClassification Report:")
print(classification_report(all_targets, all_predictions,
                          target_names=[f'Class {i}' for i in range(n_classes)]))

# Confusion matrix
cm = confusion_matrix(all_targets, all_predictions)
print("\nConfusion Matrix:")
print(cm)

# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================

print("\n" + "="*80)
print("PART 6: VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Training curves
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy curves
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
ax2.plot(history['test_acc'], label='Test Acc', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curves', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=[f'C{i}' for i in range(n_classes)],
            yticklabels=[f'C{i}' for i in range(n_classes)])
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_title('Confusion Matrix', fontweight='bold')

# Plot 4: Per-class accuracy
ax4 = plt.subplot(2, 3, 4)
per_class_acc = []
for i in range(n_classes):
    mask = all_targets == i
    acc = (all_predictions[mask] == all_targets[mask]).mean()
    per_class_acc.append(acc)

bars = ax4.bar(range(n_classes), per_class_acc, color='steelblue', alpha=0.7)
ax4.set_xlabel('Class')
ax4.set_ylabel('Accuracy')
ax4.set_title('Per-Class Accuracy', fontweight='bold')
ax4.set_xticks(range(n_classes))
ax4.set_xticklabels([f'Class {i}' for i in range(n_classes)])
ax4.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, per_class_acc):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')

# Plot 5: Probability distributions
ax5 = plt.subplot(2, 3, 5)
for i in range(n_classes):
    mask = all_targets == i
    probs = all_probabilities[mask, i]  # Probability of correct class
    ax5.hist(probs, bins=30, alpha=0.5, label=f'Class {i}')

ax5.set_xlabel('Predicted Probability (for true class)')
ax5.set_ylabel('Count')
ax5.set_title('Confidence Distribution', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Model weights visualization
ax6 = plt.subplot(2, 3, 6)
weights = model.linear.weight.data.cpu().numpy()  # Shape: (n_classes, input_dim)
im = ax6.imshow(weights, aspect='auto', cmap='RdBu_r', center=0)
ax6.set_xlabel('Feature Index')
ax6.set_ylabel('Class')
ax6.set_title('Model Weights', fontweight='bold')
ax6.set_yticks(range(n_classes))
ax6.set_yticklabels([f'Class {i}' for i in range(n_classes)])
plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.show()

print("Visualizations created!")

# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. MULTI-CLASS VS BINARY
   Binary: 1 output → Sigmoid → BCE
   Multi-class: K outputs → Softmax → CrossEntropy

2. IMPORTANT DIFFERENCES
   ✓ Output layer: Linear(input_dim, num_classes)
   ✓ Return logits (not probabilities)
   ✓ Use CrossEntropyLoss
   ✓ Targets are class indices (Long)
   ✓ Predictions: argmax of logits

3. CROSSENTROPYLOSS
   ✓ Combines softmax + log + NLL
   ✓ More numerically stable
   ✓ Expects logits as input
   ✓ Expects class indices as targets

4. EVALUATION
   ✓ Per-class accuracy
   ✓ Confusion matrix
   ✓ Precision/Recall per class
   ✓ F1-score per class

5. WHEN TO USE
   ✓ More than 2 classes
   ✓ Mutually exclusive classes
   ✓ Single-label classification
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Try with different number of classes (5, 10)

2. MEDIUM: Implement top-k accuracy:
   - Check if true class is in top 2 or top 3 predictions

3. MEDIUM: Add class weights to handle imbalance:
   - Use class_weight parameter in CrossEntropyLoss

4. HARD: Implement one-vs-rest approach:
   - Train K binary classifiers
   - Compare with direct multi-class

5. HARD: Add label smoothing:
   - Soft targets instead of hard 0/1
   - Improves generalization
""")

print("\n" + "="*80)
print("NEXT: 03_regularization.py - Preventing overfitting")
print("="*80)
