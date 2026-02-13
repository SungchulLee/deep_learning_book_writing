"""
==============================================================================
03_with_sklearn_data.py - Working with Real Datasets
================================================================================

LEARNING OBJECTIVES:
- Load and work with real-world datasets
- Understand data preprocessing (standardization)
- Handle train/validation/test splits properly
- Evaluate model with multiple metrics

PREREQUISITES:
- Completed 02_simple_binary_classification.py
- Understanding of mean and standard deviation
- Basic statistics knowledge

TIME TO COMPLETE: ~1 hour

DIFFICULTY: ⭐⭐☆☆☆ (Easy-Medium)
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("BREAST CANCER CLASSIFICATION - A REAL-WORLD EXAMPLE")
print("="*80)

# =============================================================================
# PART 1: LOADING THE DATASET
# =============================================================================
print("\n" + "="*80)
print("PART 1: LOADING AND EXPLORING THE DATASET")
print("="*80)

print("\n1.1: About the Wisconsin Breast Cancer Dataset")
print("-" * 40)
print("""
Dataset: Wisconsin Diagnostic Breast Cancer Dataset
Source: UCI Machine Learning Repository
Samples: 569 patients
Features: 30 numerical features computed from digitized images
Target: Malignant (1) or Benign (0)

Features include:
  - radius (mean of distances from center to points on perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter, area, smoothness, compactness, etc.
  
Goal: Predict whether a tumor is malignant or benign based on these features
""")

# Load the dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

print(f"\nDataset loaded successfully!")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names (first 5): {bc.feature_names[:5]}")
print(f"Target names: {bc.target_names}")  # ['malignant' 'benign']
print(f"\nClass distribution:")
print(f"  Malignant (0): {(y==0).sum()} ({100*(y==0).sum()/len(y):.1f}%)")
print(f"  Benign (1): {(y==1).sum()} ({100*(y==1).sum()/len(y):.1f}%)")

# =============================================================================
# PART 2: DATA PREPROCESSING
# =============================================================================
print("\n" + "="*80)
print("PART 2: DATA PREPROCESSING")
print("="*80)

print("\n2.1: Why Standardization?")
print("-" * 40)
print("""
Feature Standardization: Transform features to have mean=0, std=1

Why needed:
  1. Features have different scales (e.g., radius vs area)
  2. Gradient descent converges faster with standardized features
  3. Prevents features with large scales from dominating
  
Formula: z = (x - mean) / std

IMPORTANT: 
  - Fit scaler on TRAINING data only
  - Apply same transformation to test data
  - Never fit on test data (would leak information!)
""")

# Split data BEFORE standardization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train, transform train
X_test = scaler.transform(X_test)        # Only transform test (don't fit!)

print(f"\nAfter standardization:")
print(f"  Training mean: {X_train.mean():.6f} (should be ≈0)")
print(f"  Training std: {X_train.std():.6f} (should be ≈1)")

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"\nTensor shapes:")
print(f"  X_train: {X_train.shape}")  # (455, 30)
print(f"  y_train: {y_train.shape}")  # (455, 1)
print(f"  X_test: {X_test.shape}")    # (114, 30)
print(f"  y_test: {y_test.shape}")    # (114, 1)

# =============================================================================
# PART 3: BUILDING THE MODEL
# =============================================================================
print("\n" + "="*80)
print("PART 3: BUILDING THE MODEL")
print("="*80)

class LogisticRegressionModel(nn.Module):
    """
    Logistic Regression for Breast Cancer Classification
    
    Input: 30 features
    Output: 1 probability (benign)
    """
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

n_features = X_train.shape[1]  # 30 features
model = LogisticRegressionModel(n_features)

print(f"Model created with {n_features} input features")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# =============================================================================
# PART 4: TRAINING
# =============================================================================
print("\n" + "="*80)
print("PART 4: TRAINING THE MODEL")
print("="*80)

# Setup
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
num_epochs = 500

# Training history
history = {
    'loss': [],
    'accuracy': []
}

print(f"\nTraining for {num_epochs} epochs...")
print("-" * 40)

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    with torch.no_grad():
        predicted_classes = (y_pred >= 0.5).float()
        accuracy = (predicted_classes == y_train).float().mean()
    
    # Store history
    history['loss'].append(loss.item())
    history['accuracy'].append(accuracy.item())
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Loss: {loss.item():.4f} "
              f"Accuracy: {accuracy.item():.4f}")

print("\nTraining completed!")

# =============================================================================
# PART 5: EVALUATION WITH MULTIPLE METRICS
# =============================================================================
print("\n" + "="*80)
print("PART 5: COMPREHENSIVE EVALUATION")
print("="*80)

model.eval()
with torch.no_grad():
    # Predictions on test set
    y_pred_proba = model(X_test)
    y_pred_class = (y_pred_proba >= 0.5).float()
    
    # Convert to numpy for sklearn metrics
    y_test_np = y_test.numpy().flatten()
    y_pred_np = y_pred_class.numpy().flatten()
    y_pred_proba_np = y_pred_proba.numpy().flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_np, y_pred_np)
    precision = precision_score(y_test_np, y_pred_np)
    recall = recall_score(y_test_np, y_pred_np)
    f1 = f1_score(y_test_np, y_pred_np)
    conf_matrix = confusion_matrix(y_test_np, y_pred_np)

print("\n5.1: Classification Metrics")
print("-" * 40)
print(f"Accuracy:  {accuracy:.4f}  - Overall correctness")
print(f"Precision: {precision:.4f}  - Of predicted benign, how many were correct?")
print(f"Recall:    {recall:.4f}  - Of actual benign, how many did we find?")
print(f"F1-Score:  {f1:.4f}  - Harmonic mean of precision and recall")

print("\n5.2: Confusion Matrix")
print("-" * 40)
print("                Predicted")
print("              Malig  Benign")
print(f"Actual Malig    {conf_matrix[0,0]:3d}    {conf_matrix[0,1]:3d}")
print(f"       Benign   {conf_matrix[1,0]:3d}    {conf_matrix[1,1]:3d}")

# Calculate specific metrics from confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nTrue Negatives (TN):  {tn} - Correctly identified malignant")
print(f"False Positives (FP): {fp} - Incorrectly predicted benign (BAD!)")
print(f"False Negatives (FN): {fn} - Incorrectly predicted malignant")
print(f"True Positives (TP):  {tp} - Correctly identified benign")

# =============================================================================
# PART 6: VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("PART 6: CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(15, 10))

# Plot 1: Training Loss
plt.subplot(2, 3, 1)
plt.plot(history['loss'], 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss', fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 2: Training Accuracy
plt.subplot(2, 3, 2)
plt.plot(history['accuracy'], 'g-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy', fontweight='bold')
plt.ylim([0.5, 1.0])
plt.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
plt.subplot(2, 3, 3)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix', fontweight='bold')

# Plot 4: Metrics Comparison
plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
plt.ylim([0, 1])
plt.ylabel('Score')
plt.title('Performance Metrics', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

# Plot 5: Prediction Distribution
plt.subplot(2, 3, 5)
plt.hist(y_pred_proba_np[y_test_np==0], bins=30, alpha=0.6, label='Malignant', color='red')
plt.hist(y_pred_proba_np[y_test_np==1], bins=30, alpha=0.6, label='Benign', color='blue')
plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Prediction Distribution', fontweight='bold')
plt.legend()

# Plot 6: Summary
plt.subplot(2, 3, 6)
summary = f"""
MODEL SUMMARY
{'='*40}

Dataset: Breast Cancer Wisconsin
Samples: {len(X)} total
  - Training: {len(X_train)}
  - Test: {len(X_test)}

Features: {n_features}

Training:
  - Epochs: {num_epochs}
  - Final Loss: {history['loss'][-1]:.4f}
  - Final Train Acc: {history['accuracy'][-1]:.4f}

Test Performance:
  - Accuracy: {accuracy:.4f}
  - Precision: {precision:.4f}
  - Recall: {recall:.4f}
  - F1-Score: {f1:.4f}

Clinical Interpretation:
  - Missed cancers (FP): {fp}
  - False alarms (FN): {fn}
"""
plt.text(0.1, 0.5, summary, fontsize=9, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_logistic_regression_tutorial/01_basics/breast_cancer_results.png',
            dpi=150, bbox_inches='tight')
print("Visualization saved!")

# =============================================================================
# KEY TAKEAWAYS
# =============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. DATA PREPROCESSING
   - Always standardize features
   - Fit scaler on training data only
   - Apply same transformation to test data

2. EVALUATION METRICS
   - Accuracy: Good for balanced datasets
   - Precision: Important when false positives are costly
   - Recall: Important when false negatives are costly
   - F1-Score: Balance between precision and recall

3. MEDICAL APPLICATIONS
   - False positives: Unnecessary worry/procedures
   - False negatives: Missed diseases (very dangerous!)
   - Often prioritize high recall (catch all diseases)

4. BEST PRACTICES
   - Use multiple metrics, not just accuracy
   - Understand confusion matrix
   - Consider domain-specific costs of errors
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Try different test_size values (0.1, 0.3, 0.5)
   How does it affect performance?

2. MEDIUM: Adjust decision threshold from 0.5 to 0.3
   How does it affect precision vs recall?

3. MEDIUM: Try different learning rates
   Plot learning curves for lr=0.01, 0.1, 1.0

4. HARD: Implement weighted loss function
   Give more penalty to false negatives
   Hint: Use pos_weight parameter in BCELoss

5. HARD: Feature importance analysis
   Which features are most important?
   Look at model.linear.weight values
""")

print("\n" + "="*80)
print("NEXT: 04_bce_vs_bcewithlogits.py")
print("Learn about numerical stability and better loss functions!")
print("="*80)
