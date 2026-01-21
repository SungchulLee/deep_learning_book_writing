# Confusion Matrix

A confusion matrix is a table showing the counts of correct and incorrect predictions, broken down by class. It's the foundation for computing most classification metrics.

---

## Basic Confusion Matrix

### 1. Binary Classification

```python
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
```

### 2. Matrix Layout

```
                 Predicted
                 Neg    Pos
        Neg   |  TN  |  FP  |
Actual        |------|------|
        Pos   |  FN  |  TP  |
```

```python
# Extract values
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
```

### 3. Visualization

```python
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Neg', 'Predicted Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

---

## Multiclass Confusion Matrix

### 1. Basic Usage

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Multiclass Confusion Matrix:")
print(cm)
```

### 2. Visualization with Labels

```python
class_names = ['Setosa', 'Versicolor', 'Virginica']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Multiclass Confusion Matrix')
plt.show()
```

### 3. Normalized Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Normalize by true labels (rows)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Count
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=class_names, 
    ax=axes[0], cmap='Blues'
)
axes[0].set_title('Counts')

# Normalize by true (recall per class)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=class_names,
    normalize='true', ax=axes[1], cmap='Blues', values_format='.2f'
)
axes[1].set_title('Normalized by True (Recall)')

# Normalize by pred (precision per class)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=class_names,
    normalize='pred', ax=axes[2], cmap='Blues', values_format='.2f'
)
axes[2].set_title('Normalized by Predicted (Precision)')

plt.tight_layout()
plt.show()
```

---

## Deriving Metrics from Confusion Matrix

### 1. Binary Metrics

```python
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# All metrics from confusion matrix
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = 2 * precision * recall / (precision + recall)

print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score:    {f1:.4f}")
```

### 2. Per-class Metrics (Multiclass)

```python
def per_class_metrics(cm):
    """Calculate precision, recall, F1 for each class"""
    n_classes = cm.shape[0]
    metrics = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # Column sum minus diagonal
        fn = cm[i, :].sum() - tp  # Row sum minus diagonal
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'class': i,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        })
    
    return metrics

# Calculate
cm = confusion_matrix(y_test, y_pred)
metrics = per_class_metrics(cm)

for m in metrics:
    print(f"Class {m['class']}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
```

---

## Understanding Errors

### 1. Error Analysis

```python
# Find misclassified samples
misclassified = y_test != y_pred
misclassified_indices = np.where(misclassified)[0]

print(f"Misclassified samples: {len(misclassified_indices)}")
print(f"Misclassification rate: {len(misclassified_indices)/len(y_test):.2%}")

# What were the errors?
for idx in misclassified_indices[:5]:  # Show first 5
    print(f"Sample {idx}: True={y_test[idx]}, Predicted={y_pred[idx]}")
```

### 2. Most Confused Pairs

```python
def most_confused_pairs(cm, class_names):
    """Find class pairs with most confusion"""
    n_classes = cm.shape[0]
    pairs = []
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                pairs.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j]
                })
    
    # Sort by count descending
    pairs.sort(key=lambda x: x['count'], reverse=True)
    return pairs

# Find most common errors
class_names = ['Setosa', 'Versicolor', 'Virginica']
confused = most_confused_pairs(cm, class_names)

print("Most confused pairs:")
for pair in confused[:3]:
    if pair['count'] > 0:
        print(f"  {pair['true']} → {pair['pred']}: {pair['count']} times")
```

---

## Multilabel Confusion Matrix

### 1. Basic Usage

```python
from sklearn.metrics import multilabel_confusion_matrix

# Multilabel predictions (each sample can have multiple labels)
y_true_ml = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
y_pred_ml = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1]])

mcm = multilabel_confusion_matrix(y_true_ml, y_pred_ml)

print(f"Shape: {mcm.shape}")  # (n_labels, 2, 2)

# Each label gets its own 2x2 confusion matrix
for i, label_cm in enumerate(mcm):
    print(f"\nLabel {i}:")
    print(label_cm)
```

---

## Threshold Effects

### 1. How Threshold Affects Confusion Matrix

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate binary data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Different thresholds
thresholds = [0.3, 0.5, 0.7]

fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 4))

for ax, thresh in zip(axes, thresholds):
    y_pred_t = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'], ax=ax)
    ax.set_title(f'Threshold = {thresh}')

plt.tight_layout()
plt.show()
```

### 2. Threshold Selection Trade-offs

```python
# Lower threshold: More positive predictions
#   → Higher recall (catch more positives)
#   → Lower precision (more false positives)

# Higher threshold: Fewer positive predictions
#   → Lower recall (miss some positives)
#   → Higher precision (fewer false positives)

# Example metrics at different thresholds
print(f"{'Threshold':<10} {'Precision':<12} {'Recall':<12} {'F1':<10}")
print("-" * 45)

for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred_t = (y_proba >= thresh).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f = f1_score(y_test, y_pred_t)
    
    print(f"{thresh:<10} {p:<12.4f} {r:<12.4f} {f:<10.4f}")
```

---

## Cost-sensitive Analysis

### 1. Weighted Confusion Matrix

```python
# Different costs for different errors
cost_matrix = np.array([
    [0, 1],   # Cost of FP (false alarm) = 1
    [10, 0]   # Cost of FN (missed detection) = 10
])

def total_cost(cm, cost_matrix):
    """Calculate total cost from confusion matrix"""
    return np.sum(cm * cost_matrix)

cm = confusion_matrix(y_test, y_pred)
cost = total_cost(cm, cost_matrix)
print(f"Total cost: {cost}")
```

### 2. Finding Cost-optimal Threshold

```python
def cost_optimal_threshold(y_true, y_proba, cost_fp=1, cost_fn=10):
    """Find threshold that minimizes total cost"""
    best_cost = float('inf')
    best_threshold = 0.5
    
    for thresh in np.linspace(0.01, 0.99, 99):
        y_pred_t = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        
        cost = fp * cost_fp + fn * cost_fn
        
        if cost < best_cost:
            best_cost = cost
            best_threshold = thresh
    
    return best_threshold, best_cost

best_t, best_c = cost_optimal_threshold(y_test, y_proba, cost_fp=1, cost_fn=10)
print(f"Best threshold: {best_t:.2f} (cost: {best_c})")
```

---

## Visualization Best Practices

### 1. Pretty Confusion Matrix

```python
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', 
                          normalize=False, figsize=(8, 6)):
    """Plot a nicely formatted confusion matrix"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    
    # Use different colormaps for emphasis
    mask = np.eye(len(cm), dtype=bool)
    
    # Plot
    ax = sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     square=True, linewidths=0.5)
    
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Rotate labels if many classes
    if len(class_names) > 5:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    return ax

# Example
class_names = ['Setosa', 'Versicolor', 'Virginica']
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names, normalize=True)
plt.show()
```

### 2. Side-by-side Comparison

```python
def compare_models_cm(cms, model_names, class_names):
    """Compare confusion matrices from multiple models"""
    n_models = len(cms)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, cm, name in zip(axes, cms, model_names):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, square=True)
        ax.set_title(name)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    return fig

# Compare two models
# cms = [cm_model1, cm_model2]
# compare_models_cm(cms, ['Model 1', 'Model 2'], class_names)
```

---

## Summary

**Key interpretations:**
- **Diagonal elements**: Correct predictions
- **Off-diagonal elements**: Errors (misclassifications)
- **Row i, Column j**: True class i predicted as class j

**Common patterns:**
- High diagonal, low off-diagonal = Good model
- Large off-diagonal = Confused classes (may need more features)
- Asymmetric errors = Bias toward certain predictions

**Usage tips:**
- Always visualize before computing metrics
- Normalize for imbalanced classes
- Check which class pairs are most confused
- Use cost matrices for business decisions
