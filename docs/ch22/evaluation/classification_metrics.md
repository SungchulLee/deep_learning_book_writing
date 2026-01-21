# Classification Metrics

Classification metrics evaluate how well models distinguish between classes. Different metrics prioritize different aspects of performance, so choose based on your specific use case.

---

## Accuracy

### 1. Definition and Usage

```python
from sklearn.metrics import accuracy_score
import numpy as np

y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # 0.75
```

### 2. Mathematical Definition

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
# Manual calculation
correct = np.sum(y_true == y_pred)
total = len(y_true)
accuracy_manual = correct / total
print(f"Manual Accuracy: {accuracy_manual:.4f}")
```

### 3. Limitations

```python
# Accuracy fails with imbalanced classes
y_true_imb = np.array([0]*950 + [1]*50)  # 95% class 0
y_pred_all_zero = np.zeros(1000)  # Predict all 0

# High accuracy but useless model!
acc = accuracy_score(y_true_imb, y_pred_all_zero)
print(f"Accuracy (all zeros): {acc:.2%}")  # 95%
```

---

## Precision

### 1. Definition and Usage

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")
```

### 2. Mathematical Definition

$$Precision = \frac{TP}{TP + FP}$$

**"Of all predicted positive, how many were actually positive?"**

```python
# Manual calculation
tp = np.sum((y_true == 1) & (y_pred == 1))
fp = np.sum((y_true == 0) & (y_pred == 1))
precision_manual = tp / (tp + fp)
print(f"Manual Precision: {precision_manual:.4f}")
```

### 3. When to Prioritize Precision

- **Spam detection**: Don't want false positives (real email marked as spam)
- **Legal search**: Returning irrelevant documents wastes time
- **Recommendation systems**: Bad recommendations hurt trust

---

## Recall (Sensitivity, TPR)

### 1. Definition and Usage

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")
```

### 2. Mathematical Definition

$$Recall = \frac{TP}{TP + FN}$$

**"Of all actual positives, how many did we catch?"**

```python
# Manual calculation
tp = np.sum((y_true == 1) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))
recall_manual = tp / (tp + fn)
print(f"Manual Recall: {recall_manual:.4f}")
```

### 3. When to Prioritize Recall

- **Disease screening**: Don't want to miss sick patients
- **Fraud detection**: Missing fraud is costly
- **Security systems**: Missing threats is dangerous

---

## F1 Score

### 1. Definition and Usage

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
```

### 2. Mathematical Definition

$$F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

**Harmonic mean of precision and recall**

```python
# Manual calculation
f1_manual = 2 * (precision * recall) / (precision + recall)
print(f"Manual F1: {f1_manual:.4f}")
```

### 3. F-beta Score

```python
from sklearn.metrics import fbeta_score

# β > 1: Favor recall
# β < 1: Favor precision
# β = 1: F1 score

f2 = fbeta_score(y_true, y_pred, beta=2)  # Recall-heavy
f05 = fbeta_score(y_true, y_pred, beta=0.5)  # Precision-heavy

print(f"F2 (recall focus): {f2:.4f}")
print(f"F0.5 (precision focus): {f05:.4f}")
```

---

## Specificity (TNR)

### 1. Definition

$$Specificity = \frac{TN}{TN + FP}$$

**"Of all actual negatives, how many did we correctly identify?"**

```python
# sklearn doesn't have specificity directly
def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)

specificity = specificity_score(y_true, y_pred)
print(f"Specificity: {specificity:.4f}")
```

---

## ROC-AUC

### 1. ROC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Need probability predictions
y_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.9, 0.3])

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. AUC Score

```python
auc = roc_auc_score(y_true, y_proba)
print(f"AUC: {auc:.4f}")
```

### 3. Interpretation

```python
# AUC = 1.0: Perfect classifier
# AUC = 0.5: Random classifier
# AUC < 0.5: Worse than random (flip predictions)

# AUC is threshold-independent
# Measures ranking ability (positive ranked higher than negative)
```

### 4. Multiclass ROC-AUC

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Multiclass data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                           n_informative=5, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_proba_multi = model.predict_proba(X_test)

# One-vs-Rest AUC
auc_ovr = roc_auc_score(y_test, y_proba_multi, multi_class='ovr')
# One-vs-One AUC
auc_ovo = roc_auc_score(y_test, y_proba_multi, multi_class='ovo')

print(f"AUC (OvR): {auc_ovr:.4f}")
print(f"AUC (OvO): {auc_ovo:.4f}")
```

---

## Precision-Recall Curve

### 1. PR Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, 'b-', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
```

### 2. Average Precision (AP)

```python
ap = average_precision_score(y_true, y_proba)
print(f"Average Precision: {ap:.4f}")
```

### 3. When to Use PR Curve vs ROC Curve

```python
# PR Curve: Better for imbalanced datasets
# - Doesn't include true negatives
# - More sensitive to changes with rare positive class

# ROC Curve: Good for balanced datasets
# - Includes both classes equally
# - Can be overly optimistic with imbalanced data
```

---

## Log Loss (Cross-Entropy)

### 1. Definition and Usage

```python
from sklearn.metrics import log_loss

logloss = log_loss(y_true, y_proba)
print(f"Log Loss: {logloss:.4f}")
```

### 2. Mathematical Definition

$$LogLoss = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

```python
# Manual calculation
eps = 1e-15  # Avoid log(0)
y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
logloss_manual = -np.mean(y_true * np.log(y_proba_clipped) + 
                          (1 - y_true) * np.log(1 - y_proba_clipped))
print(f"Manual Log Loss: {logloss_manual:.4f}")
```

### 3. Properties

- Penalizes confident wrong predictions heavily
- Proper scoring rule (optimized by true probabilities)
- Used as loss function in training

---

## Matthews Correlation Coefficient (MCC)

### 1. Definition and Usage

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
print(f"MCC: {mcc:.4f}")
```

### 2. Mathematical Definition

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 3. Properties

- Range: -1 to +1
- +1: Perfect prediction
- 0: Random prediction
- -1: Total disagreement
- **Balanced measure** that works well with imbalanced classes

---

## Cohen's Kappa

### 1. Definition and Usage

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")
```

### 2. Properties

- Measures agreement beyond chance
- Range: -1 to +1
- Accounts for random agreement

---

## Multiclass Averaging

### 1. Average Options

```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Multiclass data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                           n_informative=5, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_multi = model.predict(X_test)

# Different averaging strategies
for average in ['micro', 'macro', 'weighted']:
    p = precision_score(y_test, y_pred_multi, average=average)
    r = recall_score(y_test, y_pred_multi, average=average)
    f = f1_score(y_test, y_pred_multi, average=average)
    print(f"{average:8}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")
```

### 2. Averaging Explained

```python
# micro: Global TP, FP, FN across all classes
# - Good when each sample matters equally
# - Favors majority class

# macro: Average of per-class metrics (unweighted)
# - Treats all classes equally
# - Good for balanced evaluation

# weighted: Average weighted by support (class size)
# - Accounts for class imbalance
# - Compromise between micro and macro
```

---

## Classification Report

### 1. Comprehensive Report

```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred_multi, target_names=['Class 0', 'Class 1', 'Class 2'])
print(report)
```

### 2. As Dictionary

```python
report_dict = classification_report(y_test, y_pred_multi, output_dict=True)
print(f"Class 0 F1: {report_dict['Class 0']['f1-score']:.4f}")
```

---

## Threshold Selection

### 1. Finding Optimal Threshold

```python
from sklearn.metrics import precision_recall_curve, f1_score

# Find threshold that maximizes F1
precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

# Calculate F1 for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])  # Last element is artificial
best_threshold = thresholds[best_idx]

print(f"Best threshold: {best_threshold:.4f}")
print(f"Best F1: {f1_scores[best_idx]:.4f}")
```

### 2. Business-specific Threshold

```python
# Cost-based threshold selection
def find_cost_optimal_threshold(y_true, y_proba, fp_cost=1, fn_cost=1):
    """Find threshold minimizing total cost"""
    thresholds = np.linspace(0, 1, 100)
    costs = []
    
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        fp = np.sum((y_true == 0) & (y_pred_t == 1))
        fn = np.sum((y_true == 1) & (y_pred_t == 0))
        total_cost = fp * fp_cost + fn * fn_cost
        costs.append(total_cost)
    
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]

# Example: FN is 10x more costly than FP (don't miss positives)
best_t, min_cost = find_cost_optimal_threshold(y_true, y_proba, fp_cost=1, fn_cost=10)
print(f"Cost-optimal threshold: {best_t:.4f}")
```

---

## PyTorch Equivalents

```python
import torch
import torch.nn as nn

# Cross-Entropy Loss (combines softmax + NLL)
criterion = nn.CrossEntropyLoss()

# For binary classification
criterion_binary = nn.BCEWithLogitsLoss()

# Note: PyTorch losses expect:
# - CrossEntropyLoss: logits (raw scores), not probabilities
# - BCEWithLogitsLoss: logits for binary
# - BCELoss: probabilities (after sigmoid)
```

---

## Choosing the Right Metric

| Scenario | Primary Metric | Why |
|----------|---------------|-----|
| Balanced classes | Accuracy, F1 | Simple, interpretable |
| Imbalanced classes | F1, AUC-PR, MCC | Handle class imbalance |
| Cost-sensitive | Custom with costs | Business requirements |
| Ranking needed | AUC-ROC | Threshold-independent |
| Probability quality | Log Loss | Calibration matters |

**Guidelines:**
- **Start with classification_report** for full picture
- **Use F1** for imbalanced binary classification
- **Use AUC-ROC** when threshold can be tuned later
- **Use Log Loss** when probabilities matter
- **Use MCC** for balanced single metric with imbalanced data
