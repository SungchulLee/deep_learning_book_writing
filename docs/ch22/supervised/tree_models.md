# Tree-Based Models

Decision trees recursively partition the feature space into regions that best separate the target variable. They naturally handle non-linear relationships, mixed feature types, and provide interpretable decision rules.

---

## Decision Tree Classifier

### 1. Basic Usage

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Tree Visualization

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=[f'Feature_{i}' for i in range(X.shape[1])],
          class_names=['Class 0', 'Class 1'], rounded=True)
plt.title('Decision Tree Structure')
plt.tight_layout()
plt.show()
```

### 3. Text Representation

```python
from sklearn.tree import export_text

# Export as text rules
tree_rules = export_text(model, feature_names=[f'Feature_{i}' for i in range(X.shape[1])])
print(tree_rules)
```

### 4. Split Criteria

```python
# Gini impurity (default)
model_gini = DecisionTreeClassifier(criterion='gini', random_state=42)

# Information gain (entropy)
model_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Log loss
model_logloss = DecisionTreeClassifier(criterion='log_loss', random_state=42)

for name, m in [('Gini', model_gini), ('Entropy', model_entropy), ('Log Loss', model_logloss)]:
    m.fit(X_train, y_train)
    print(f"{name}: Accuracy = {m.score(X_test, y_test):.4f}")
```

### 5. Mathematical Foundation

**Gini Impurity:**
$$G(p) = \sum_{k=1}^{K} p_k (1 - p_k) = 1 - \sum_{k=1}^{K} p_k^2$$

**Entropy:**
$$H(p) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

```python
# Manual Gini calculation
def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

# Manual entropy calculation
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

print(f"Gini impurity: {gini_impurity(y_train):.4f}")
print(f"Entropy: {entropy(y_train):.4f}")
```

---

## Decision Tree Regressor

### 1. Basic Usage

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

print(f"Train R²: {model.score(X_train, y_train):.4f}")
print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

### 2. Split Criteria for Regression

```python
# Mean Squared Error (default)
model_mse = DecisionTreeRegressor(criterion='squared_error', random_state=42)

# Friedman MSE (improved)
model_friedman = DecisionTreeRegressor(criterion='friedman_mse', random_state=42)

# Mean Absolute Error
model_mae = DecisionTreeRegressor(criterion='absolute_error', random_state=42)

# Poisson deviance
model_poisson = DecisionTreeRegressor(criterion='poisson', random_state=42)

for name, m in [('MSE', model_mse), ('Friedman', model_friedman), ('MAE', model_mae)]:
    m.fit(X_train, y_train)
    print(f"{name}: R² = {m.score(X_test, y_test):.4f}")
```

---

## Overfitting Control

### 1. Hyperparameters

```python
# Full control over tree growth
model = DecisionTreeClassifier(
    max_depth=5,              # Maximum depth of tree
    min_samples_split=10,     # Minimum samples to split a node
    min_samples_leaf=5,       # Minimum samples in leaf node
    max_features='sqrt',      # Features to consider at each split
    max_leaf_nodes=20,        # Maximum number of leaf nodes
    min_impurity_decrease=0.01,  # Minimum impurity decrease for split
    random_state=42
)
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Effect of max_depth

```python
import matplotlib.pyplot as plt

depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'b-', label='Train')
plt.plot(depths, test_scores, 'r-', label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Depth vs Performance')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Cost Complexity Pruning

```python
# Get cost complexity pruning path
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Train trees with different alphas
trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Plot accuracy vs alpha
train_scores = [t.score(X_train, y_train) for t in trees]
test_scores = [t.score(X_test, y_test) for t in trees]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, 'b-', label='Train')
plt.plot(ccp_alphas, test_scores, 'r-', label='Test')
plt.xlabel('Alpha (ccp_alpha)')
plt.ylabel('Accuracy')
plt.title('Cost Complexity Pruning')
plt.legend()
plt.show()

# Find best alpha
best_idx = np.argmax(test_scores)
best_alpha = ccp_alphas[best_idx]
print(f"Best alpha: {best_alpha:.6f}")
```

### 4. Cross-Validation for Pruning

```python
from sklearn.model_selection import cross_val_score

# Search for best alpha with CV
alphas = ccp_alphas[::5]  # Sample every 5th alpha
cv_scores = []

for alpha in alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(tree, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

best_alpha = alphas[np.argmax(cv_scores)]
print(f"Best alpha (CV): {best_alpha:.6f}")

# Final model
final_model = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
final_model.fit(X_train, y_train)
print(f"Test Accuracy: {final_model.score(X_test, y_test):.4f}")
```

---

## Feature Importance

### 1. Built-in Feature Importance

```python
# Gini importance (mean decrease in impurity)
importances = model.feature_importances_
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# Sort by importance
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance (Gini)')
plt.tight_layout()
plt.show()
```

### 2. Permutation Importance

```python
from sklearn.inspection import permutation_importance

# More reliable than built-in importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# Sort and plot
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.boxplot([perm_importance.importances[i] for i in sorted_idx[:10]],
            labels=[feature_names[i] for i in sorted_idx[:10]])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Permutation Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3. Gini vs Permutation Importance

```python
# Compare both methods
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gini importance
axes[0].barh(range(X.shape[1]), importances[indices])
axes[0].set_yticks(range(X.shape[1]))
axes[0].set_yticklabels([feature_names[i] for i in indices])
axes[0].set_xlabel('Gini Importance')
axes[0].set_title('Feature Importance (Gini)')

# Permutation importance
axes[1].barh(range(X.shape[1]), perm_importance.importances_mean[sorted_idx])
axes[1].set_yticks(range(X.shape[1]))
axes[1].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[1].set_xlabel('Permutation Importance')
axes[1].set_title('Feature Importance (Permutation)')

plt.tight_layout()
plt.show()
```

---

## Handling Different Data Types

### 1. Categorical Features

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Example with categorical data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'price': [10, 20, 30, 15, 25],
    'sold': [1, 0, 1, 1, 0]
})

# Encode categorical features
encoder = OrdinalEncoder()
X_cat = encoder.fit_transform(df[['color', 'size']])
X = np.hstack([X_cat, df[['price']].values])
y = df['sold'].values

# Trees handle encoded categoricals naturally
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)
```

### 2. Missing Values

```python
# Sklearn trees don't handle NaN directly - use imputation first
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

model = DecisionTreeClassifier()
model.fit(X_imputed, y)
```

### 3. Feature Scaling

```python
# Trees are scale-invariant - no need for scaling
# But scaling doesn't hurt

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model_unscaled = DecisionTreeClassifier(random_state=42)
model_scaled = DecisionTreeClassifier(random_state=42)

model_unscaled.fit(X_train, y_train)
model_scaled.fit(X_scaled, y_train)

print(f"Unscaled Accuracy: {model_unscaled.score(X_test, y_test):.4f}")
print(f"Scaled Accuracy: {model_scaled.score(scaler.transform(X_test), y_test):.4f}")
# Should be nearly identical
```

---

## Tree Properties and Analysis

### 1. Tree Structure

```python
# Number of nodes
print(f"Total nodes: {model.tree_.node_count}")

# Depth
print(f"Max depth: {model.get_depth()}")

# Number of leaves
print(f"Number of leaves: {model.get_n_leaves()}")

# Feature used at each node
print(f"Features at nodes: {model.tree_.feature}")

# Threshold at each node
print(f"Thresholds: {model.tree_.threshold}")
```

### 2. Decision Path

```python
# Get decision path for samples
node_indicator = model.decision_path(X_test[:5])

# Which nodes were visited
for sample_id in range(5):
    node_indices = node_indicator[sample_id].indices
    print(f"Sample {sample_id}: visited nodes {node_indices}")
```

### 3. Leaf Node Analysis

```python
# Which leaf each sample falls into
leaf_indices = model.apply(X_test)
print(f"Leaf indices: {leaf_indices[:10]}")

# Samples per leaf
unique_leaves, counts = np.unique(leaf_indices, return_counts=True)
print(f"Samples per leaf: {dict(zip(unique_leaves, counts))}")
```

---

## Class Imbalance

### 1. Class Weights

```python
# Balanced weights
model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Custom weights
model = DecisionTreeClassifier(class_weight={0: 1, 1: 10}, random_state=42)
model.fit(X_train, y_train)
```

### 2. Sample Weights

```python
# Give more weight to certain samples
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 1] = 3  # Upweight minority class

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

---

## Multi-output Trees

### 1. Multi-output Classification

```python
from sklearn.datasets import make_multilabel_classification

# Multi-label data
X, Y = make_multilabel_classification(n_samples=1000, n_features=10,
                                       n_classes=3, n_labels=2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Single tree for multiple outputs
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

print(f"Predictions shape: {model.predict(X_test).shape}")
```

### 2. Multi-output Regression

```python
from sklearn.datasets import make_regression

# Multi-target regression
X, Y = make_regression(n_samples=1000, n_features=10, n_targets=3, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, Y_train)

print(f"R² scores: {model.score(X_test, Y_test):.4f}")
```

---

## Extra Trees

### 1. Extra-Randomized Trees

```python
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor

# More random than standard decision trees
# Splits are chosen randomly instead of optimally
model = ExtraTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Comparison

```python
# Compare Decision Tree vs Extra Tree
dt = DecisionTreeClassifier(random_state=42)
et = ExtraTreeClassifier(random_state=42)

dt.fit(X_train, y_train)
et.fit(X_train, y_train)

print(f"Decision Tree: {dt.score(X_test, y_test):.4f}")
print(f"Extra Tree: {et.score(X_test, y_test):.4f}")
```

---

## PyTorch Comparison

### 1. Soft Decision Tree

```python
import torch
import torch.nn as nn

class SoftDecisionTree(nn.Module):
    """
    A differentiable soft decision tree.
    Uses sigmoid functions for soft routing.
    """
    def __init__(self, input_dim, depth, n_classes):
        super().__init__()
        self.depth = depth
        self.n_classes = n_classes
        self.n_internal = 2**depth - 1
        self.n_leaves = 2**depth
        
        # Internal nodes: linear layers for soft routing
        self.internal = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(self.n_internal)
        ])
        
        # Leaf nodes: class distributions
        self.leaves = nn.Parameter(torch.randn(self.n_leaves, n_classes))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute routing probabilities
        # Start at root, traverse tree
        # This is a simplified version
        
        # For each sample, compute probability of reaching each leaf
        leaf_probs = torch.ones(batch_size, 1, device=x.device)
        
        for d in range(self.depth):
            # Get nodes at this depth
            start_node = 2**d - 1
            end_node = 2**(d+1) - 1
            
            new_probs = []
            for i in range(start_node, min(end_node, self.n_internal)):
                prob_right = torch.sigmoid(self.internal[i](x))
                prob_left = 1 - prob_right
                # This needs proper indexing for full implementation
        
        # Return class probabilities
        return torch.softmax(self.leaves, dim=1).mean(dim=0).unsqueeze(0).expand(batch_size, -1)

# Note: Full soft decision tree implementation is more complex
# This shows the concept of differentiable routing
```

### 2. Neural Network Alternative

```python
# For tabular data, consider:
# 1. Standard MLPs
# 2. TabNet (attention-based)
# 3. Neural Oblivious Decision Ensembles (NODE)

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# For tabular data, trees often outperform neural networks
# unless you have very large datasets or need end-to-end learning
```

---

## When to Use Decision Trees

### Advantages

1. **Interpretability**: Easy to visualize and explain
2. **No scaling needed**: Invariant to feature scales
3. **Handles mixed types**: Numerical and categorical features
4. **Non-linear relationships**: Captures interactions automatically
5. **Fast inference**: O(log n) prediction time

### Disadvantages

1. **High variance**: Small data changes can create different trees
2. **Overfitting**: Deep trees memorize training data
3. **Biased importance**: Favors high-cardinality features
4. **Axis-aligned splits**: Can't capture diagonal boundaries efficiently

### When to Use

- Need interpretable models
- Baseline before trying ensembles
- Data has complex interactions
- Mixed feature types
- Fast inference is important

### When NOT to Use

- Need stable predictions
- Smooth decision boundaries
- High-dimensional sparse data
- When ensembles (Random Forest, XGBoost) are acceptable

---

## Summary

| Aspect | Decision Tree | Neural Network |
|--------|---------------|----------------|
| **Interpretability** | High | Low |
| **Feature scaling** | Not needed | Required |
| **Training speed** | Fast | Slow |
| **Inference speed** | Fast | Medium |
| **Handles NaN** | With imputation | With imputation |
| **Non-linear** | Yes (axis-aligned) | Yes (any) |
| **Overfitting** | High (prune!) | Regularization |

**Key hyperparameters:**
- `max_depth`: Most important for controlling overfitting
- `min_samples_split`: Prevents overly specific splits
- `min_samples_leaf`: Ensures leaves have enough samples
- `ccp_alpha`: Post-pruning complexity control
