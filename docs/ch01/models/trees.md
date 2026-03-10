# Decision Trees

Decision trees recursively partition the feature space by axis-aligned splits, creating interpretable models that require no feature scaling. They are the building blocks of ensemble methods (Random Forests, Gradient Boosting) and motivate the piecewise-linear behavior of ReLU networks.

## Definition

A decision tree partitions input space by greedily selecting the feature $j$ and threshold $t$ that minimize impurity after splitting. For classification, the two standard impurity measures are:

$$
\text{Gini: } G(p) = 1 - \sum_{k=1}^{K} p_k^2 \qquad \text{Entropy: } H(p) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

where $p_k$ is the fraction of class $k$ in a node. Each leaf outputs the majority class (classification) or mean target (regression).

## Explanation

Trees are greedy: at each node, they pick the locally optimal split without considering future splits. This makes training fast but means the resulting tree may not be globally optimal.

**Overfitting control** is the central challenge:

- `max_depth`: Limits tree depth. The most important regularization parameter.
- `min_samples_leaf`: Ensures each leaf has sufficient support.
- **Cost-complexity pruning** (`ccp_alpha`): Removes subtrees that provide little impurity reduction relative to their complexity.

**Strengths**: Interpretable, no scaling needed, handles mixed feature types, fast inference ($O(\log n)$ per sample).

**Weaknesses**: High variance (small data changes produce different trees), axis-aligned splits cannot efficiently capture diagonal boundaries, and single trees tend to overfit.

**Connection to deep learning**: A ReLU network with one hidden layer implements a piecewise-linear function similar to a decision tree. However, neural networks learn oblique (non-axis-aligned) partitions and share parameters across regions, giving them much greater capacity.

## Examples

```python
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score

X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Select max_depth via cross-validation
best_depth, best_score = 1, 0.0
for d in range(1, 15):
    score = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42),
                            X_tr, y_tr, cv=5).mean()
    if score > best_score:
        best_depth, best_score = d, score
print(f"Best depth={best_depth}, CV accuracy={best_score:.4f}")

tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
tree.fit(X_tr, y_tr)
print(f"Test accuracy: {tree.score(X_te, y_te):.4f}")
print(f"Nodes: {tree.tree_.node_count}, Leaves: {tree.get_n_leaves()}")

# Feature importance (Gini-based)
importances = tree.feature_importances_
top3 = np.argsort(importances)[-3:][::-1]
for i in top3:
    print(f"  Feature {i}: importance={importances[i]:.4f}")

# Compare: a simple neural network on the same data
X_t = torch.tensor(X_tr, dtype=torch.float32)
y_t = torch.tensor(y_tr, dtype=torch.long)
net = torch.nn.Sequential(
    torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(200):
    loss = torch.nn.functional.cross_entropy(net(X_t), y_t)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    nn_acc = (net(X_te_t).argmax(1).numpy() == y_te).mean()
print(f"Neural net accuracy: {nn_acc:.4f}")
```
