# Support Vector Machines

Support Vector Machines find the maximum-margin separating hyperplane between classes. Understanding the SVM objective clarifies the role of margin, regularization, and kernel methods -- concepts that directly inform the design of loss functions and feature representations in deep learning.

## Definition

The SVM solves the constrained optimization problem:

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \;\; \xi_i \geq 0
$$

The margin is $2 / \|\mathbf{w}\|$. The parameter $C$ trades off margin width against misclassification. Points on or inside the margin are **support vectors** -- only these determine the decision boundary.

## Explanation

**Kernel trick**: SVMs can learn nonlinear boundaries by mapping inputs into a higher-dimensional space via a kernel function $K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$ without computing $\phi$ explicitly. The RBF kernel $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ maps to infinite-dimensional space.

**Key hyperparameters**:

- $C$: Small $C$ allows wider margin with more violations (better generalization). Large $C$ enforces strict classification (risk of overfitting).
- $\gamma$ (RBF kernel): Controls the influence radius of each support vector. Large $\gamma$ creates complex boundaries.

**Connection to deep learning**: The hinge loss $\max(0, 1 - y \cdot f(x))$ used in SVMs is closely related to the ReLU activation. Neural networks can be viewed as learning the feature map $\phi$ that SVMs assume is given by the kernel.

**Limitations**: Training complexity is $O(n^2)$ to $O(n^3)$, making SVMs impractical for large datasets where neural networks excel.

## Examples

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data (scaling required for SVMs)
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

# Sklearn SVM with RBF kernel
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_tr_s, y_tr)
print(f"SVM accuracy: {svm.score(X_te_s, y_te):.4f}")
print(f"Support vectors: {svm.n_support_}")

# PyTorch: linear SVM via hinge loss
X_t = torch.tensor(X_tr_s, dtype=torch.float32)
y_t = torch.tensor(2 * y_tr - 1, dtype=torch.float32)  # {0,1} -> {-1,+1}

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

for _ in range(500):
    out = model(X_t).squeeze()
    loss = torch.clamp(1 - y_t * out, min=0).mean()  # hinge loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32)
    preds = (model(X_te_t).squeeze() > 0).long().numpy()
    acc = (preds == y_te).mean()
    print(f"PyTorch SVM accuracy: {acc:.4f}")
```
