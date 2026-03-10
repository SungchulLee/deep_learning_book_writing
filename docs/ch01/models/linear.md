# Linear Models

Linear models are the foundation of supervised learning and the building block of neural networks. Every fully connected layer in a deep network is a linear model followed by a nonlinearity, so understanding linear regression and logistic regression is prerequisite to understanding deep learning.

## Definition

A linear model predicts the output as a linear combination of input features:

$$
\hat{y} = \mathbf{x}^\top \boldsymbol{\beta} + \beta_0
$$

For regression, the output is continuous. For classification, the linear combination passes through a sigmoid (binary) or softmax (multiclass) to produce probabilities. The closed-form solution for least-squares regression is the normal equation:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

## Explanation

**Regularization** prevents overfitting by penalizing large weights:

- **Ridge (L2)**: Minimizes $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha \|\boldsymbol{\beta}\|^2$. Shrinks all coefficients toward zero but never to exactly zero. Use when features are correlated.
- **Lasso (L1)**: Minimizes $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \alpha \|\boldsymbol{\beta}\|_1$. Drives some coefficients to exactly zero, performing feature selection.
- **ElasticNet**: Combines L1 and L2 penalties. Use when features are correlated and you want sparsity.

**Connection to deep learning**: A single `nn.Linear` layer in PyTorch implements exactly this model. L2 regularization corresponds to `weight_decay` in the optimizer. L1 regularization must be added manually to the loss.

## Examples

```python
import torch
import torch.nn as nn
import numpy as np

# Generate regression data
np.random.seed(42)
n, d = 200, 5
X_np = np.random.randn(n, d)
true_w = np.array([3.0, -1.5, 0.0, 0.0, 2.0])
y_np = X_np @ true_w + 0.5 * np.random.randn(n)

# Normal equation (closed-form)
beta_hat = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
print(f"True weights:     {true_w}")
print(f"Estimated weights:{np.round(beta_hat, 2)}")

# PyTorch linear regression with SGD
X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.float32)

model = nn.Linear(d, 1, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # L2
loss_fn = nn.MSELoss()

for epoch in range(500):
    pred = model(X_t).squeeze()
    loss = loss_fn(pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

learned = model.weight.detach().numpy().flatten()
print(f"PyTorch weights:  {np.round(learned, 2)}")
print(f"Final MSE loss:   {loss.item():.4f}")
```
