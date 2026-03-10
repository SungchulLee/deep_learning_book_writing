# Pseudocode Conventions

Pseudocode describes algorithms in a language-independent way. In deep learning literature, pseudocode communicates training loops, architectures, and optimization procedures without tying them to a specific framework.

## Definition

Pseudocode is a structured, informal notation for algorithms that uses standard mathematical and programming conventions:

$$
\begin{array}{ll}
\textbf{for } i = 1 \textbf{ to } n & \text{Iterate over a range} \\
\textbf{while } \text{condition} & \text{Loop while true} \\
\textbf{return } \text{value} & \text{Output a result} \\
x \leftarrow f(x) & \text{Assignment}
\end{array}
$$

## Explanation

Deep learning papers use pseudocode to specify training algorithms precisely. A standard training loop in pseudocode:

```
TRAIN(model, data, lr, epochs)
  for epoch = 1 to epochs
    for (x, y) in data
      y_hat = model(x)          // forward pass
      L = loss(y_hat, y)        // compute loss
      g = gradient(L, params)   // backward pass
      params = params - lr * g  // parameter update
  return params
```

This pseudocode abstracts away framework details (PyTorch vs TensorFlow, GPU placement, mixed precision) while capturing the essential logic. When reading papers, pseudocode is often the most precise description of the proposed method.

Key conventions in deep learning pseudocode: $\nabla_\theta$ denotes the gradient with respect to parameters, $\leftarrow$ denotes assignment (as opposed to $=$ for equality), and $\sim$ denotes sampling from a distribution.

## Examples

```python
import torch
import torch.nn as nn

# Translating pseudocode to PyTorch
# Pseudocode: ADAM(params, lr, beta1, beta2, eps)
#   m = 0, v = 0, t = 0
#   repeat:
#     t = t + 1
#     g = gradient(loss, params)
#     m = beta1 * m + (1 - beta1) * g
#     v = beta2 * v + (1 - beta2) * g^2
#     m_hat = m / (1 - beta1^t)
#     v_hat = v / (1 - beta2^t)
#     params = params - lr * m_hat / (sqrt(v_hat) + eps)

# PyTorch implementation of the pseudocode above
model = nn.Linear(5, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

x = torch.randn(32, 5)
y = torch.randn(32, 1)

for step in range(100):
    y_hat = model(x)                           # forward pass
    loss = nn.functional.mse_loss(y_hat, y)    # compute loss
    optimizer.zero_grad()                       # clear gradients
    loss.backward()                            # backward pass (compute g)
    optimizer.step()                           # parameter update

print(f"Final loss: {loss.item():.6f}")
print(f"Parameters: {list(model.parameters())[0].shape}")
```
