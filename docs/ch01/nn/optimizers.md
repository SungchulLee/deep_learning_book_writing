# Loss Functions and

Training neural networks requires defining a **loss function** and an **optimizer** to adjust parameters.

---

## Loss functions

A loss function measures model error:

```python
import torch.nn as nn

loss_fn = nn.MSELoss()
```

Common losses:
- Mean Squared Error (MSE)
- Cross-Entropy Loss
- Negative Log-Likelihood

---

## Computing loss

```python
pred = model(x)
loss = loss_fn(pred, y)
```

The loss is a scalar tensor.

---

## Optimizers

Optimizers update parameters using gradients:

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Popular optimizers:
- SGD
- Adam
- RMSprop

---

## Training step

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

This is the core training loop.

---

## Financial

Optimizers are used for:
- model calibration,
- risk minimization,
- learning pricing functionals.

---

## Key takeaways

- Loss functions define objectives.
- Optimizers update parameters via gradients.
- Training loops follow a standard pattern.
