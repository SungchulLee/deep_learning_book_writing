# Autograd Mechanics

**Automatic differentiation (autograd)** computes derivatives of functions defined by tensor operations, enabling gradient-based optimization.

---

## Why autograd?

In optimization and learning, we need:
- gradients of loss functions,
- sensitivities with respect to parameters.

Manual differentiation is error-prone and unscalable.

---

## Computational graphs

Autograd builds a **computational graph**:
- nodes = tensor operations,
- edges = data dependencies.

During the forward pass, the graph is recorded.

---

## Backpropagation

Calling backward computes gradients:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 1
y.backward()
x.grad   # 4.0
```

Gradients flow backward through the graph.

---

## Leaf tensors and

- Only tensors with `requires_grad=True` accumulate gradients.
- Gradients are stored in `.grad`.

Gradients accumulate unless reset.

---

## Financial

Autograd enables:
- calibration via gradient descent,
- adjoint methods,
- differentiable Monte Carlo.

---

## Key takeaways

- Autograd builds computational graphs.
- Backpropagation computes gradients automatically.
- Essential for modern quantitative modeling.
