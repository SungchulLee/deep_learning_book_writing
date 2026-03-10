# Correctness

Correctness means an algorithm or model produces the right output for every valid input. In deep learning, correctness encompasses both implementation correctness (the code does what you intend) and statistical correctness (the model generalizes beyond training data).

## Definition

An algorithm is correct if for every valid input $x$, it halts and produces the expected output:

$$
\forall \, x \in \mathcal{X}: \; f(x) = \text{Expected}(x)
$$

For deterministic algorithms, correctness is absolute. For neural networks, correctness is statistical: we seek low expected loss $\mathbb{E}[\ell(f(x), y)]$ over the data distribution.

## Explanation

**Loop invariants** are the classical technique for proving algorithm correctness. A loop invariant is a property that holds before and after every iteration:

1. **Initialization**: True before the first iteration
2. **Maintenance**: If true before an iteration, it remains true after
3. **Termination**: When the loop ends, the invariant implies correctness

In deep learning, the analogous concept is the **training invariant**: that the loss monotonically decreases (in expectation) with each gradient step under appropriate learning rate. Verifying correctness in practice means:

- **Gradient checking**: Comparing autograd gradients against numerical finite-difference gradients
- **Overfitting a single batch**: A correct model should achieve near-zero loss on a single batch, confirming the architecture and loss function work together
- **Unit testing tensor shapes**: Asserting intermediate tensor shapes at each layer

## Examples

```python
import torch
import torch.nn as nn

# Gradient checking: verify autograd correctness
def numerical_gradient(f, x, eps=1e-5):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone(); x_plus.view(-1)[i] += eps
        x_minus = x.clone(); x_minus.view(-1)[i] -= eps
        grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

x = torch.randn(3, requires_grad=True)
f = lambda x: (x ** 3).sum()
f(x).backward()

num_grad = numerical_gradient(f, x.detach())
print(f"Autograd:   {x.grad.tolist()}")
print(f"Numerical:  {num_grad.tolist()}")
print(f"Match: {torch.allclose(x.grad, num_grad, atol=1e-4)}")

# Overfit a single batch (correctness test)
model = nn.Linear(5, 2)
x_batch = torch.randn(8, 5)
y_batch = torch.randint(0, 2, (8,))
opt = torch.optim.Adam(model.parameters(), lr=0.1)
for _ in range(200):
    loss = nn.functional.cross_entropy(model(x_batch), y_batch)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"Single-batch loss: {loss.item():.6f} (should be ~0)")
```
