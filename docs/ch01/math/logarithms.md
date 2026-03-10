# Logarithms

Logarithms are central to deep learning: the natural logarithm defines cross-entropy loss, log-probabilities stabilize numerical computation, and logarithmic scaling appears in learning rate schedules and information-theoretic quantities.

## Definition

The logarithm base $b$ of $x$ is the exponent to which $b$ must be raised to produce $x$:

$$
\log_b x = y \iff b^y = x
$$

The natural logarithm $\ln x = \log_e x$ is the most important variant in deep learning because it is the inverse of the exponential function and arises naturally in maximum likelihood estimation.

## Explanation

Key properties used throughout deep learning:

$$
\begin{array}{ll}
\ln(xy) = \ln x + \ln y \\
\ln(x/y) = \ln x - \ln y \\
\ln(x^n) = n \ln x \\
\log_b x = \frac{\ln x}{\ln b} & \text{(change of base)}
\end{array}
$$

Why logarithms matter in deep learning:

- **Cross-entropy loss**: The negative log-likelihood $-\ln p(y \mid x)$ is the standard classification loss. The logarithm converts products of probabilities into sums, which are numerically stable and easy to differentiate.
- **Log-sum-exp trick**: Computing $\ln \sum_i e^{x_i}$ via $\max(x) + \ln \sum_i e^{x_i - \max(x)}$ prevents overflow.
- **Information theory**: Entropy $H = -\sum p_i \ln p_i$ and KL divergence are defined using logarithms.
- **Logarithmic scaling**: Learning rates often decay on a log scale (e.g., from $10^{-2}$ to $10^{-5}$).

## Examples

```python
import torch
import numpy as np

# Cross-entropy loss uses log internally
logits = torch.tensor([2.0, 1.0, 0.1])
target = 0  # correct class
probs = torch.softmax(logits, dim=0)
loss = -torch.log(probs[target])
print(f"Softmax probs: {probs.tolist()}")
print(f"Cross-entropy loss: {loss.item():.4f}")

# Log-sum-exp trick for numerical stability
x = torch.tensor([1000.0, 1001.0, 1002.0])
# Naive: overflow
# stable: subtract max first
max_x = x.max()
stable = max_x + torch.log(torch.exp(x - max_x).sum())
print(f"Log-sum-exp (stable): {stable.item():.4f}")

# PyTorch built-in
builtin = torch.logsumexp(x, dim=0)
print(f"Log-sum-exp (builtin): {builtin.item():.4f}")

# Change of base
val = 1024.0
print(f"log2({val:.0f}) = {np.log2(val):.2f}")
print(f"ln({val:.0f}) = {np.log(val):.4f}")
```
