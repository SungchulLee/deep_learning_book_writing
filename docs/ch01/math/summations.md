# Summation Formulas

Summation formulas are essential tools for analyzing computational cost in deep learning. They quantify the total operations in nested loops, the cumulative effect of learning rate schedules, and the parameter counts in multi-layer architectures.

## Definition

A summation formula gives a closed-form expression for $\sum_{i=a}^{b} f(i)$. The three most important families are:

$$
\text{Arithmetic: } \sum_{i=1}^{n} i = \frac{n(n+1)}{2} \qquad \text{Geometric: } \sum_{i=0}^{n} r^i = \frac{r^{n+1}-1}{r-1} \; (r \neq 1)
$$

$$
\text{Harmonic: } H_n = \sum_{i=1}^{n} \frac{1}{i} = \ln n + \gamma + O\!\left(\frac{1}{n}\right)
$$

where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant.

## Explanation

Each series type appears in different contexts:

- **Arithmetic series**: Counting total parameters in a network with linearly growing layer widths. If layer $i$ has $id$ neurons, total parameters scale as $\sum_{i=1}^{L} i \cdot d^2 = \Theta(L^2 d^2)$.
- **Geometric series**: Analyzing exponential learning rate decay. If $\eta_t = \eta_0 r^t$ with $r < 1$, the cumulative step size converges to $\eta_0 / (1 - r)$.
- **Harmonic series**: Appears in the analysis of stochastic gradient descent convergence rates and in the coupon collector problem (relevant to data sampling with replacement).

Additional useful formulas:

$$
\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6} \qquad \sum_{i=0}^{\log_2 n} 2^i = 2n - 1
$$

## Examples

```python
import torch
import numpy as np

# Parameter count in a network with linearly growing widths
d = 64
num_layers = 6
widths = [i * d for i in range(1, num_layers + 1)]
total_params = sum(widths[i] * widths[i + 1] for i in range(len(widths) - 1))
closed_form = d * d * sum(i * (i + 1) for i in range(1, num_layers))
print(f"Layer widths: {widths}")
print(f"Total weight params: {total_params}")

# Geometric series: cumulative learning rate with exponential decay
eta_0, r, T = 0.01, 0.9, 50
cumulative = sum(eta_0 * r ** t for t in range(T))
closed = eta_0 * (1 - r ** T) / (1 - r)
print(f"\nCumulative LR (sum):    {cumulative:.6f}")
print(f"Cumulative LR (closed): {closed:.6f}")
print(f"Limit as T -> inf:      {eta_0 / (1 - r):.6f}")

# Verify arithmetic sum with torch
n = torch.arange(1, 101, dtype=torch.float32)
actual = n.sum().item()
formula = 100 * 101 / 2
print(f"\nsum(1..100) = {actual:.0f}, formula = {formula:.0f}")
```
