# Binary Counter

The binary counter is a classic example of amortized analysis where incrementing a $k$-bit counter costs $O(1)$ amortized per increment despite individual increments flipping up to $k$ bits. This pattern mirrors gradient accumulation counters and step schedulers in deep learning.

## Definition

A $k$-bit binary counter stores an integer in $k$ bits. The INCREMENT operation adds 1, flipping bits from position 0 upward until a 0-bit is found. The worst-case cost of a single increment is $O(k)$ (all bits flip), but the amortized cost over $n$ increments is:

$$
\hat{c} = \frac{\text{total bit flips}}{n} \leq \frac{2n}{n} = O(1)
$$

## Explanation

The key observation is that bit $i$ flips once every $2^i$ increments:

- Bit 0 flips every increment: $n$ times total
- Bit 1 flips every 2 increments: $n/2$ times total
- Bit $i$ flips every $2^i$ increments: $n/2^i$ times total

Total flips over $n$ increments:

$$
\sum_{i=0}^{k-1} \frac{n}{2^i} < 2n
$$

This geometric series argument appears in deep learning when analyzing:

- **Learning rate warmup schedulers** that update the rate at geometrically decreasing frequencies
- **Exponential moving averages** in batch normalization, where the contribution of each past batch decays geometrically
- **Gradient accumulation counters** that trigger parameter updates at fixed intervals

## Examples

```python
import torch

# Binary counter simulation
def increment(counter):
    """Increment binary counter, return number of bit flips."""
    flips = 0
    i = 0
    while i < len(counter) and counter[i] == 1:
        counter[i] = 0
        flips += 1
        i += 1
    if i < len(counter):
        counter[i] = 1
        flips += 1
    return flips

k = 8  # 8-bit counter
counter = [0] * k
total_flips = 0
n = 256

for step in range(n):
    total_flips += increment(counter)

print(f"Increments: {n}, Total flips: {total_flips}")
print(f"Amortized flips per increment: {total_flips / n:.2f}")
print(f"Worst case per increment: {k}")

# Deep learning analogy: exponential moving average
# Each past value's influence decays geometrically (like bit flip frequency)
momentum = 0.9
values = torch.randn(100)
ema = torch.tensor(0.0)
for v in values:
    ema = momentum * ema + (1 - momentum) * v
print(f"\nEMA of 100 random values: {ema.item():.4f}")
print(f"Direct mean: {values.mean().item():.4f}")
```
