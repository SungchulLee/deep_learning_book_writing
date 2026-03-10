# Dynamic Array

The dynamic array (Python list, C++ vector) doubles its capacity when full, achieving $O(1)$ amortized append despite occasional $O(n)$ copy operations. This pattern directly applies to how PyTorch accumulates tensors during training (e.g., collecting losses or predictions).

## Definition

A dynamic array maintains a backing buffer of capacity $C$. When an append would exceed $C$, the array allocates a new buffer of capacity $2C$ and copies all elements. The amortized cost of $n$ appends is:

$$
\hat{c} = \frac{\sum_{i=1}^{n} c_i}{n} = \frac{n + n/2 + n/4 + \cdots}{n} < \frac{2n}{n} = O(1)
$$

## Explanation

The doubling strategy ensures that expensive copy operations happen infrequently. After a copy, the array has $C/2$ elements in a buffer of size $C$, so the next $C/2$ appends are cheap ($O(1)$ each). These cheap operations "pay" for the next doubling.

In deep learning, this pattern appears when:

- **Accumulating predictions**: Appending model outputs to a list during evaluation. Pre-allocating a tensor of the correct size avoids dynamic resizing entirely.
- **Variable-length sequences**: Collecting tokens during autoregressive generation.
- **Loss history**: Appending scalar losses to a list for logging.

The practical lesson: when you know the final size, pre-allocate. When you do not, Python lists provide $O(1)$ amortized append, but converting to a tensor at the end incurs an $O(n)$ copy.

## Examples

```python
import torch

# Dynamic accumulation (common in eval loops)
model = torch.nn.Linear(5, 3)
data = [torch.randn(5) for _ in range(100)]

# Bad: repeatedly torch.cat (O(n^2) total)
# Good: accumulate in list, cat once (O(n) total)
preds = []
with torch.no_grad():
    for x in data:
        preds.append(model(x))
all_preds = torch.stack(preds)  # single O(n) copy
print(f"Predictions shape: {all_preds.shape}")

# Even better: pre-allocate when size is known
all_preds_pre = torch.empty(len(data), 3)
with torch.no_grad():
    for i, x in enumerate(data):
        all_preds_pre[i] = model(x)
print(f"Pre-allocated shape: {all_preds_pre.shape}")
print(f"Results match: {torch.allclose(all_preds, all_preds_pre)}")
```
