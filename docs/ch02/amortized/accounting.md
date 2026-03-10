# Accounting Method

The accounting method assigns different charges to different operations so that cheap operations overpay and expensive operations are subsidized. This perspective explains why dynamic memory allocation in PyTorch's autograd graph has low average cost despite occasional expensive reallocations.

## Definition

In the accounting method, each operation $i$ receives an amortized cost $\hat{c}_i$ such that:

$$
\sum_{i=1}^{n} \hat{c}_i \geq \sum_{i=1}^{n} c_i
$$

where $c_i$ is the actual cost. The excess payment (credit) is stored and used to pay for future expensive operations. The amortized cost per operation is $\hat{c}_i$, which may differ from $c_i$.

## Explanation

The accounting method is conceptually similar to how GPU memory allocators work in PyTorch:

- **Cheap operations overpay**: Small tensor allocations are rounded up to fixed block sizes. The "overpayment" (unused memory within the block) acts as credit.
- **Expensive operations use credit**: When a large allocation is needed, the allocator can reuse previously freed blocks without calling the expensive `cudaMalloc`.
- **Amortized guarantee**: Over a sequence of allocations and deallocations, the average cost is much lower than the worst-case cost of any single allocation.

The key constraint is that accumulated credit must never go negative -- you cannot borrow against future cheap operations.

## Examples

```python
import torch

# Demonstrate amortized allocation: PyTorch's caching allocator
# First allocation is expensive (calls cudaMalloc internally on GPU)
# Subsequent allocations of same/smaller size reuse cached memory

sizes = []
for i in range(1, 11):
    t = torch.randn(i * 1000)
    sizes.append(t.numel())
    # On GPU, PyTorch would reuse cached blocks here
print(f"Allocation sizes: {sizes}")

# Accounting method for dynamic list (like accumulating losses)
class AmortizedList:
    """List with O(1) amortized append (doubles capacity)."""
    def __init__(self):
        self.data = [None] * 2
        self.size = 0
        self.total_cost = 0

    def append(self, val):
        if self.size == len(self.data):
            # Expensive: copy all elements (cost = size)
            self.data = self.data + [None] * len(self.data)
            self.total_cost += self.size
        self.data[self.size] = val
        self.size += 1
        self.total_cost += 1  # cheap: single write

lst = AmortizedList()
for i in range(1000):
    lst.append(i)
print(f"Total ops: {lst.size}, total cost: {lst.total_cost}")
print(f"Amortized cost per op: {lst.total_cost / lst.size:.2f}")
```
