# Multipop Stack

The multipop stack supports a MULTIPOP(k) operation that pops up to $k$ elements at once. Despite individual multipop operations costing up to $O(n)$, any sequence of $n$ push/pop/multipop operations costs $O(n)$ total. This amortized reasoning applies to batch operations in deep learning.

## Definition

A multipop stack supports three operations:

- PUSH(x): Push element $x$ onto the stack. Cost: $O(1)$.
- POP: Remove and return the top element. Cost: $O(1)$.
- MULTIPOP(k): Pop $\min(k, \text{size})$ elements. Cost: $O(\min(k, \text{size}))$.

The amortized cost per operation over any sequence of $n$ operations is $O(1)$ because each element can be popped at most once after being pushed.

$$
\text{Total pops} \leq \text{Total pushes} \leq n \implies \text{Total cost} = O(n)
$$

## Explanation

The key insight is that every element is pushed at most once and popped at most once. So across any sequence of $n$ operations, the total number of individual element movements (pushes + pops) is at most $2n$, giving $O(1)$ amortized cost per operation.

In deep learning, batch operations have similar amortized properties:

- **Gradient zeroing**: `optimizer.zero_grad()` clears all parameter gradients. The cost is proportional to the number of parameters, but it happens once per step alongside many other operations.
- **Cache clearing**: Periodically clearing PyTorch's CUDA memory cache is expensive but amortized over many forward/backward passes.
- **Checkpoint purging**: Deleting old model checkpoints in bulk is expensive but infrequent.

## Examples

```python
import torch

class MultipopStack:
    def __init__(self):
        self.items = []
        self.total_ops = 0

    def push(self, x):
        self.items.append(x)
        self.total_ops += 1

    def multipop(self, k):
        actual = min(k, len(self.items))
        for _ in range(actual):
            self.items.pop()
        self.total_ops += actual
        return actual

stack = MultipopStack()
# Push 100 elements
for i in range(100):
    stack.push(i)
# Multipop all at once
popped = stack.multipop(100)
print(f"Pushed: 100, Popped: {popped}, Total ops: {stack.total_ops}")
print(f"Amortized per operation: {stack.total_ops / 101:.2f}")

# Deep learning analogy: gradient accumulation + bulk zero
model = torch.nn.Linear(10, 1)
accumulate_steps = 4
x = torch.randn(8, 10)
y = torch.randn(8, 1)

for step in range(accumulate_steps):
    loss = torch.nn.functional.mse_loss(model(x), y) / accumulate_steps
    loss.backward()  # accumulate (push)
# Single optimizer step + zero_grad (multipop equivalent)
torch.optim.SGD(model.parameters(), lr=0.01).step()
model.zero_grad()  # bulk clear all gradients
print(f"Accumulated {accumulate_steps} steps, one bulk update")
```
