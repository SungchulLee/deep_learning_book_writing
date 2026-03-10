# Aggregate Method

The aggregate method computes the total cost of a sequence of $n$ operations and divides by $n$ to obtain the amortized cost per operation. This is the simplest form of amortized analysis and directly applies to understanding the cost of training loops in deep learning.

## Definition

Given $n$ operations with individual costs $c_1, c_2, \ldots, c_n$, the amortized cost per operation under the aggregate method is:

$$
\hat{c} = \frac{1}{n} \sum_{i=1}^{n} c_i
$$

Every operation receives the same amortized cost $\hat{c}$, regardless of its actual cost.

## Explanation

The aggregate method answers: "What is the average cost if we run $n$ operations?" This is directly relevant to deep learning:

- **Training loop cost**: If most gradient steps are cheap but every $k$-th step involves an expensive operation (checkpointing, logging, validation), the amortized cost per step is the total cost divided by the number of steps.
- **Dynamic batching**: Some batches may require more computation (longer sequences in NLP). The aggregate cost across an epoch tells you the true throughput.
- **Gradient accumulation**: Accumulating gradients over $k$ mini-batches before updating is cheap per step (no optimizer step), but the $k$-th step is more expensive (optimizer step + zero_grad). The amortized cost includes both.

The limitation of the aggregate method is that it assigns the same amortized cost to every operation, even when different operations have genuinely different costs.

## Examples

```python
import torch
import time

# Aggregate analysis of a training loop with periodic validation
n_steps = 100
val_every = 10
total_cost = 0.0

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

for step in range(1, n_steps + 1):
    start = time.time()
    # Training step (cheap)
    loss = torch.nn.functional.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % val_every == 0:
        # Validation step (expensive: extra forward pass)
        with torch.no_grad():
            val_loss = torch.nn.functional.mse_loss(model(x), y)
    total_cost += time.time() - start

amortized = total_cost / n_steps
print(f"Total time: {total_cost*1000:.1f} ms")
print(f"Amortized per step: {amortized*1000:.3f} ms")
print(f"Validation steps: {n_steps // val_every}")
```
