# Accounting Method

When analyzing a sequence of operations on a data structure, some operations are cheap while others are expensive. The aggregate method assigns the same amortized cost to every operation, but this can be too coarse. The accounting method provides a finer-grained analysis by assigning different amortized costs to different operations, allowing cheap operations to overpay and store credit that subsidizes future expensive operations.

## Definition

In the accounting method, each operation $i$ receives an **amortized cost** $\hat{c}_i$ that may differ from its actual cost $c_i$. The fundamental requirement is that the total amortized cost is an upper bound on the total actual cost:

$$
\sum_{i=1}^{n} \hat{c}_i \geq \sum_{i=1}^{n} c_i
$$

When an operation's amortized cost $\hat{c}_i$ exceeds its actual cost $c_i$, the difference $\hat{c}_i - c_i$ is deposited as **credit** on the data structure. When an operation's actual cost exceeds its amortized cost, the shortfall is paid from accumulated credit.

The key invariant is that the total accumulated credit must remain non-negative after every operation:

$$
\text{Credit after operation } j = \sum_{i=1}^{j} (\hat{c}_i - c_i) \geq 0 \quad \text{for all } j = 1, 2, \ldots, n
$$

This non-negativity constraint ensures that the amortized costs never undercount the actual costs for any prefix of operations.

## How It Works

The accounting method proceeds in three steps:

1. **Choose amortized costs.** Assign $\hat{c}_i$ to each type of operation. This choice requires insight into the data structure's behavior.
2. **Track credit.** After each operation, compute the credit as $\hat{c}_i - c_i$. Credit may be associated with specific elements or locations in the data structure.
3. **Verify non-negativity.** Prove that the accumulated credit never drops below zero across any sequence of operations.

Unlike the aggregate method (which assigns a uniform cost), the accounting method allows different operation types to carry different amortized costs. Unlike the potential method (which defines a global potential function), the accounting method tracks credit at a local, per-element level.

## Example: Stack with Multipop

Consider a stack supporting three operations: `PUSH`, `POP`, and `MULTIPOP(k)`, where `MULTIPOP(k)` pops the top $\min(k, s)$ elements from a stack of size $s$.

**Actual costs:**

- `PUSH`: $c = 1$
- `POP`: $c = 1$
- `MULTIPOP(k)`: $c = \min(k, s)$

A single `MULTIPOP` can cost up to $O(n)$, so a naive worst-case analysis of $n$ operations gives $O(n^2)$.

**Amortized costs (accounting method):**

Assign the following amortized costs:

- `PUSH`: $\hat{c} = 2$ (overpays by 1; the extra unit is deposited as credit on the pushed element)
- `POP`: $\hat{c} = 0$ (paid by the credit stored on the popped element)
- `MULTIPOP(k)`: $\hat{c} = 0$ (each popped element pays for itself using its stored credit)

**Credit invariant:** Each element on the stack carries exactly 1 unit of credit, deposited when it was pushed. Since every element that gets popped (by either `POP` or `MULTIPOP`) was previously pushed, there is always sufficient credit to pay for the removal. The total credit equals the stack size, which is always non-negative.

**Result:** The total amortized cost of $n$ operations is at most $2n$ (since each operation costs at most $\hat{c} = 2$), giving an amortized cost of $O(1)$ per operation.

## Example: Dynamic Array

Consider a dynamic array that doubles its capacity when full. The `APPEND` operation has two cases:

- **No resize:** Insert the element in $O(1)$.
- **Resize:** Allocate a new array of double the capacity, copy all $s$ existing elements, then insert. The actual cost is $s + 1$.

**Amortized costs (accounting method):**

Assign $\hat{c} = 3$ for every `APPEND`:

- 1 unit pays for inserting the element itself
- 1 unit is stored as credit on the newly inserted element
- 1 unit is stored as credit on the element at position $s/2$ (an element that was present before the last resize but has not yet "saved" for the next one)

When a resize occurs at size $s$, the $s$ elements each carry 1 unit of credit, providing exactly enough to pay for copying them into the new array.

**Result:** The total amortized cost of $n$ appends is at most $3n$, giving an amortized cost of $O(1)$ per append.

## Python Example

```python
"""
Accounting method demonstration for a dynamic array.

Shows how assigning an amortized cost of 3 per append covers all
actual costs including expensive resizing operations.
"""


# ===================================================================
# Dynamic Array with Credit Tracking
# ===================================================================
class DynamicArray:
    """Dynamic array that tracks actual cost and credit per element."""

    def __init__(self):
        self.capacity = 1
        self.data = [None] * self.capacity
        self.size = 0
        self.total_actual_cost = 0
        self.total_amortized_cost = 0
        self.credit = 0

    def append(self, value):
        """Append with amortized cost tracking."""
        amortized = 3  # accounting method charge
        actual = 1     # cost for the insertion itself

        if self.size == self.capacity:
            # Resize: copy all elements (actual cost += size)
            actual += self.size
            new_data = [None] * (2 * self.capacity)
            for i in range(self.size):
                new_data[i] = self.data[i]
            self.data = new_data
            self.capacity *= 2

        self.data[self.size] = value
        self.size += 1

        self.total_actual_cost += actual
        self.total_amortized_cost += amortized
        self.credit += (amortized - actual)

    def stats(self):
        """Return cost statistics."""
        return {
            "size": self.size,
            "capacity": self.capacity,
            "total_actual": self.total_actual_cost,
            "total_amortized": self.total_amortized_cost,
            "credit": self.credit,
        }


# ===================================================================
# Demonstration
# ===================================================================
if __name__ == "__main__":
    arr = DynamicArray()
    print(f"{'Op':>4} {'Actual':>7} {'Amortized':>10} {'Credit':>7}")
    print("-" * 32)
    for i in range(1, 17):
        prev_actual = arr.total_actual_cost
        arr.append(i)
        actual_this = arr.total_actual_cost - prev_actual
        print(
            f"{i:>4} {actual_this:>7} {3:>10} {arr.credit:>7}"
        )

    stats = arr.stats()
    print(f"\nTotal actual cost:    {stats['total_actual']}")
    print(f"Total amortized cost: {stats['total_amortized']}")
    print(f"Remaining credit:     {stats['credit']}")
    print(f"Credit >= 0:          {stats['credit'] >= 0}")
    avg = stats["total_actual"] / stats["size"]
    print(f"Average actual cost:  {avg:.2f}")
```

## Comparison with Other Methods

| Aspect | Aggregate | Accounting | Potential |
|--------|-----------|------------|-----------|
| Cost assignment | Same for all operations | Different per operation type | Derived from potential function |
| Credit tracking | Not explicit | Per-element credit | Global potential function |
| Flexibility | Low | Medium | High |
| Best for | Simple uniform analysis | Per-operation cost bounds | Complex multi-operation analysis |

The accounting method is especially useful when credit can be naturally associated with specific elements in the data structure, as in the stack and dynamic array examples above.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.
