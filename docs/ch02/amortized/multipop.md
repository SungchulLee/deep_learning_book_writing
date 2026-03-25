# Multipop Stack

The multipop stack is the most commonly used introductory example for amortized analysis. A standard stack supports `PUSH` and `POP` in $O(1)$ each. Adding a `MULTIPOP(k)` operation that removes up to $k$ elements at once introduces a single operation with worst-case cost $O(n)$. Naive worst-case analysis of $n$ mixed operations gives $O(n^2)$, but amortized analysis shows the true cost is only $O(n)$ total, or $O(1)$ per operation.

## Operations

The multipop stack supports three operations on a stack of current size $s$:

| Operation | Description | Actual cost |
|-----------|-------------|-------------|
| `PUSH(x)` | Push element $x$ onto the stack | $1$ |
| `POP()` | Remove and return the top element | $1$ |
| `MULTIPOP(k)` | Remove the top $\min(k, s)$ elements | $\min(k, s)$ |

The pseudocode for `MULTIPOP` is:

```
MULTIPOP(S, k):
    while S is not empty and k > 0:
        POP(S)
        k = k - 1
```

## Naive Worst-Case Analysis

A single `MULTIPOP(k)` on a stack of size $n$ costs $O(n)$. Over $n$ operations, the worst case appears to be $O(n) \times n = O(n^2)$. However, this analysis is too pessimistic because it ignores the constraint that each element must be pushed before it can be popped.

## Aggregate Analysis

The key observation is that each element is popped at most once for every time it is pushed. Over any sequence of $n$ operations starting from an empty stack:

- The total number of pushes is at most $n$.
- The total number of pops (across all `POP` and `MULTIPOP` calls combined) is at most $n$, because you cannot pop more elements than you have pushed.

Therefore the total cost is:

$$
T(n) \leq n + n = 2n
$$

The amortized cost per operation is:

$$
\hat{c} = \frac{T(n)}{n} \leq 2 = O(1)
$$

## Accounting Analysis

Assign the following amortized costs:

- `PUSH`: $\hat{c} = 2$ (1 for the push itself, 1 deposited as credit on the element)
- `POP`: $\hat{c} = 0$ (paid by the credit on the popped element)
- `MULTIPOP(k)`: $\hat{c} = 0$ (each of the $\min(k, s)$ popped elements pays for itself)

**Credit invariant:** Every element currently on the stack carries exactly 1 unit of credit, deposited when it was pushed. Since every pop (whether from `POP` or `MULTIPOP`) removes an element that was previously pushed, there is always sufficient credit. The total credit equals the stack size, which is always non-negative.

**Result:** The total amortized cost of $n$ operations is at most $2n$ (at most $n$ pushes, each costing $\hat{c} = 2$), giving $O(1)$ amortized per operation.

## Potential Analysis

Define the potential function as the stack size:

$$
\Phi(D) = |S|
$$

where $|S|$ is the number of elements on the stack. This satisfies $\Phi(D_0) = 0$ and $\Phi(D_i) \geq 0$ for all $i$.

**PUSH amortized cost:**

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = 1 + 1 = 2
$$

**POP amortized cost:**

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = 1 + (-1) = 0
$$

**MULTIPOP(k) amortized cost** (popping $k' = \min(k, s)$ elements):

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = k' + (-k') = 0
$$

The total amortized cost is:

$$
\sum_{i=1}^{n} \hat{c}_i \leq 2n = O(n)
$$

Since $\sum \hat{c}_i \geq \sum c_i$ (because $\Phi(D_n) \geq \Phi(D_0) = 0$), this confirms $O(1)$ amortized cost per operation.

## Python Example

```python
"""
Multipop stack amortized analysis demonstration.

Implements a stack with MULTIPOP and tracks actual costs,
verifying the O(1) amortized bound.
"""


# ===================================================================
# Multipop Stack Implementation
# ===================================================================
class MultipopStack:
    """Stack with push, pop, and multipop with cost tracking."""

    def __init__(self):
        self.items = []
        self.total_cost = 0
        self.num_ops = 0

    def push(self, x):
        """Push x onto the stack. Cost = 1."""
        self.items.append(x)
        self.total_cost += 1
        self.num_ops += 1
        return 1

    def pop(self):
        """Pop and return top element. Cost = 1."""
        if not self.items:
            raise IndexError("pop from empty stack")
        val = self.items.pop()
        self.total_cost += 1
        self.num_ops += 1
        return val

    def multipop(self, k):
        """Pop min(k, size) elements. Cost = number popped."""
        num_popped = min(k, len(self.items))
        for _ in range(num_popped):
            self.items.pop()
        self.total_cost += num_popped
        self.num_ops += 1
        return num_popped

    def size(self):
        """Return current stack size (also the potential)."""
        return len(self.items)

    def amortized_cost(self):
        """Return average cost per operation so far."""
        if self.num_ops == 0:
            return 0.0
        return self.total_cost / self.num_ops


# ===================================================================
# Demonstration: Worst Case Looks Bad, Amortized Is Fine
# ===================================================================
def demo_multipop():
    """Show that multipop cost is O(1) amortized."""
    stack = MultipopStack()

    # Push n elements, then multipop all at once
    n = 1000
    for i in range(n):
        stack.push(i)
    print(f"After {n} pushes: size={stack.size()}, "
          f"total_cost={stack.total_cost}, ops={stack.num_ops}")

    # One expensive multipop
    popped = stack.multipop(n)
    print(f"Multipop({n}): popped={popped}, "
          f"total_cost={stack.total_cost}, ops={stack.num_ops}")
    print(f"Amortized cost/op: {stack.amortized_cost():.4f}")
    print(f"Bound (2.0): {stack.amortized_cost() <= 2.0}")


# ===================================================================
# Demonstration: Mixed Operations
# ===================================================================
def demo_mixed():
    """Mixed push/pop/multipop operations."""
    stack = MultipopStack()
    operations = []

    # Simulate a sequence of mixed operations
    import random
    random.seed(42)
    for _ in range(500):
        r = random.random()
        if r < 0.6 or stack.size() == 0:
            stack.push(random.randint(1, 100))
            operations.append("PUSH")
        elif r < 0.8:
            stack.pop()
            operations.append("POP")
        else:
            k = random.randint(1, max(1, stack.size()))
            stack.multipop(k)
            operations.append(f"MULTIPOP")

    push_count = operations.count("PUSH")
    pop_count = operations.count("POP")
    mpop_count = sum(1 for op in operations if op == "MULTIPOP")

    print(f"\nMixed operations: {push_count} PUSH, "
          f"{pop_count} POP, {mpop_count} MULTIPOP")
    print(f"Total ops: {stack.num_ops}, total cost: {stack.total_cost}")
    print(f"Amortized cost/op: {stack.amortized_cost():.4f}")
    print(f"Bound (2.0): {stack.amortized_cost() <= 2.0}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("=== Worst-Case Multipop ===")
    demo_multipop()

    print("\n=== Mixed Operations ===")
    demo_mixed()
```

## Why This Example Matters

The multipop stack illustrates a pattern that appears throughout algorithm design: an operation that is occasionally expensive (like `MULTIPOP`) is offset by many cheap operations (like `PUSH`). The same pattern appears in dynamic arrays (occasional resize), hash table rehashing (occasional rebuild), and splay tree operations (occasional deep rotation sequence). Understanding the multipop stack provides the intuition needed for these more complex analyses.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.
