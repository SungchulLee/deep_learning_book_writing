# Aggregate Method

When analyzing algorithms, worst-case analysis of individual operations can be overly pessimistic. A single expensive operation does not mean every operation is expensive. The aggregate method is the simplest form of amortized analysis: it computes the total cost of an entire sequence of $n$ operations and divides by $n$ to obtain a per-operation average. This average, called the amortized cost, provides a tighter bound on performance than naive worst-case analysis.

## Definition

Given a sequence of $n$ operations with individual actual costs $c_1, c_2, \ldots, c_n$, the **amortized cost per operation** under the aggregate method is:

$$
\hat{c} = \frac{1}{n} \sum_{i=1}^{n} c_i
$$

The aggregate method assigns the same amortized cost $\hat{c}$ to every operation, regardless of whether a particular operation is cheap or expensive. This distinguishes it from the accounting and potential methods, which can assign different amortized costs to different operations.

## When to Use It

The aggregate method works best when:

- All operations are of the same type (so a uniform cost makes sense).
- The total cost over $n$ operations has a clean closed-form expression.
- You need a quick upper bound without tracking per-element credit or defining a potential function.

Its main limitation is the uniform assignment: if different operation types have genuinely different average costs, the accounting or potential methods provide a more precise analysis.

## Example: Stack with Multipop

Consider a stack that supports `PUSH`, `POP`, and `MULTIPOP(k)`. The `MULTIPOP(k)` operation pops $\min(k, s)$ elements from a stack of size $s$, costing $\min(k, s)$ time.

**Naive worst-case analysis:** A single `MULTIPOP` can cost $O(n)$, so $n$ operations could cost $O(n^2)$.

**Aggregate analysis:** Observe that each element can be popped at most once for each time it is pushed. Over $n$ operations, the total number of pushes is at most $n$, so the total number of pops (across all `POP` and `MULTIPOP` calls) is also at most $n$. Therefore:

$$
\sum_{i=1}^{n} c_i \leq 2n
$$

The amortized cost per operation is:

$$
\hat{c} = \frac{2n}{n} = 2 = O(1)
$$

Even though a single `MULTIPOP` can cost $O(n)$, the amortized cost per operation is $O(1)$.

## Example: Binary Counter

Consider a $k$-bit binary counter that supports only the `INCREMENT` operation. Each increment flips some bits from 0 to 1 and from 1 to 0.

**Naive worst-case analysis:** A single increment can flip up to $k$ bits (e.g., incrementing $0111\ldots1$ to $1000\ldots0$), so $n$ increments could cost $O(nk)$.

**Aggregate analysis:** Count how many times each bit position flips over $n$ increments:

- Bit 0 (least significant) flips every increment: $n$ times
- Bit 1 flips every 2nd increment: $\lfloor n/2 \rfloor$ times
- Bit 2 flips every 4th increment: $\lfloor n/4 \rfloor$ times
- Bit $j$ flips every $2^j$-th increment: $\lfloor n / 2^j \rfloor$ times

The total number of bit flips is:

$$
\sum_{j=0}^{k-1} \left\lfloor \frac{n}{2^j} \right\rfloor < n \sum_{j=0}^{\infty} \frac{1}{2^j} = 2n
$$

The amortized cost per increment is:

$$
\hat{c} = \frac{2n}{n} = 2 = O(1)
$$

## Python Example

```python
"""
Aggregate method demonstration for a binary counter.

Counts the total number of bit flips across n increments
and verifies that the amortized cost per increment is O(1).
"""


# ===================================================================
# Binary Counter with Cost Tracking
# ===================================================================
class BinaryCounter:
    """Binary counter that tracks total bit-flip cost."""

    def __init__(self, num_bits):
        self.bits = [0] * num_bits
        self.num_bits = num_bits
        self.total_flips = 0

    def increment(self):
        """Increment the counter by 1, tracking bit flips."""
        flips = 0
        i = 0
        while i < self.num_bits and self.bits[i] == 1:
            self.bits[i] = 0  # flip 1 -> 0
            flips += 1
            i += 1
        if i < self.num_bits:
            self.bits[i] = 1  # flip 0 -> 1
            flips += 1
        self.total_flips += flips
        return flips

    def value(self):
        """Return the current counter value."""
        return sum(b * (2 ** i) for i, b in enumerate(self.bits))


# ===================================================================
# Demonstration
# ===================================================================
if __name__ == "__main__":
    k = 8  # number of bits
    n = 100  # number of increments
    counter = BinaryCounter(k)

    print(f"{'Step':>5} {'Value':>6} {'Flips':>6} {'Total':>6} {'Amortized':>10}")
    print("-" * 37)
    for step in range(1, n + 1):
        flips = counter.increment()
        amortized = counter.total_flips / step
        if step <= 16 or step == n:
            print(
                f"{step:>5} {counter.value():>6} {flips:>6} "
                f"{counter.total_flips:>6} {amortized:>10.2f}"
            )

    print(f"\nTotal increments:     {n}")
    print(f"Total bit flips:      {counter.total_flips}")
    print(f"Amortized cost/op:    {counter.total_flips / n:.2f}")
    print(f"Upper bound (2n):     {2 * n}")
```

## Comparison with Other Methods

The aggregate method is the simplest of the three amortized analysis techniques:

| Aspect | Aggregate | Accounting | Potential |
|--------|-----------|------------|-----------|
| Cost assignment | Uniform $\hat{c}$ for all operations | Different $\hat{c}_i$ per operation type | Derived from $\Phi$ changes |
| Main tool | Total cost formula | Per-element credit | Potential function |
| Flexibility | Low | Medium | High |
| Typical use case | Single operation type | Credit on elements | Complex state changes |

For problems where different operations need different amortized costs, the accounting method or potential method provides a more refined analysis.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.
