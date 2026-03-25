# Dynamic Array Example

Dynamic arrays (Python's `list`, C++'s `std::vector`, Java's `ArrayList`) support constant-time access by index and append operations that are fast *on average*, even though individual appends occasionally trigger an expensive resize. This page analyzes the amortized cost of appending to a dynamic array using all three amortized analysis methods: aggregate, accounting, and potential.

## The Doubling Strategy

A dynamic array maintains an internal buffer of some capacity $C$. When an append would exceed the capacity, the array allocates a new buffer of size $2C$, copies all existing elements, and then inserts the new element.

**Actual cost of the $i$-th append:**

$$
c_i = \begin{cases} i & \text{if } i - 1 \text{ is an exact power of 2 (resize occurs)} \\ 1 & \text{otherwise} \end{cases}
$$

Here $i$ counts from 1 and we assume the initial capacity is 1. The cost includes $i - 1$ for copying plus 1 for inserting, giving $i$ total when a resize happens.

## Aggregate Analysis

The total cost of $n$ appends is the sum of all cheap appends plus all resizing costs. Resizes occur at appends $1, 2, 3, 5, 9, 17, \ldots$ (when the size equals a power of 2 plus 1, triggering a copy of $2^0, 2^1, 2^2, \ldots$ elements):

$$
T(n) = n + \sum_{j=0}^{\lfloor \log_2(n-1) \rfloor} 2^j < n + 2n = 3n
$$

The first $n$ accounts for the 1-unit cost of each insertion, and the sum accounts for copying during resizes. The geometric sum is bounded by $2n$.

The amortized cost per append is:

$$
\hat{c} = \frac{T(n)}{n} < 3 = O(1)
$$

## Accounting Analysis

Assign an amortized cost of $\hat{c} = 3$ to every append:

- **1 unit** pays for inserting the new element.
- **1 unit** is stored as credit on the newly inserted element.
- **1 unit** is stored as credit on one element in the first half of the array that does not yet have credit.

When a resize occurs at size $s$, each of the $s$ elements in the old array carries 1 unit of credit, providing exactly enough to pay for copying them into the new array.

**Credit invariant:** After each append (without a resize), the number of elements with credit equals $s - C/2$, where $s$ is the current size and $C$ is the current capacity. At the moment of resize ($s = C$), every element has credit, so the $s$ units of credit pay for the $s$ copies. After the resize, all credit is consumed and the new capacity is $2C$, restarting the cycle.

## Potential Analysis

Define the potential function:

$$
\Phi(D) = 2s - C
$$

where $s$ is the current number of elements and $C$ is the current capacity. This potential satisfies:

- $\Phi(D_0) = 2(0) - 1 = -1$, but after the first insert $\Phi(D_1) = 2(1) - 1 = 1 \geq 0$.
- For a non-resizing append: $s$ increases by 1 and $C$ stays the same, so $\Delta\Phi = 2$.
- Just before a resize: $s = C$, so $\Phi = 2C - C = C \geq 0$.

**Amortized cost (no resize):**

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1}) = 1 + 2 = 3
$$

**Amortized cost (resize at size $s$, old capacity $C = s$):**

After the resize, the new capacity is $2s$ and the size becomes $s + 1$:

$$
\hat{c}_i = (s + 1) + \Phi(D_i) - \Phi(D_{i-1}) = (s + 1) + [2(s+1) - 2s] - [2s - s] = (s+1) + 2 - s = 3
$$

In both cases, the amortized cost is exactly 3.

!!! note "Why Doubling?"
    The factor of 2 in the doubling strategy is not special. Any multiplicative growth factor $\alpha > 1$ yields $O(1)$ amortized append. The amortized cost becomes $\frac{\alpha}{\alpha - 1}$ per append. Doubling ($\alpha = 2$) gives amortized cost 3 and wastes at most 50% of allocated memory. A factor of $\alpha = 1.5$ gives amortized cost 5 but wastes at most 33%. Python's `list` uses a growth factor of approximately 1.125 to reduce memory overhead.

## Table Contraction

If the array also supports deletions, the strategy must handle shrinking. A natural approach is to halve the capacity when the array is half empty ($s = C/4$), not when $s = C/2$. Halving at $s = C/2$ causes pathological behavior: alternating inserts and deletes near the threshold trigger repeated resizes.

With the $C/4$ shrinking threshold, the amortized cost of both insert and delete remains $O(1)$. The potential function for this combined analysis is:

$$
\Phi(D) = \begin{cases} 2s - C & \text{if } s \geq C/2 \\ C/2 - s & \text{if } s < C/2 \end{cases}
$$

## Python Example

```python
"""
Dynamic array amortized analysis demonstration.

Implements a dynamic array with doubling and tracks actual vs amortized
costs, verifying the O(1) amortized bound.
"""


# ===================================================================
# Dynamic Array with Cost Tracking
# ===================================================================
class DynamicArray:
    """Dynamic array that doubles capacity on overflow."""

    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.data = [None] * self.capacity
        self.total_cost = 0
        self.resize_count = 0

    def append(self, value):
        """Append value, doubling capacity if needed. Returns actual cost."""
        cost = 1  # insertion cost
        if self.size == self.capacity:
            # Resize: copy all elements
            cost += self.size
            new_data = [None] * (2 * self.capacity)
            for i in range(self.size):
                new_data[i] = self.data[i]
            self.data = new_data
            self.capacity *= 2
            self.resize_count += 1

        self.data[self.size] = value
        self.size += 1
        self.total_cost += cost
        return cost

    def potential(self):
        """Compute the potential function Phi = 2*size - capacity."""
        return 2 * self.size - self.capacity


# ===================================================================
# Verification
# ===================================================================
def verify_amortized_cost(n):
    """Verify that total cost < 3n for n appends."""
    arr = DynamicArray()
    for i in range(n):
        arr.append(i)
    ratio = arr.total_cost / n
    bound_holds = arr.total_cost < 3 * n
    print(f"n={n:>6}: total_cost={arr.total_cost:>8}, "
          f"ratio={ratio:.4f}, resizes={arr.resize_count}, "
          f"<3n: {bound_holds}")
    return bound_holds


# ===================================================================
# Step-by-Step Trace
# ===================================================================
def trace_appends(n):
    """Print per-operation details for first n appends."""
    arr = DynamicArray()
    print(f"\n{'Op':>3} {'Size':>5} {'Cap':>5} {'Cost':>5} "
          f"{'Total':>6} {'Phi':>5} {'Amort':>6}")
    print("-" * 40)
    for i in range(1, n + 1):
        cost = arr.append(i)
        phi = arr.potential()
        amortized = arr.total_cost / i
        print(f"{i:>3} {arr.size:>5} {arr.capacity:>5} {cost:>5} "
              f"{arr.total_cost:>6} {phi:>5} {amortized:>6.2f}")


# ===================================================================
# Demonstration
# ===================================================================
if __name__ == "__main__":
    print("=== Aggregate Analysis Verification ===")
    for n in [100, 1000, 10000, 100000]:
        verify_amortized_cost(n)

    print("\n=== Step-by-Step Trace (first 17 appends) ===")
    trace_appends(17)
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16: Amortized Analysis. MIT Press.
