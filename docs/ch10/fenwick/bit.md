# Binary Indexed Tree

Many applications require repeatedly computing prefix sums over an array that also receives point updates. A naive approach recomputes the prefix sum in $O(n)$ after each update, while a flat prefix-sum array makes updates $O(n)$ because every subsequent entry must be adjusted. The **Binary Indexed Tree** (BIT), also called a **Fenwick tree** after Peter Fenwick's 1994 paper, achieves $O(\log n)$ for both operations by exploiting the binary representation of indices.

## Core Insight -- the Lowest Set Bit

The entire structure of a BIT rests on a single bit-manipulation trick: for any positive integer $i$, the expression

$$
\text{lowbit}(i) = i \;\&\; (-i)
$$

isolates the lowest set bit of $i$. In two's-complement arithmetic, $-i$ equals the bitwise complement of $i$ plus one, so the AND operation zeroes out all bits except the rightmost `1`.

!!! example "Lowest Set Bit Examples"
    | $i$ (decimal) | $i$ (binary) | $i \;\&\; (-i)$ | Result |
    |:---:|:---:|:---:|:---:|
    | 6 | `110` | `010` | 2 |
    | 12 | `1100` | `0100` | 4 |
    | 7 | `0111` | `0001` | 1 |

## Tree Structure

A BIT is stored as a flat array `tree[1..n]` (1-indexed). Each position $i$ is responsible for a range of the original array:

$$
\texttt{tree}[i] = \sum_{j = i - \text{lowbit}(i) + 1}^{i} a[j]
$$

In other words, `tree[i]` stores the sum of exactly $\text{lowbit}(i)$ elements ending at position $i$. This means:

- `tree[1]` covers `a[1]` (1 element, since $\text{lowbit}(1)=1$).
- `tree[2]` covers `a[1..2]` (2 elements, since $\text{lowbit}(2)=2$).
- `tree[3]` covers `a[3]` (1 element).
- `tree[4]` covers `a[1..4]` (4 elements, since $\text{lowbit}(4)=4$).

This implicit tree has $O(\log n)$ levels because each index has at most $\lfloor \log_2 n \rfloor$ bits.

## Operations

### Prefix Query

To compute $\text{prefix}(i) = \sum_{j=1}^{i} a[j]$, accumulate `tree` values while removing the lowest set bit:

$$
\text{prefix}(i) = \texttt{tree}[i] + \texttt{tree}[i - \text{lowbit}(i)] + \cdots
$$

Each step decreases $i$ by at least one bit, so the loop runs at most $\lfloor \log_2 n \rfloor$ times.

### Point Update

To add a value $\delta$ to position $i$, update all tree nodes whose ranges include $i$ by **adding** the lowest set bit:

$$
i \;\leftarrow\; i + \text{lowbit}(i)
$$

This walks up the implicit tree until $i > n$.

## Implementation

```python
"""
Binary Indexed Tree (Fenwick Tree).

Provides O(log n) point updates and prefix sum queries on a
1-indexed array using the lowest-set-bit trick.
"""


# === Fenwick Tree Class ===

class FenwickTree:
    """Binary Indexed Tree supporting point updates and prefix queries."""

    def __init__(self, n: int):
        """Initialize a BIT of size n with all zeros."""
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    def update(self, i: int, delta: int) -> None:
        """Add delta to position i. Propagates to all ancestors."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # move to parent

    def query(self, i: int) -> int:
        """Return prefix sum from index 1 to i."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)  # strip lowest set bit
        return s

    def range_query(self, l: int, r: int) -> int:
        """Return sum from index l to r (inclusive)."""
        return self.query(r) - self.query(l - 1)

    def build(self, arr: list) -> None:
        """Build the BIT from a 0-indexed array in O(n) time."""
        for i, v in enumerate(arr, 1):
            self.update(i, v)


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    ft = FenwickTree(len(data))
    ft.build(data)

    print(f"Array: {data}")
    print(f"Prefix sum [1..3]: {ft.query(3)}")
    print(f"Range sum [2..5]:  {ft.range_query(2, 5)}")

    # Point update: add 10 to position 3
    ft.update(3, 10)
    print(f"After adding 10 to position 3:")
    print(f"Prefix sum [1..3]: {ft.query(3)}")
    print(f"Range sum [2..5]:  {ft.range_query(2, 5)}")
```

**Output:**
```
Array: [1, 3, 5, 7, 9]
Prefix sum [1..3]: 9
Range sum [2..5]:  24
After adding 10 to position 3:
Prefix sum [1..3]: 19
Range sum [2..5]:  34
```

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Build | $O(n \log n)$ | $O(n)$ |
| Point update | $O(\log n)$ | $O(1)$ |
| Prefix query | $O(\log n)$ | $O(1)$ |
| Range query | $O(\log n)$ | $O(1)$ |

The $O(n)$ build is also possible by propagating each element to its immediate parent in a single bottom-up pass.

!!! tip "When to Use a BIT"
    A BIT is ideal when you need **point updates** and **prefix sum queries** on a 1D array. It uses half the memory of a segment tree and has smaller constant factors. However, if you need range updates or non-invertible operations (like range minimum), a segment tree with lazy propagation is more appropriate.

## Reference

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
