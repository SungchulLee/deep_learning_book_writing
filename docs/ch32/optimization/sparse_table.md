# Sparse Table

A **sparse table** is a data structure for answering **range minimum/maximum queries** (RMQ) on a static array in $O(1)$ time after $O(n \log n)$ preprocessing.

## Key Idea

Precompute answers for every range of length $2^j$:

$$\text{table}[i][j] = \min(a[i], a[i+1], \ldots, a[i + 2^j - 1])$$

For a query $[l, r]$, find the largest $k$ such that $2^k \le r - l + 1$, then:

$$\text{RMQ}(l, r) = \min(\text{table}[l][k],\ \text{table}[r - 2^k + 1][k])$$

This works because min/max are **idempotent** -- overlapping ranges give the same answer.

## Implementation

```python
import math

class SparseTable:
    def __init__(self, arr):
        n = len(arr)
        self.LOG = max(1, int(math.log2(n)) + 1)
        self.table = [[0] * n for _ in range(self.LOG)]
        self.table[0] = arr[:]

        for j in range(1, self.LOG):
            for i in range(n - (1 << j) + 1):
                self.table[j][i] = min(
                    self.table[j-1][i],
                    self.table[j-1][i + (1 << (j-1))]
                )

    def query(self, l, r):
        length = r - l + 1
        k = int(math.log2(length))
        return min(self.table[k][l], self.table[k][r - (1 << k) + 1])

arr = [1, 3, 2, 7, 9, 11, 3, 5, 2, 6]
st = SparseTable(arr)
print(st.query(0, 4))  # Output: 1
print(st.query(2, 7))  # Output: 2
print(st.query(5, 9))  # Output: 2
```

## Comparison with Other Range Query Structures

| Structure | Build | Query | Update | Use Case |
|---|---|---|---|---|
| Prefix Sums | $O(n)$ | $O(1)$ | Rebuild $O(n)$ | Range sum (static) |
| Sparse Table | $O(n \log n)$ | $O(1)$ | Rebuild $O(n \log n)$ | Range min/max (static) |
| Segment Tree | $O(n)$ | $O(\log n)$ | $O(\log n)$ | Any associative op (dynamic) |
| BIT/Fenwick | $O(n)$ | $O(\log n)$ | $O(\log n)$ | Range sum (dynamic) |

Use a sparse table when the array is **static** (no updates) and you need **idempotent** range queries (min, max, GCD).

# Reference

- [Sparse Table -- CP-Algorithms](https://cp-algorithms.com/data_structures/sparse-table.html)
- Bender, M. & Farach-Colton, M. "The LCA Problem Revisited", 2000.
