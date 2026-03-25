# 2D Fenwick Trees

A one-dimensional BIT answers prefix sum queries over a linear array. Many applications — image processing, 2D frequency counting, grid-based games — require prefix sums over a two-dimensional matrix. A **2D Fenwick Tree** extends the 1D structure to handle point updates and rectangular prefix sum queries on an $n \times m$ grid, both in $O(\log n \cdot \log m)$ time.

## From 1D to 2D

In a 1D BIT, each index $i$ governs a range of $\text{lowbit}(i)$ consecutive elements. The 2D extension applies this idea to both the row and column dimensions independently. The 2D BIT is stored as a 2D array `tree[1..n][1..m]`, where:

$$
\texttt{tree}[i][j] = \sum_{\substack{i - \text{lowbit}(i) < r \leq i \\ j - \text{lowbit}(j) < c \leq j}} a[r][c]
$$

Each entry `tree[i][j]` accumulates the sum of a rectangular subregion whose row range is determined by $\text{lowbit}(i)$ and whose column range is determined by $\text{lowbit}(j)$.

## 2D Point Update

To add $\delta$ to position $(x, y)$ in the matrix, update all BIT entries whose ranges include $(x, y)$. This requires a nested loop: the outer loop advances the row index $i$ by adding $\text{lowbit}(i)$, and the inner loop advances the column index $j$ by adding $\text{lowbit}(j)$.

## 2D Prefix Query

The 2D prefix sum $\text{prefix}(x, y) = \sum_{r=1}^{x} \sum_{c=1}^{y} a[r][c]$ is computed by a nested loop that strips the lowest set bit in each dimension:

$$
\text{prefix}(x, y) = \sum \texttt{tree}[i][j]
$$

where $i$ iterates down from $x$ by removing $\text{lowbit}(i)$, and for each such $i$, $j$ iterates down from $y$ by removing $\text{lowbit}(j)$.

## 2D Range Sum via Inclusion-Exclusion

To compute the sum over a rectangle with corners $(r_1, c_1)$ and $(r_2, c_2)$, apply the **inclusion-exclusion** principle:

$$
\text{rangeSum}(r_1, c_1, r_2, c_2) = \text{prefix}(r_2, c_2) - \text{prefix}(r_1 - 1, c_2) - \text{prefix}(r_2, c_1 - 1) + \text{prefix}(r_1 - 1, c_1 - 1)
$$

This is the 2D analogue of the 1D formula $\text{rangeSum}(l, r) = \text{prefix}(r) - \text{prefix}(l-1)$.

!!! note "Inclusion-Exclusion Illustrated"
    To find the sum inside a rectangle, start with the prefix sum to the bottom-right corner, subtract the two regions that extend too far (left and above), then add back the overlap region that was subtracted twice.

## Implementation

```python
"""
2D Binary Indexed Tree (Fenwick Tree).

Supports point updates and rectangular range sum queries on a
2D grid using the inclusion-exclusion principle with nested
lowest-set-bit traversals.
"""


# === 2D Fenwick Tree ===

class FenwickTree2D:
    """Two-dimensional BIT for point updates and rectangle sum queries."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, x: int, y: int, delta: int) -> None:
        """Add delta to position (x, y). Both 1-indexed."""
        i = x
        while i <= self.rows:
            j = y
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix(self, x: int, y: int) -> int:
        """Return sum of all elements in the rectangle [1..x, 1..y]."""
        s = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                s += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return s

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Return sum of elements in the rectangle [r1..r2, c1..c2].

        Uses the inclusion-exclusion formula:
          prefix(r2,c2) - prefix(r1-1,c2) - prefix(r2,c1-1) + prefix(r1-1,c1-1)
        """
        return (self.prefix(r2, c2)
                - self.prefix(r1 - 1, c2)
                - self.prefix(r2, c1 - 1)
                + self.prefix(r1 - 1, c1 - 1))


# === Demonstration ===

if __name__ == "__main__":
    # 4x4 matrix
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]

    rows, cols = len(matrix), len(matrix[0])
    ft = FenwickTree2D(rows, cols)

    # Build the 2D BIT
    for i in range(rows):
        for j in range(cols):
            ft.update(i + 1, j + 1, matrix[i][j])

    print("Matrix:")
    for row in matrix:
        print(f"  {row}")
    print()

    # Prefix sum queries
    print(f"prefix(2, 3) = {ft.prefix(2, 3)}")  # sum of [1..2, 1..3]
    expected = sum(matrix[r][c] for r in range(2) for c in range(3))
    print(f"  Expected: {expected}")
    print()

    # Range sum queries
    print(f"rangeSum(2, 2, 3, 4) = {ft.range_sum(2, 2, 3, 4)}")
    expected = sum(matrix[r][c] for r in range(1, 3) for c in range(1, 4))
    print(f"  Expected: {expected}")
    print()

    print(f"rangeSum(1, 1, 4, 4) = {ft.range_sum(1, 1, 4, 4)}")
    expected = sum(matrix[r][c] for r in range(4) for c in range(4))
    print(f"  Expected: {expected}")
```

**Output:**
```
Matrix:
  [1, 2, 3, 4]
  [5, 6, 7, 8]
  [9, 10, 11, 12]
  [13, 14, 15, 16]

prefix(2, 3) = 24
  Expected: 24

rangeSum(2, 2, 3, 4) = 54
  Expected: 54

rangeSum(1, 1, 4, 4) = 136
  Expected: 136
```

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Build | $O(nm \log n \log m)$ | $O(nm)$ |
| Point update | $O(\log n \cdot \log m)$ | $O(1)$ |
| Prefix query | $O(\log n \cdot \log m)$ | $O(1)$ |
| Range sum query | $O(\log n \cdot \log m)$ | $O(1)$ |

The space overhead is the same as storing the matrix itself, since the BIT array has the same dimensions.

## Higher Dimensions

The same idea generalizes to $d$ dimensions. A $d$-dimensional BIT supports point updates and prefix queries in $O(\log^d n)$ time with $O(n^d)$ space. However, the constant factors grow rapidly, so in practice 2D and occasionally 3D are the most common variants.

## Reference

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
- Mishra, S. (2013). 2D Binary Indexed Trees. *TopCoder Tutorials*.
