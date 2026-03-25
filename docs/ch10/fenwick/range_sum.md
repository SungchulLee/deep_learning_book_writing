# Range Sum Queries

A prefix sum query returns the cumulative total from index 1 to $i$. In practice, we often need the sum over an arbitrary sub-range $[l, r]$. This page shows how a Binary Indexed Tree (BIT) answers **range sum queries** by combining two prefix queries, and explores the edge cases and extensions of this technique.

## Reduction to Prefix Sums

The sum of elements from index $l$ to index $r$ (inclusive) can be expressed as the difference of two prefix sums:

$$
\text{rangeSum}(l, r) = \text{prefix}(r) - \text{prefix}(l - 1)
$$

where $\text{prefix}(i) = \sum_{j=1}^{i} a[j]$ and $\text{prefix}(0) = 0$ by convention.

This identity holds because $\text{prefix}(r)$ includes all elements $a[1]$ through $a[r]$, and subtracting $\text{prefix}(l-1)$ removes the elements $a[1]$ through $a[l-1]$, leaving exactly $a[l] + a[l+1] + \cdots + a[r]$.

!!! warning "Boundary Case: l = 1"
    When $l = 1$, the formula evaluates $\text{prefix}(0) = 0$. The BIT query function must handle $i = 0$ correctly by returning 0. Since the while loop condition is `i > 0`, an input of $i = 0$ naturally returns 0 without entering the loop.

## Step-by-Step Trace

Consider the array $a = [1, 3, 5, 7, 9]$.

**Query: rangeSum(2, 4)**

$$
\text{rangeSum}(2, 4) = \text{prefix}(4) - \text{prefix}(1)
$$

- $\text{prefix}(4) = a[1]+a[2]+a[3]+a[4] = 1+3+5+7 = 16$
- $\text{prefix}(1) = a[1] = 1$
- $\text{rangeSum}(2, 4) = 16 - 1 = 15$

Each prefix query takes $O(\log n)$, so the range sum query takes $O(\log n)$ total (the constant factor is 2, but both queries share the same BIT).

## Correctness Argument

The range sum formula relies on the **telescoping property** of prefix sums:

$$
\sum_{j=l}^{r} a[j] = \sum_{j=1}^{r} a[j] - \sum_{j=1}^{l-1} a[j]
$$

This identity holds for any values (positive, negative, or zero) and for any valid indices $1 \leq l \leq r \leq n$. The only requirement is that addition is associative and has an inverse (i.e., we work in an abelian group), which holds for integer and floating-point addition.

## Implementation

```python
"""
Range sum queries using a Binary Indexed Tree.

Demonstrates the reduction of range queries to two prefix
queries with boundary handling and comprehensive testing.
"""


# === Fenwick Tree with Range Queries ===

class FenwickTree:
    """BIT supporting point updates and range sum queries."""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """Add delta to position i."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """Return sum of elements from index 1 to i."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, l: int, r: int) -> int:
        """Return sum of elements from index l to r (inclusive).

        Uses the identity: sum(l, r) = prefix(r) - prefix(l - 1).
        When l = 1, prefix(0) naturally returns 0.
        """
        return self.prefix(r) - self.prefix(l - 1)


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9]
    n = len(data)
    ft = FenwickTree(n)
    for i, v in enumerate(data, 1):
        ft.update(i, v)

    print(f"Array: {data}")
    print()

    # Various range queries
    queries = [(1, 3), (2, 5), (2, 4), (1, 1), (1, 5), (3, 3)]
    for l, r in queries:
        result = ft.range_sum(l, r)
        expected = sum(data[l - 1:r])
        print(f"  rangeSum({l}, {r}) = prefix({r}) - prefix({l - 1}) "
              f"= {ft.prefix(r)} - {ft.prefix(l - 1)} = {result}  "
              f"{'OK' if result == expected else 'MISMATCH'}")

    # After an update
    print()
    print("After adding 10 to position 3:")
    ft.update(3, 10)
    data[2] += 10  # mirror in the array
    for l, r in [(1, 3), (2, 5), (3, 3)]:
        result = ft.range_sum(l, r)
        expected = sum(data[l - 1:r])
        print(f"  rangeSum({l}, {r}) = {result}  "
              f"{'OK' if result == expected else 'MISMATCH'}")
```

**Output:**
```
Array: [1, 3, 5, 7, 9]

  rangeSum(1, 3) = prefix(3) - prefix(0) = 9 - 0 = 9  OK
  rangeSum(2, 5) = prefix(5) - prefix(1) = 25 - 1 = 24  OK
  rangeSum(2, 4) = prefix(4) - prefix(1) = 16 - 1 = 15  OK
  rangeSum(1, 1) = prefix(1) - prefix(0) = 1 - 0 = 1  OK
  rangeSum(1, 5) = prefix(5) - prefix(0) = 25 - 0 = 25  OK
  rangeSum(3, 3) = prefix(3) - prefix(2) = 9 - 4 = 5  OK

After adding 10 to position 3:
  rangeSum(1, 3) = 19  OK
  rangeSum(2, 5) = 34  OK
  rangeSum(3, 3) = 15  OK
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Range sum query | $O(\log n)$ | $O(1)$ |

The range sum performs two prefix queries, each $O(\log n)$. Since $O(\log n) + O(\log n) = O(\log n)$, the overall complexity remains logarithmic.

## Extensions

!!! tip "Point Value Retrieval"
    To retrieve a single element $a[i]$, compute $\text{rangeSum}(i, i) = \text{prefix}(i) - \text{prefix}(i-1)$. This takes $O(\log n)$ — the same as a range sum query. For $O(1)$ retrieval, maintain a separate copy of the original array alongside the BIT.

## Reference

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
