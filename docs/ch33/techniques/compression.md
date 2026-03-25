# Coordinate Compression

Coordinate compression replaces raw values with their ranks, mapping a potentially
huge value range down to $\{0, 1, \dots, m{-}1\}$ where $m$ is the number of
distinct values. This makes array-indexed data structures feasible even when the
original values span billions.

## Intuition

Suppose you have $n$ points with $x$-coordinates in $[0, 10^9]$ and need to build
a Fenwick tree indexed by coordinate. Allocating $10^9$ cells is impractical, but
at most $n$ distinct coordinates matter. Sorting them and replacing each with its
rank yields indices in $[0, n{-}1]$ — a manageable range.

## Formal Definition

Given a multiset $S = \{a_1, a_2, \dots, a_n\}$ of values from some totally ordered
universe $U$, coordinate compression produces a mapping

$$
f \colon S \to \{0, 1, \dots, m - 1\}
$$

where $m = |\text{distinct}(S)|$, satisfying the **order-preserving** property:

$$
a_i < a_j \implies f(a_i) < f(a_j)
$$

Equal elements receive the same compressed value: $a_i = a_j \implies f(a_i) = f(a_j)$.

## Algorithm

1. **Collect** all values into a list.
2. **Sort and deduplicate** to get the sorted distinct values $v_0 < v_1 < \cdots < v_{m-1}$.
3. **Map** each original value $a_i$ to its index in the sorted distinct list (its rank).

The rank lookup can use binary search on the sorted list or a hash map built in
the deduplication step.

### Complexity

| Step | Time | Space |
|------|------|-------|
| Sort | $O(n \log n)$ | $O(n)$ |
| Deduplicate | $O(n)$ | $O(m)$ |
| Map (binary search) | $O(n \log m)$ | $O(1)$ extra |
| Map (hash map) | $O(n)$ expected | $O(m)$ |

**Overall**: $O(n \log n)$ time, $O(n)$ space.

## Worked Example

Given $S = [100, 30, 100, 50, 30]$:

| Step | Result |
|------|--------|
| Sorted distinct values | $[30, 50, 100]$ |
| Rank of $30$ | $0$ |
| Rank of $50$ | $1$ |
| Rank of $100$ | $2$ |
| Compressed $S$ | $[2, 0, 2, 1, 0]$ |

The value range shrinks from $[30, 100]$ to $[0, 2]$, enabling a size-3 array.

## Applications

- **Fenwick / segment trees on large domains**: Compress coordinates before
  indexing into $O(m)$-sized structures instead of $O(|U|)$.
- **Sweep-line algorithms**: Rectangle union area, segment intersection counting —
  compress $y$-coordinates to enable efficient range updates.
- **Counting inversions**: Compress values so a Fenwick tree of size $n$ suffices.
- **2-D problems**: Compress both axes independently to reduce the grid size.

## Implementation

```python
"""Coordinate compression utility."""

from bisect import bisect_left


# === Core compression ===
def compress(values):
    """Return compressed values and the sorted distinct list.

    Parameters
    ----------
    values : list[int]
        Raw values to compress.

    Returns
    -------
    compressed : list[int]
        Each entry is the rank of the corresponding input value.
    sorted_distinct : list[int]
        Sorted list of distinct values (used to decompress).
    """
    sorted_distinct = sorted(set(values))
    rank = {v: i for i, v in enumerate(sorted_distinct)}
    compressed = [rank[v] for v in values]
    return compressed, sorted_distinct


# === Binary-search variant (no hash map) ===
def compress_bisect(values):
    """Compress using binary search instead of a hash map."""
    sorted_distinct = sorted(set(values))
    compressed = [bisect_left(sorted_distinct, v) for v in values]
    return compressed, sorted_distinct


# === Decompression ===
def decompress(compressed, sorted_distinct):
    """Map compressed ranks back to original values."""
    return [sorted_distinct[c] for c in compressed]


# === Demo ===
if __name__ == "__main__":
    raw = [100, 30, 100, 50, 30]
    comp, mapping = compress(raw)
    print(f"Original:   {raw}")
    print(f"Compressed: {comp}")
    print(f"Mapping:    {mapping}")
    print(f"Recovered:  {decompress(comp, mapping)}")

    # Verify order preservation
    for i in range(len(raw)):
        for j in range(len(raw)):
            if raw[i] < raw[j]:
                assert comp[i] < comp[j], "Order not preserved!"
    print("Order-preserving property verified.")
```

## Common Pitfalls

!!! warning "Duplicates"
    Forgetting to deduplicate before ranking leads to gaps in the compressed range,
    wasting space in downstream structures.

!!! warning "Off-by-one"
    Some problems require 1-indexed ranks (e.g., Fenwick trees that skip index 0).
    Add 1 to each rank after compression.

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
