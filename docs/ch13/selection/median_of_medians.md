# Median of Medians

Quickselect's $O(n^2)$ worst case stems from bad pivot choices. If the pivot always lands near an extreme rank, nearly all elements survive to the next recursive call. The **median of medians** is a pivot-selection technique that guarantees the chosen pivot has rank between $3n/10$ and $7n/10$. This ensures each recursive call eliminates at least $30\%$ of elements, yielding $O(n)$ worst-case selection. The technique was introduced by Blum, Floyd, Pratt, Rivest, and Tarjan in 1973.

## The Technique

The median-of-medians procedure selects a pivot from an array $A$ of $n$ elements:

1. **Group**: Divide $A$ into $\lceil n/5 \rceil$ groups of 5 elements each (the last group may be smaller).
2. **Sort each group**: Use insertion sort (at most 6 comparisons for 5 elements).
3. **Extract medians**: Take the median (middle element) of each sorted group. This produces $\lceil n/5 \rceil$ medians.
4. **Recurse**: Recursively apply the selection algorithm to find the median of these $\lceil n/5 \rceil$ medians. This is the pivot.

## Pivot Quality Guarantee

The pivot $m$ selected by median-of-medians is guaranteed to be a good splitter. Among the $\lceil n/5 \rceil$ group medians, at least half are $\leq m$. Each of those group medians has at least 2 elements below it in its group (since it is the median of 5). Therefore:

$$
\text{elements} \leq m \geq 3 \cdot \left\lceil \frac{1}{2} \cdot \lceil n/5 \rceil \right\rceil \geq \frac{3n}{10} - 6
$$

By symmetry, at least $3n/10 - 6$ elements are $\geq m$. The partition around $m$ produces a larger side with at most $7n/10 + 6$ elements.

## Recurrence

The total work satisfies:

$$
T(n) \leq T(\lceil n/5 \rceil) + T(7n/10 + 6) + O(n)
$$

- $T(\lceil n/5 \rceil)$: finding the median of the group medians recursively.
- $T(7n/10 + 6)$: the recursive selection on the larger partition.
- $O(n)$: grouping, sorting groups, and partitioning.

Since $n/5 + 7n/10 = 9n/10 < n$, we can prove by substitution that $T(n) \leq cn$ for a sufficiently large constant $c$, giving $T(n) = O(n)$.

!!! note "Substitution Proof Sketch"
    Assume $T(n) \leq cn$ for all smaller $n$. Then $T(n) \leq c \cdot n/5 + c(7n/10 + 6) + an = cn(9/10) + 6c + an = cn - cn/10 + 6c + an$. Choosing $c \geq 10a$ and $n \geq 60$ makes $T(n) \leq cn$.

## Why Groups of Five

The group size of 5 is the smallest odd number that makes the recurrence solve to $O(n)$:

| Group size $g$ | Fraction eliminated | Recursive fractions | Sum |
|---|---|---|---|
| 3 | $\geq n/4$ | $n/3 + 3n/4$ | $13/12 > 1$ |
| 5 | $\geq 3n/10$ | $n/5 + 7n/10$ | $9/10 < 1$ |
| 7 | $\geq 2n/7$ | $n/7 + 5n/7$ | $6/7 < 1$ |

Groups of 3 fail because the two recursive subproblems sum to more than $n$. Groups of 7 or larger work but increase the constant factor without improving the asymptotic bound.

## Implementation

```python
"""
Median-of-medians pivot selection for worst-case linear selection.

Divides the array into groups of 5, finds each group's median,
then recursively selects the median of those medians. This
guarantees a pivot that eliminates at least 30% of elements.
"""


# === Sort Small Groups ===

def sort5(arr: list, lo: int, hi: int) -> None:
    """Sort arr[lo..hi] using insertion sort (for groups of <= 5)."""
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# === Median of Medians ===

def median_of_medians(arr: list, lo: int, hi: int) -> int:
    """Select a pivot using median-of-medians. Returns pivot value."""
    n = hi - lo + 1
    if n <= 5:
        sort5(arr, lo, hi)
        return arr[lo + n // 2]

    # Find median of each group of 5
    num_groups = (n + 4) // 5
    for i in range(num_groups):
        group_lo = lo + i * 5
        group_hi = min(group_lo + 4, hi)
        sort5(arr, group_lo, group_hi)
        # Move group median to front of array
        median_idx = group_lo + (group_hi - group_lo) // 2
        arr[lo + i], arr[median_idx] = arr[median_idx], arr[lo + i]

    # Recursively find median of the group medians
    return median_of_medians(arr, lo, lo + num_groups - 1)


def select_mom(arr: list, k: int):
    """Find k-th smallest (0-indexed) in O(n) worst case."""
    data = arr.copy()
    return _select(data, 0, len(data) - 1, k)


def _select(arr: list, lo: int, hi: int, k: int):
    """Recursive selection with median-of-medians pivot."""
    if lo == hi:
        return arr[lo]

    pivot = median_of_medians(arr, lo, hi)

    # Three-way partition
    lt, gt = lo, hi
    i = lo
    while i <= gt:
        if arr[i] < pivot:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    if k < lt:
        return _select(arr, lo, lt - 1, k)
    elif k > gt:
        return _select(arr, gt + 1, hi, k)
    else:
        return arr[k]


# === Demonstration ===

if __name__ == "__main__":
    data = [31, 12, 5, 23, 7, 19, 42, 3, 15, 8, 27, 35, 1, 10, 20]
    print(f"Array:  {data}")
    print(f"Sorted: {sorted(data)}")
    print()

    for k in [0, 4, 7, 11, 14]:
        result = select_mom(data, k)
        expected = sorted(data)[k]
        status = "OK" if result == expected else "MISMATCH"
        print(f"k={k:2d}: got {result:3d}, expected {expected:3d} [{status}]")
```

**Output:**
```
Array:  [31, 12, 5, 23, 7, 19, 42, 3, 15, 8, 27, 35, 1, 10, 20]
Sorted: [1, 3, 5, 7, 8, 10, 12, 15, 19, 20, 23, 27, 31, 35, 42]

k= 0: got   1, expected   1 [OK]
k= 4: got   8, expected   8 [OK]
k= 7: got  15, expected  15 [OK]
k=11: got  27, expected  27 [OK]
k=14: got  42, expected  42 [OK]
```

## Reference

- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 9. MIT Press.
