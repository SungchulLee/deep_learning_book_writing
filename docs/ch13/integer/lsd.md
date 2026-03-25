# LSD Radix Sort

The general radix sort page introduces the idea of sorting integers digit by digit.
This page examines the **Least Significant Digit (LSD)** variant in detail:
its correctness invariant, why stability is essential, and practical optimizations
that make LSD radix sort the most widely used non-comparison sort for fixed-width
integer keys.

## Core Idea

LSD radix sort processes digits from the *rightmost* (least significant) to the
*leftmost* (most significant).  At first glance this seems backwards -- we usually
compare numbers starting from the most significant digit.  The insight is that a
**stable** subroutine sort preserves the ordering established by previous passes,
so by the time we finish the last (most significant) digit, the array is fully sorted.

## Correctness Invariant

**Claim:** After the $i$-th pass of LSD radix sort (processing digit position $i$
counting from 1 at the least significant), the array is sorted with respect to the
$i$ least significant digits.

*Proof by induction.*

- **Base case ($i = 1$).** After sorting by the least significant digit, elements are
  trivially ordered by that single digit.
- **Inductive step.** Assume the array is sorted by the $i$ least significant digits.
  Pass $i + 1$ sorts by digit position $i + 1$ using a stable sort.  Consider two
  elements $a$ and $b$:
    - If digit $i+1$ of $a$ is less than digit $i+1$ of $b$, then $a$ precedes $b$
      after this pass.
    - If digit $i+1$ of $a$ equals digit $i+1$ of $b$, stability preserves their
      relative order, which by the inductive hypothesis is correct for the $i$ least
      significant digits.

  In both cases, $a$ and $b$ are correctly ordered with respect to the $i+1$ least
  significant digits. $\square$

## Why Stability is Essential

If the digit-level subroutine is **not** stable, the inductive step fails.  Equal
elements in pass $i+1$ may be reordered arbitrarily, destroying the ordering
established by passes $1$ through $i$.  Counting sort -- which is stable -- is the
standard choice for the subroutine.

## Complexity

Let $n$ be the number of elements, $d$ the number of digit positions, and $r$ the
radix (base).

$$
T(n, d, r) = \Theta\bigl(d\,(n + r)\bigr), \qquad S(n, r) = \Theta(n + r)
$$

Each of the $d$ passes runs a stable counting sort over a digit alphabet of size $r$,
costing $\Theta(n + r)$ per pass.

## Worked Example

Sort $A = [329, 457, 657, 839, 436, 720, 355]$ using LSD with radix $r = 10$.

**Pass 1 -- ones digit:**

| Element | Digit | Sorted |
|---------|-------|--------|
| 720 | 0 | 720 |
| 355 | 5 | 355 |
| 436 | 6 | 436 |
| 457 | 7 | 457 |
| 657 | 7 | 657 |
| 329 | 9 | 329 |
| 839 | 9 | 839 |

Result: $[720, 355, 436, 457, 657, 329, 839]$

**Pass 2 -- tens digit:**

| Element | Digit | Sorted |
|---------|-------|--------|
| 720 | 2 | 720 |
| 329 | 2 | 329 |
| 436 | 3 | 436 |
| 839 | 3 | 839 |
| 355 | 5 | 355 |
| 457 | 5 | 457 |
| 657 | 5 | 657 |

Result: $[720, 329, 436, 839, 355, 457, 657]$

Note: $720$ and $329$ both have tens digit 2.  Stability preserves their pass-1 order.

**Pass 3 -- hundreds digit:**

Result: $[329, 355, 436, 457, 657, 720, 839]$

## Byte-Level Optimization

For 32-bit integers, choosing radix $r = 256$ (one byte) reduces the number of passes
to exactly 4.  Each pass extracts one byte using bit shifts and masking:

$$
\text{digit}_i(x) = (x \gg 8i) \;\&\; \texttt{0xFF}
$$

This approach is cache-friendly because the count array has only 256 entries, and 4
passes suffice regardless of the magnitude of the values.

## Implementation

```python
"""
LSD radix sort -- processes digits from least to most significant.

Uses counting sort as the stable subroutine.
Time:  Theta(d * (n + r))
Space: Theta(n + r)
"""

# === Stable counting sort by digit ==========================================

def _counting_sort_by_digit(arr: list[int], exp: int, radix: int) -> list[int]:
    """Sort *arr* stably by the digit at position *exp*."""
    n = len(arr)
    output = [0] * n
    count = [0] * radix

    for x in arr:
        digit = (x // exp) % radix
        count[digit] += 1

    for i in range(1, radix):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % radix
        count[digit] -= 1
        output[count[digit]] = arr[i]

    return output


# === LSD radix sort =========================================================

def lsd_radix_sort(arr: list[int], radix: int = 10) -> list[int]:
    """Sort non-negative integers using LSD radix sort.

    Parameters
    ----------
    arr : list[int]
        Input array of non-negative integers.
    radix : int
        Base for digit extraction (default 10).

    Returns
    -------
    list[int]
        Sorted array.
    """
    if not arr:
        return arr

    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        arr = _counting_sort_by_digit(arr, exp, radix)
        exp *= radix

    return arr


# === Demo ===================================================================

if __name__ == "__main__":
    data = [329, 457, 657, 839, 436, 720, 355]
    sorted_data = lsd_radix_sort(data)
    print(f"Input:  {data}")
    print(f"Sorted: {sorted_data}")

    # Byte-level radix (r=256) for 32-bit integers
    large = [170, 45, 75, 90, 802, 24, 2, 66]
    sorted_large = lsd_radix_sort(large, radix=256)
    print(f"\nByte-level radix sort:")
    print(f"Input:  {large}")
    print(f"Sorted: {sorted_large}")
```

**Output:**
```
Input:  [329, 457, 657, 839, 436, 720, 355]
Sorted: [329, 355, 436, 457, 657, 720, 839]

Byte-level radix sort:
Input:  [170, 45, 75, 90, 802, 24, 2, 66]
Sorted: [2, 24, 45, 66, 75, 90, 170, 802]
```

## LSD vs MSD Summary

| Property | LSD | MSD |
|----------|-----|-----|
| Processing order | Right to left | Left to right |
| Stability required | Yes (essential) | No (per-bucket recursion) |
| Implementation | Iterative | Recursive |
| Early termination | No | Yes (single-element buckets) |
| Variable-length keys | Awkward | Natural |
| Fixed-width integers | Preferred | Less common |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
