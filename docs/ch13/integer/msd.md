# MSD Radix Sort

LSD radix sort processes digits from the least significant to the most significant and
requires a stable subroutine.  **Most Significant Digit (MSD) radix sort** works in the
opposite direction: it partitions elements by their most significant digit first, then
recursively sorts each bucket by the next digit.  This top-down approach mirrors how
humans sort alphanumerically -- first by the leading character, then by the second, and
so on.  MSD is the natural choice for variable-length strings and offers early
termination when buckets contain a single element.

## Algorithm Overview

Given $n$ elements with at most $d$ digit positions in base $r$:

1. **Partition** the array into $r$ buckets according to the current (most significant
   remaining) digit.
2. **Recurse** on each bucket that contains more than one element, moving to the next
   digit position.
3. **Base case:** a bucket with zero or one element is already sorted; a bucket that
   has exhausted all $d$ digits contains elements that are identical (up to $d$ digits).

The recursion tree has depth at most $d$, and each level processes every element
exactly once.

## Complexity

$$
T(n, d, r) = \Theta\bigl(d\,(n + r)\bigr), \qquad S(n, d, r) = \Theta(n + d \cdot r)
$$

The time complexity matches LSD radix sort.  The space includes $\Theta(n)$ for the
output array and $\Theta(d \cdot r)$ for the count arrays across all $d$ recursion
levels (or $\Theta(r)$ per level times depth $d$).

In practice, MSD can be faster than LSD when many buckets reduce to a single element
early, skipping further digit processing.

## Worked Example

Sort $A = [170, 045, 075, 090, 802, 024, 002, 066]$ using MSD with radix 10.
All numbers are zero-padded to 3 digits.

**Level 1 -- hundreds digit:**

| Digit | Bucket contents |
|-------|----------------|
| 0 | $045, 075, 090, 024, 002, 066$ |
| 1 | $170$ |
| 8 | $802$ |

Buckets with one element ($170$, $802$) are done.

**Level 2 -- tens digit of bucket 0:**

| Digit | Bucket contents |
|-------|----------------|
| 0 | $002$ |
| 2 | $024$ |
| 4 | $045$ |
| 6 | $066$ |
| 7 | $075$ |
| 9 | $090$ |

All sub-buckets have one element, so recursion stops.

**Final result:** $[002, 024, 045, 066, 075, 090, 170, 802]$.

## MSD-Specific Advantages

1. **Early termination.** Buckets with a single element need no further processing.
   For data with many distinct prefixes, this yields significant speedups.
2. **Variable-length keys.** MSD naturally handles strings of different lengths by
   treating shorter strings as having trailing "empty" characters that sort before
   any real character.
3. **In-place variants.** American Flag Sort is an in-place MSD radix sort that uses
   two passes per digit (count, then swap), avoiding the auxiliary output array.

## Handling Variable-Length Strings

For strings, the digit at position $i$ of string $s$ is:

- $s[i]$ if $i < \text{len}(s)$
- A sentinel value $-1$ (sorting before all real characters) if $i \ge \text{len}(s)$

This ensures that "cat" sorts before "cats" (the shorter string's sentinel sorts
before 's').

## Implementation

```python
"""
MSD radix sort -- processes digits from most to least significant.

Uses recursive bucket partitioning. Suitable for both integers and strings.
Time:  Theta(d * (n + r))
Space: Theta(n + d * r)
"""

# === MSD radix sort =========================================================

def msd_radix_sort(arr: list[int], radix: int = 10) -> list[int]:
    """Sort non-negative integers using MSD radix sort.

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
    # Find the highest digit position
    max_exp = 1
    while max_val // (max_exp * radix) > 0:
        max_exp *= radix

    result = list(arr)
    _msd_sort(result, 0, len(result) - 1, max_exp, radix)
    return result


def _msd_sort(
    arr: list[int], lo: int, hi: int, exp: int, radix: int
) -> None:
    """Recursively sort arr[lo..hi] by digit at position *exp*."""
    if lo >= hi or exp < 1:
        return

    # Count occurrences of each digit
    count = [0] * (radix + 1)
    for i in range(lo, hi + 1):
        digit = (arr[i] // exp) % radix
        count[digit + 1] += 1

    # Cumulative counts -- count[d] = start index for digit d
    for d in range(radix):
        count[d + 1] += count[d]

    # Distribute into auxiliary array
    aux = [0] * (hi - lo + 1)
    for i in range(lo, hi + 1):
        digit = (arr[i] // exp) % radix
        aux[count[digit]] = arr[i]
        count[digit] += 1

    # Copy back
    for i in range(lo, hi + 1):
        arr[i] = aux[i - lo]

    # Recurse on each bucket
    # Reset count for bucket boundaries
    count = [0] * (radix + 1)
    for i in range(lo, hi + 1):
        digit = (arr[i] // exp) % radix
        count[digit + 1] += 1
    for d in range(radix):
        count[d + 1] += count[d]

    for d in range(radix):
        bucket_lo = lo + count[d]
        bucket_hi = lo + count[d + 1] - 1
        if bucket_lo < bucket_hi:
            _msd_sort(arr, bucket_lo, bucket_hi, exp // radix, radix)


# === Demo ===================================================================

if __name__ == "__main__":
    data = [170, 45, 75, 90, 802, 24, 2, 66]
    sorted_data = msd_radix_sort(data)
    print(f"Input:  {data}")
    print(f"Sorted: {sorted_data}")

    data2 = [329, 457, 657, 839, 436, 720, 355]
    sorted_data2 = msd_radix_sort(data2)
    print(f"\nInput:  {data2}")
    print(f"Sorted: {sorted_data2}")
```

**Output:**
```
Input:  [170, 45, 75, 90, 802, 24, 2, 66]
Sorted: [2, 24, 45, 66, 75, 90, 170, 802]

Input:  [329, 457, 657, 839, 436, 720, 355]
Sorted: [329, 355, 436, 457, 657, 720, 839]
```

## MSD vs LSD Comparison

| Property | MSD | LSD |
|----------|-----|-----|
| Processing order | Left to right | Right to left |
| Strategy | Recursive (divide and conquer) | Iterative (multi-pass) |
| Stability | Not inherently stable | Stable (with stable subroutine) |
| Early termination | Yes | No |
| Variable-length keys | Natural | Requires padding |
| Cache behavior | Worse (scattered sub-buckets) | Better (sequential passes) |
| In-place variant | American Flag Sort | Uncommon |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
- Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.), Section 5.1. Addison-Wesley.
