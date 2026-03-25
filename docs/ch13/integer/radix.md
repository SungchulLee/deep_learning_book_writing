# Radix Sort

Counting sort runs in $\Theta(n + k)$ time, which is excellent when the range $k$ is
small.  But when keys span a large range -- say, 32-bit integers with $k = 2^{32} - 1$
-- allocating a count array of that size is prohibitive.  **Radix sort** solves this
problem by decomposing each key into $d$ digits in some base $r$ (the *radix*) and
sorting one digit at a time using a stable subroutine such as counting sort.  Because
each digit lies in $\{0, 1, \dots, r-1\}$, the per-pass cost stays manageable.

## How Radix Sort Works

Given $n$ integers, each with at most $d$ digits in base $r$:

1. For $i = 1$ to $d$ (least significant digit to most significant):
    - Sort the array by the $i$-th digit using a **stable** sort.

Stability is the key invariant.  After pass $i$, the array is sorted with respect to
the $i$ least significant digits.  The stable subroutine in pass $i+1$ preserves the
order among elements whose $(i+1)$-th digit is equal, so the invariant is maintained.

## Complexity

Let $n$ be the number of elements, $d$ the number of digits, and $r$ the radix (base).
Using counting sort as the stable subroutine, each pass takes $\Theta(n + r)$ time.

$$
T(n, d, r) = \Theta\bigl(d\,(n + r)\bigr)
$$

If the integers lie in $\{0, 1, \dots, k\}$, then $d = \lfloor \log_r k \rfloor + 1$.
Choosing $r = \Theta(n)$ yields:

$$
T = \Theta\!\left(\frac{\log k}{\log n} \cdot n\right)
$$

When $k = O(n^c)$ for a constant $c$, this simplifies to $\Theta(n)$.

**Space:** $\Theta(n + r)$ for the counting-sort auxiliary arrays.

## Worked Example

Sort $A = [170, 45, 75, 90, 802, 24, 2, 66]$ using radix 10 ($r = 10$).

| Pass | Digit | Sorted array |
|------|-------|-------------|
| 1 | Ones | $[170, 90, 802, 2, 24, 45, 75, 66]$ |
| 2 | Tens | $[802, 2, 24, 45, 66, 170, 75, 90]$ |
| 3 | Hundreds | $[2, 24, 45, 66, 75, 90, 170, 802]$ |

After the final pass, the array is fully sorted.  Notice how stability ensures that
after pass 2, the order $170 < 75$ (established by their ones digits in pass 1 where
$0 < 5$) is preserved among elements with equal tens digit.

## Choosing the Radix

The radix $r$ controls the trade-off between the number of passes $d$ and the per-pass
cost $\Theta(n + r)$.

| Choice | Passes $d$ | Per-pass cost | Total |
|--------|-----------|---------------|-------|
| $r = 2$ | $\log_2 k$ | $\Theta(n)$ | $\Theta(n \log k)$ |
| $r = 10$ | $\log_{10} k$ | $\Theta(n)$ | $\Theta(n \log_{10} k)$ |
| $r = n$ | $\log_n k$ | $\Theta(n)$ | $\Theta(n \log_n k)$ |
| $r = k$ | $1$ | $\Theta(n + k)$ | Reduces to counting sort |

In practice, $r = 256$ (byte-level) is popular because it fits cache lines well and
limits the number of passes to 4 for 32-bit integers.

## LSD vs MSD

Radix sort comes in two variants:

- **LSD (Least Significant Digit first):** processes digits from the rightmost to the
  leftmost.  Requires a stable subroutine.  Naturally iterative and the most common
  variant.
- **MSD (Most Significant Digit first):** processes digits from the leftmost to the
  rightmost.  Recursively sorts sub-buckets.  Can short-circuit when buckets have a
  single element.  Does not require overall stability.

The implementation below uses the LSD approach.

## Implementation

```python
"""
Radix sort -- LSD variant using counting sort as stable subroutine.

Sorts an array of non-negative integers.
Time:  Theta(d * (n + r))  where d = number of digits, r = radix
Space: Theta(n + r)
"""

# === Stable counting sort by a specific digit ===============================

def _counting_sort_by_digit(arr: list[int], exp: int, radix: int = 10) -> list[int]:
    """Sort *arr* by the digit at position *exp* using counting sort.

    The digit extracted from element x is (x // exp) % radix.
    """
    n = len(arr)
    output = [0] * n
    count = [0] * radix

    # Count occurrences of each digit
    for x in arr:
        digit = (x // exp) % radix
        count[digit] += 1

    # Cumulative counts
    for i in range(1, radix):
        count[i] += count[i - 1]

    # Place elements (right-to-left for stability)
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % radix
        count[digit] -= 1
        output[count[digit]] = arr[i]

    return output


# === Radix sort (LSD) =======================================================

def radix_sort(arr: list[int], radix: int = 10) -> list[int]:
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
    data = [170, 45, 75, 90, 802, 24, 2, 66]
    sorted_data = radix_sort(data)
    print(f"Input:  {data}")
    print(f"Sorted: {sorted_data}")
```

**Output:**
```
Input:  [170, 45, 75, 90, 802, 24, 2, 66]
Sorted: [2, 24, 45, 66, 75, 90, 170, 802]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
