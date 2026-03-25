# Bucket Sort

Counting sort and radix sort exploit integer structure to beat the $\Omega(n \log n)$
comparison-based lower bound.  **Bucket sort** takes a different approach: it assumes
the input is drawn from a *uniform distribution* over a known interval (typically
$[0, 1)$) and distributes elements into equally spaced buckets.  If the distribution
is roughly uniform, each bucket contains $O(1)$ elements on average, and sorting every
bucket with insertion sort takes linear total time.

## Algorithm Overview

Given $n$ elements uniformly distributed over $[0, 1)$:

1. **Create** $n$ empty buckets $B[0], B[1], \dots, B[n-1]$.
2. **Distribute.** For each element $x$, insert it into bucket $B[\lfloor n \cdot x \rfloor]$.
3. **Sort** each bucket individually (insertion sort works well since buckets are small).
4. **Concatenate** all buckets in order to produce the sorted output.

## Complexity

### Expected time (uniform input)

When the $n$ input values are drawn independently and uniformly from $[0, 1)$, the
expected number of elements per bucket is 1.  Let $n_i$ denote the number of elements
in bucket $i$.  The expected cost of sorting all buckets with insertion sort is:

$$
E\!\left[\sum_{i=0}^{n-1} O(n_i^2)\right] = \sum_{i=0}^{n-1} O\!\left(E[n_i^2]\right)
$$

Each $n_i$ follows a Binomial$(n, 1/n)$ distribution, so $E[n_i] = 1$ and
$E[n_i^2] = \text{Var}(n_i) + (E[n_i])^2 = (1 - 1/n) + 1 = 2 - 1/n$.

$$
\text{Expected total cost} = \sum_{i=0}^{n-1} O(2 - 1/n) = O(n)
$$

Adding $\Theta(n)$ for distribution and concatenation:

$$
T(n) = \Theta(n) \quad \text{expected, for uniform input}
$$

### Worst case

If all elements land in a single bucket, the algorithm degenerates to the subroutine
sort.  With insertion sort this gives $O(n^2)$ worst case.

### Space

$$
S(n) = \Theta(n)
$$

The $n$ buckets collectively store $n$ elements plus the bucket structure overhead.

## Worked Example

Sort $A = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]$ with
$n = 10$ buckets.

| Bucket | Elements | After sort |
|--------|----------|------------|
| $B[0]$ | -- | -- |
| $B[1]$ | $0.17, 0.12$ | $0.12, 0.17$ |
| $B[2]$ | $0.26, 0.21, 0.23$ | $0.21, 0.23, 0.26$ |
| $B[3]$ | $0.39$ | $0.39$ |
| $B[4]$ -- $B[5]$ | -- | -- |
| $B[6]$ | $0.68$ | $0.68$ |
| $B[7]$ | $0.78, 0.72$ | $0.72, 0.78$ |
| $B[8]$ | -- | -- |
| $B[9]$ | $0.94$ | $0.94$ |

Concatenating: $[0.12, 0.17, 0.21, 0.23, 0.26, 0.39, 0.68, 0.72, 0.78, 0.94]$.

## Generalizing to Arbitrary Ranges

For inputs in $[a, b)$ rather than $[0, 1)$, map each element $x$ to bucket index
$\lfloor n \cdot (x - a) / (b - a) \rfloor$, clamping to $n - 1$ if $x = b$.

For integer inputs in $\{0, 1, \dots, k\}$, use $\lfloor n \cdot x / (k + 1) \rfloor$.
When $k$ is small enough that each bucket holds at most one distinct value, bucket sort
reduces to counting sort.

## Implementation

```python
"""
Bucket sort -- expected linear time for uniformly distributed input.

Sorts floating-point values in [0, 1).
Time:  Theta(n) expected (uniform), O(n^2) worst case
Space: Theta(n)
"""

# === Insertion sort for small buckets =======================================

def _insertion_sort(arr: list[float]) -> None:
    """Sort *arr* in place using insertion sort."""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# === Bucket sort ============================================================

def bucket_sort(arr: list[float], num_buckets: int | None = None) -> list[float]:
    """Sort values in [0, 1) using bucket sort.

    Parameters
    ----------
    arr : list[float]
        Input array with values in [0, 1).
    num_buckets : int or None
        Number of buckets (defaults to len(arr)).

    Returns
    -------
    list[float]
        Sorted array.
    """
    n = len(arr)
    if n <= 1:
        return list(arr)

    if num_buckets is None:
        num_buckets = n

    # Create empty buckets
    buckets: list[list[float]] = [[] for _ in range(num_buckets)]

    # Distribute elements into buckets
    for x in arr:
        idx = int(num_buckets * x)
        if idx == num_buckets:      # handle x == 1.0 edge case
            idx = num_buckets - 1
        buckets[idx].append(x)

    # Sort each bucket
    for bucket in buckets:
        _insertion_sort(bucket)

    # Concatenate
    result: list[float] = []
    for bucket in buckets:
        result.extend(bucket)

    return result


# === Demo ===================================================================

if __name__ == "__main__":
    data = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]
    sorted_data = bucket_sort(data)
    print(f"Input:  {data}")
    print(f"Sorted: {sorted_data}")
```

**Output:**
```
Input:  [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]
Sorted: [0.12, 0.17, 0.21, 0.23, 0.26, 0.39, 0.68, 0.72, 0.78, 0.94]
```

## When to Use Bucket Sort

| Scenario | Recommendation |
|----------|---------------|
| Uniform or near-uniform distribution | Ideal -- expected $\Theta(n)$ |
| Highly skewed distribution | Avoid -- many elements in few buckets |
| Floating-point keys in known range | Natural fit |
| Integer keys, small range | Counting sort is simpler |
| Stability required | Stable if the per-bucket sort is stable |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
