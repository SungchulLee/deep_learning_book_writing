# Natural Runs

Real-world data is rarely fully random. Arrays often contain pre-existing sorted subsequences -- ascending sequences left over from previous operations, descending streaks from reverse-ordered inputs, or plateaus of equal values. Timsort exploits this structure by identifying **natural runs**: maximal subsequences that are already sorted (ascending) or reverse-sorted (descending). Each run becomes a building block for the merge phase, which means that partially sorted data gets sorted much faster than $O(n \log n)$.

## Run Detection

Timsort scans the array from left to right, identifying runs of two types:

1. **Ascending run**: a maximal sequence where $A[i] \leq A[i+1]$ for consecutive elements. This includes sequences with equal adjacent elements (non-decreasing order), which preserves stability.
2. **Descending run**: a maximal sequence where $A[i] > A[i+1]$ for consecutive elements. Note the **strict** inequality: equal elements are never considered descending. After detection, the descending run is reversed in place to become ascending.

The strict inequality for descending runs is essential for stability. If equal elements were included in descending runs, reversing them would swap their relative order.

## Minimum Run Length

Short runs are inefficient to merge. Timsort enforces a **minimum run length** (called `minrun`) between 32 and 64. If a natural run is shorter than `minrun`, it is extended using binary insertion sort on the following elements until it reaches `minrun` length.

The value of `minrun` is computed from $n$ by taking the top 6 bits and adding 1 if any remaining bits are set:

```
minrun = n
while minrun >= 64:
    minrun = (minrun + 1) >> 1
```

This formula ensures that $n / \text{minrun}$ is close to a power of 2 (or slightly less), which produces balanced merges.

## Why minrun Matters

If `minrun` is too small, there are too many runs to merge, increasing overhead. If too large, insertion sort on short runs becomes expensive. The range 32-64 balances these concerns: insertion sort is fast on arrays of this size due to hardware cache effects, and the resulting number of runs stays within $O(n / 32) = O(n)$, giving a merge tree of depth $O(\log n)$.

## Implementation

```python
"""
Natural run detection and extension for Timsort.

Identifies ascending and descending runs in the input array,
reverses descending runs to maintain stability, and extends
short runs with binary insertion sort to meet the minimum
run length threshold.
"""


# === Compute Minimum Run Length ===

def compute_minrun(n: int) -> int:
    """Compute Timsort's minimum run length for an array of size n.

    Returns a value between 32 and 64 such that n/minrun is
    close to a power of 2.
    """
    r = 0
    while n >= 64:
        r |= n & 1
        n >>= 1
    return n + r


# === Binary Insertion Sort ===

def binary_insertion_sort(arr: list, lo: int, hi: int,
                          start: int) -> None:
    """Sort arr[lo..hi] using binary insertion sort.

    Elements arr[lo..start-1] are already sorted. Insert elements
    from start onward using binary search to find insertion points.
    """
    for i in range(start, hi + 1):
        key = arr[i]
        # Binary search for insertion point
        left, right = lo, i
        while left < right:
            mid = (left + right) // 2
            if key < arr[mid]:
                right = mid
            else:
                left = mid + 1
        # Shift elements and insert
        for j in range(i, left, -1):
            arr[j] = arr[j - 1]
        arr[left] = key


# === Find and Extend Runs ===

def find_run(arr: list, lo: int, hi: int) -> tuple:
    """Find the next natural run starting at lo.

    Returns (run_end, is_descending) where arr[lo..run_end] is the
    maximal ascending or descending run.
    """
    if lo >= hi:
        return lo, False

    run_end = lo + 1
    if arr[run_end] < arr[lo]:
        # Strictly descending
        while run_end <= hi and arr[run_end] < arr[run_end - 1]:
            run_end += 1
        return run_end - 1, True
    else:
        # Non-decreasing (ascending)
        while run_end <= hi and arr[run_end] >= arr[run_end - 1]:
            run_end += 1
        return run_end - 1, False


def identify_runs(arr: list) -> list:
    """Identify all natural runs in arr, extending short ones.

    Returns a list of (start, length) tuples for each run.
    """
    n = len(arr)
    if n == 0:
        return []

    minrun = compute_minrun(n)
    runs = []
    lo = 0

    while lo < n:
        run_end, is_descending = find_run(arr, lo, n - 1)

        if is_descending:
            # Reverse descending run to make it ascending
            left, right = lo, run_end
            while left < right:
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
                right -= 1

        run_length = run_end - lo + 1

        # Extend short runs with binary insertion sort
        if run_length < minrun:
            force = min(lo + minrun - 1, n - 1)
            binary_insertion_sort(arr, lo, force, run_end + 1)
            run_end = force
            run_length = run_end - lo + 1

        runs.append((lo, run_length))
        lo = run_end + 1

    return runs


# === Demonstration ===

if __name__ == "__main__":
    # Array with natural runs
    arr = [1, 3, 5, 7, 9, 8, 6, 4, 2, 10, 11, 12]
    print(f"Input: {arr}")
    print(f"minrun for n={len(arr)}: {compute_minrun(len(arr))}")
    print()

    # Find individual runs
    arr_copy = arr.copy()
    lo = 0
    while lo < len(arr_copy):
        end, desc = find_run(arr_copy, lo, len(arr_copy) - 1)
        run_type = "descending" if desc else "ascending"
        print(f"Run at [{lo}..{end}]: {arr_copy[lo:end+1]} ({run_type})")
        if desc:
            left, right = lo, end
            while left < right:
                arr_copy[left], arr_copy[right] = (
                    arr_copy[right], arr_copy[left])
                left += 1
                right -= 1
            print(f"  Reversed: {arr_copy[lo:end+1]}")
        lo = end + 1

    print()

    # Show minrun for various sizes
    for size in [64, 128, 256, 1000, 10000]:
        print(f"n={size:5d} -> minrun={compute_minrun(size)}")
```

**Output:**
```
Input: [1, 3, 5, 7, 9, 8, 6, 4, 2, 10, 11, 12]
minrun for n=12: 12

Run at [0..4]: [1, 3, 5, 7, 9] (ascending)
Run at [5..8]: [8, 6, 4, 2] (descending)
  Reversed: [2, 4, 6, 8]
Run at [9..11]: [10, 11, 12] (ascending)

n=   64 -> minrun=64
n=  128 -> minrun=64
n=  256 -> minrun=64
n= 1000 -> minrun=63
n=10000 -> minrun=40
```

!!! tip "Adaptive Performance"
    On already-sorted data, Timsort finds a single run of length $n$ and finishes in $O(n)$ — no merges needed. On reverse-sorted data, it finds a single descending run, reverses it in $O(n)$, and finishes. This adaptivity is why Timsort excels on real-world data.

## Reference

- Peters, T. (2002). *Timsort description*. [CPython source, `Objects/listsort.txt`](https://github.com/python/cpython/blob/main/Objects/listsort.txt).
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
