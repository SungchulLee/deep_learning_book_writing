# Natural Merge Sort

Standard merge sort ignores any pre-existing order in the input: it always divides the array at the midpoint, producing $\lceil \log_2 n \rceil$ levels of merging even when the data is nearly sorted.  **Natural merge sort** instead identifies maximal sorted subsequences -- called **runs** -- already present in the input and merges them directly.  On nearly sorted data this can reduce the number of merge passes dramatically, approaching $O(n)$ in the best case while retaining $O(n \log n)$ worst-case behavior.

## Key Concept: Runs

A **run** is a maximal contiguous non-decreasing subsequence of the input array.  For example, in the array $[3, 7, 8, 2, 5, 1, 4, 6]$:

- Run 1: $[3, 7, 8]$
- Run 2: $[2, 5]$
- Run 3: $[1, 4, 6]$

Natural merge sort detects these runs in a single $O(n)$ scan, then iteratively merges adjacent runs until only one run (the fully sorted array) remains.

## Algorithm

1. **Scan** the array left-to-right to identify all natural runs and record their boundaries.
2. **Merge** adjacent pairs of runs, replacing each pair with a single merged run.
3. **Repeat** until only one run remains.

Each merge pass reduces the number of runs by roughly half, so the number of passes is $\lceil \log_2 r \rceil$ where $r$ is the initial number of runs.

## Pseudocode

```
NATURAL-MERGE-SORT(A, n):
    repeat:
        runs = FIND-RUNS(A, n)
        if |runs| == 1:
            return                    // array is sorted
        for i = 0, 2, 4, ... < |runs|:
            if i + 1 < |runs|:
                MERGE(A, runs[i].start, runs[i].end,
                         runs[i+1].end)

FIND-RUNS(A, n):
    runs = []
    start = 0
    for i = 1 to n - 1:
        if A[i] < A[i - 1]:
            runs.append((start, i - 1))
            start = i
    runs.append((start, n - 1))
    return runs
```

## Complexity Analysis

**Best case.** The array is already sorted: one run is found, zero merges needed.

$$
T_{\text{best}}(n) = O(n)
$$

The $O(n)$ cost is just the initial scan to detect that the array is a single run.

**Worst case.** The array is sorted in reverse: every element starts a new run ($r = n$).  Merging $n$ singleton runs requires $\lceil \log_2 n \rceil$ passes of $O(n)$ work each:

$$
T_{\text{worst}}(n) = O(n \log n)
$$

**General case.** With $r$ initial runs:

$$
T(n) = O(n \log r)
$$

Since $1 \leq r \leq n$, this interpolates between $O(n)$ (sorted input) and $O(n \log n)$ (random input).

**Space complexity.** The merge procedure requires $O(n)$ auxiliary space, the same as standard merge sort.

## Comparison with Standard Merge Sort

| Property            | Standard merge sort | Natural merge sort    |
|---------------------|--------------------|-----------------------|
| Best-case time      | $O(n \log n)$      | $O(n)$               |
| Worst-case time     | $O(n \log n)$      | $O(n \log n)$        |
| Adaptive            | No                 | Yes                   |
| Extra scan overhead | None               | $O(n)$ per pass       |
| Run detection       | N/A                | $O(n)$ initial scan   |
| Space               | $O(n)$             | $O(n)$               |

!!! tip "Natural merge sort as a foundation for Timsort"
    Timsort, used in Python and Java, extends natural merge sort with several optimizations: minimum run lengths enforced via insertion sort, a merge stack with invariants that control merge order, and galloping mode to speed up merges with unequal-sized runs.  Understanding natural merge sort is essential background for studying Timsort.

## Step-by-Step Example

Sort $[3, 7, 8, 2, 5, 1, 4, 6]$:

**Initial scan** identifies 3 runs: $[3,7,8]$, $[2,5]$, $[1,4,6]$.

**Pass 1:** merge adjacent run pairs.
- Merge $[3,7,8]$ and $[2,5]$: result $[2,3,5,7,8]$.
- Run $[1,4,6]$ has no partner; it passes through unchanged.

Runs after pass 1: $[2,3,5,7,8]$, $[1,4,6]$.

**Pass 2:** merge the two remaining runs.
- Merge $[2,3,5,7,8]$ and $[1,4,6]$: result $[1,2,3,4,5,6,7,8]$.

Total merge passes: 2 (compared to $\lceil \log_2 8 \rceil = 3$ for standard merge sort).

## Python Implementation

```python
"""
Natural merge sort.

Identifies existing sorted runs in the input and merges them,
adapting to pre-existing order for better performance on nearly
sorted data.
"""


# === Find natural runs ========================================================

def find_runs(arr: list) -> list[tuple[int, int]]:
    """Identify maximal non-decreasing runs in arr.

    Returns a list of (start, end) index pairs for each run.
    """
    n = len(arr)
    if n == 0:
        return []
    runs = []
    start = 0
    for i in range(1, n):
        if arr[i] < arr[i - 1]:
            runs.append((start, i - 1))
            start = i
    runs.append((start, n - 1))
    return runs


# === Merge procedure ==========================================================

def merge(arr: list, left: int, mid: int, right: int) -> None:
    """Merge sorted subarrays arr[left..mid] and arr[mid+1..right]."""
    left_half = arr[left:mid + 1]
    right_half = arr[mid + 1:right + 1]
    i = j = 0
    k = left

    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1
    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1


# === Natural merge sort =======================================================

def natural_merge_sort(arr: list) -> None:
    """Sort arr in place using natural merge sort.

    Detects existing sorted runs and merges them, achieving O(n)
    on already-sorted input and O(n log n) in the worst case.
    """
    n = len(arr)
    if n <= 1:
        return

    while True:
        runs = find_runs(arr)
        if len(runs) == 1:
            return  # fully sorted
        # Merge adjacent pairs of runs
        i = 0
        while i + 1 < len(runs):
            left = runs[i][0]
            mid = runs[i][1]
            right = runs[i + 1][1]
            merge(arr, left, mid, right)
            i += 2


# === Main =====================================================================

if __name__ == "__main__":
    data = [3, 7, 8, 2, 5, 1, 4, 6]
    print(f"Before: {data}")
    runs = find_runs(data)
    print(f"Runs:   {[data[s:e+1] for s, e in runs]}")
    natural_merge_sort(data)
    print(f"After:  {data}")

    # Already sorted -- should detect 1 run
    sorted_data = [1, 2, 3, 4, 5]
    runs_sorted = find_runs(sorted_data)
    print(f"\nSorted input runs: {len(runs_sorted)} run(s)")
    natural_merge_sort(sorted_data)
    print(f"Result: {sorted_data}")
```

**Output:**
```
Before: [3, 7, 8, 2, 5, 1, 4, 6]
Runs:   [[3, 7, 8], [2, 5], [1, 4, 6]]
After:  [1, 2, 3, 4, 5, 6, 7, 8]

Sorted input runs: 1 run(s)
Result: [1, 2, 3, 4, 5]
```

## References

- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley, Section 5.2.4.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley, Section 2.2.
- Peters, T. (2002). Timsort description. CPython source: `Objects/listsort.txt`.
