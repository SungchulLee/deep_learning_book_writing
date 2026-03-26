# Parallel Merge Sort

Merge sort is a natural candidate for parallelization because its divide step splits the array into two independent halves. However, naively parallelizing only the recursive calls still leaves a sequential merge step with $O(n)$ span, limiting the overall parallelism. The key to achieving $O(\log^2 n)$ span is a **parallel merge** procedure that combines two sorted halves in $O(\log n)$ span using binary search.

## Sequential Baseline

Standard merge sort has work $O(n \log n)$ and span $O(n \log n)$ (fully sequential). Its recurrence:

$$
T_1(n) = 2 \cdot T_1(n/2) + O(n) = O(n \log n)
$$

The $O(n)$ merge step dominates the span in a sequential execution.

## Parallel Recursive Calls Only

The simplest parallelization forks the two recursive calls:

- **Work**: $T_1(n) = 2 \cdot T_1(n/2) + O(n) = O(n \log n)$.
- **Span**: $T_\infty(n) = T_\infty(n/2) + O(n) = O(n)$, because the merge is still sequential.
- **Parallelism**: $O(\log n)$ -- very low.

This approach barely improves over sequential execution.

## Parallel Merge via Binary Search

To reduce the span of the merge step, we use a divide-and-conquer merge. Given two sorted arrays $L$ and $R$:

1. Pick the median element $m$ of the larger array (say $L$).
2. Binary search for $m$ in $R$ to find its rank $j$.
3. All elements in $L[0 \ldots |L|/2 - 1]$ and $R[0 \ldots j - 1]$ belong before $m$; all others belong after.
4. Recursively merge the two "before" halves and the two "after" halves in parallel.

The merge recurrence becomes:

$$
M_\infty(n) = M_\infty(3n/4) + O(\log n) = O(\log^2 n)
$$

The $3n/4$ factor arises because the worst-case split divides the total elements into groups of at most $3n/4$.

## Full Parallel Merge Sort Analysis

With parallel merge, the overall recurrences become:

**Work**:

$$
T_1(n) = 2 \cdot T_1(n/2) + O(n) = O(n \log n)
$$

**Span**:

$$
T_\infty(n) = T_\infty(n/2) + O(\log^2 n) = O(\log^3 n)
$$

!!! note "Achieving O(log^2 n) span"
    With a more sophisticated parallel merge that achieves $O(\log n)$ span (using ranking), the overall span becomes $T_\infty(n) = T_\infty(n/2) + O(\log n) = O(\log^2 n)$. The binary-search-based merge described here gives $O(\log^3 n)$.

**Parallelism**: $P = O(n \log n / \log^3 n) = O(n / \log^2 n)$, which is substantial for large $n$.

## Implementation

```python
"""
Parallel merge sort with simulated parallel merge.

The parallel merge uses a divide-and-conquer approach based
on binary search. In a true parallel system, the recursive
calls in both merge_sort and merge would execute concurrently.
"""

from bisect import bisect_left

# ===================================================================
# Parallel Merge
# ===================================================================

def parallel_merge(left, right):
    """Merge two sorted arrays using divide-and-conquer.

    Uses binary search to split the merge into independent
    subproblems, enabling parallel execution.

    Args:
        left: sorted array
        right: sorted array

    Returns:
        Merged sorted array
    """
    if not left:
        return list(right)
    if not right:
        return list(left)
    if len(left) + len(right) <= 4:
        return _sequential_merge(left, right)

    # Ensure left is the larger array
    if len(left) < len(right):
        left, right = right, left

    mid = len(left) // 2
    pivot = left[mid]
    j = bisect_left(right, pivot)

    # Fork: merge two independent halves
    lower = parallel_merge(left[:mid], right[:j])
    upper = parallel_merge(left[mid + 1:], right[j:])

    return lower + [pivot] + upper

# ===================================================================
# Parallel Merge Sort
# ===================================================================

def parallel_merge_sort(arr):
    """Sort array using parallel merge sort.

    Args:
        arr: input array

    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return list(arr)

    mid = len(arr) // 2

    # Fork: sort two halves (parallel in real system)
    left = parallel_merge_sort(arr[:mid])
    right = parallel_merge_sort(arr[mid:])

    # Join: parallel merge
    return parallel_merge(left, right)


def _sequential_merge(left, right):
    """Standard sequential merge for small inputs."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import math

    arr = [38, 27, 43, 3, 9, 82, 10, 15, 72, 4]
    sorted_arr = parallel_merge_sort(arr)

    print(f"Input:  {arr}")
    print(f"Sorted: {sorted_arr}")
    print(f"Correct: {sorted_arr == sorted(arr)}")

    n = len(arr)
    work = n * math.ceil(math.log2(n))
    span = math.ceil(math.log2(n)) ** 2
    print(f"\nn = {n}")
    print(f"Work O(n log n) ~ {work}")
    print(f"Span O(log^2 n) ~ {span}")
    print(f"Parallelism ~ {work / span:.1f}")
```

**Output:**
```
Input:  [38, 27, 43, 3, 9, 82, 10, 15, 72, 4]
Sorted: [3, 4, 9, 10, 15, 27, 38, 43, 72, 82]
Correct: True

n = 10
Work O(n log n) ~ 40
Span O(log^2 n) ~ 16
Parallelism ~ 2.5
```

## Complexity Summary

| Variant | Work $T_1$ | Span $T_\infty$ | Parallelism |
|---|---|---|---|
| Sequential merge sort | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ |
| Parallel recursion only | $O(n \log n)$ | $O(n)$ | $O(\log n)$ |
| Parallel merge (binary search) | $O(n \log n)$ | $O(\log^3 n)$ | $O(n / \log^2 n)$ |
| Parallel merge (ranking) | $O(n \log n)$ | $O(\log^2 n)$ | $O(n / \log n)$ |

## Reference

- Cormen, T. H. et al. *Introduction to Algorithms*, Chapter 27 (Multithreaded Algorithms).
- Cole, R. (1988). "Parallel merge sort." *SIAM Journal on Computing*, 17(4), 770--785.
