# Parallel Quicksort

Quicksort is naturally suited to parallelism: once the array is partitioned around a
pivot, the left and right sub-arrays are independent and can be sorted concurrently.
However, the partition step itself is sequential in the naive implementation, and
unbalanced partitions lead to poor load balancing.  **Parallel quicksort** addresses
both challenges, achieving $O(n \log n)$ total work with low parallel depth.

## Sources of Parallelism

Quicksort has two potential parallelism sites:

1. **Recursive sub-problems.** After partitioning, the sub-arrays on each side of the
   pivot are independent.  Sorting them in parallel is straightforward.
2. **The partition step itself.** The standard Lomuto or Hoare partition scans the
   array sequentially.  A parallel partition distributes the scan across processors.

## Naive Parallel Quicksort

The simplest parallel quicksort spawns a new task for each recursive call:

1. Choose a pivot and partition the array sequentially in $O(n)$.
2. Spawn two parallel tasks for the left and right sub-arrays.
3. Wait for both tasks to complete.

### Complexity (naive)

Let $W(n)$ and $S(n)$ denote work and span (critical path length).

$$
W(n) = O(n \log n) \quad \text{(same as sequential)}
$$

$$
S(n) = O(n) \quad \text{expected (dominated by partition)}
$$

The span is $O(n)$ because the partition step is sequential.  Even with perfect
$n/2$ splits, the first partition takes $\Theta(n)$ time.

## Parallel Partition

To reduce the span, we can parallelize the partition step using a **prefix sum**:

1. Divide the array into $p$ blocks, one per processor.
2. Each processor counts how many elements in its block are $\le$ pivot and $>$ pivot.
3. Compute a prefix sum over these counts to determine the final position of each
   block's elements.
4. Each processor moves its elements to their final positions.

**Partition span:** $O(n/p + \log p)$ with $p$ processors.

### Improved Complexity

With parallel partition and $p = n / \log n$ processors:

$$
W(n) = O(n \log n), \qquad S(n) = O(\log^2 n) \quad \text{expected}
$$

The span comes from $O(\log n)$ recursion levels, each with $O(\log n)$ partition span.

## Load Balancing Challenges

The quality of the pivot determines load balance.  With a poor pivot, one sub-array
may contain most of the elements, leaving processors idle.

**Strategies for better pivots:**

| Strategy | Description | Overhead |
|----------|-------------|----------|
| Random pivot | Choose uniformly at random | $O(1)$ |
| Median-of-three | Median of first, middle, last | $O(1)$ |
| Sampling | Random sample of $O(\sqrt{n})$ elements, take median | $O(\sqrt{n})$ |
| Exact median | Use median-of-medians | $O(n)$ -- defeats purpose |

In practice, random pivots provide expected $O(\log n)$ depth with high probability.

## Sample Sort (Parallel Generalization)

For $p$ processors, **sample sort** generalizes quicksort:

1. Each processor sorts a random sample of its local elements.
2. Select $p - 1$ **splitters** from the combined samples (evenly spaced).
3. Use the splitters to partition all elements into $p$ buckets.
4. Each processor sorts its bucket locally.

Sample sort achieves near-perfect load balance with high probability.

$$
W(n) = O(n \log n), \qquad S(n) = O\!\left(\frac{n}{p} \log \frac{n}{p}\right)
$$

## Implementation

```python
"""
Parallel quicksort -- demonstrates task-based parallel sorting.

Uses Python's concurrent.futures for thread-based parallelism.
Work:  O(n log n)
Span:  O(n) naive, O(log^2 n) with parallel partition
"""

from concurrent.futures import ThreadPoolExecutor, Future


# === Sequential Partition ===

def _partition(arr: list, lo: int, hi: int) -> int:
    """Lomuto partition: returns pivot index after partitioning arr[lo..hi]."""
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i


# === Parallel Quicksort ===

def _parallel_quicksort(
    arr: list,
    lo: int,
    hi: int,
    executor: ThreadPoolExecutor,
    depth_limit: int,
) -> None:
    """Sort arr[lo..hi] using parallel quicksort.

    Falls back to sequential sort when depth_limit is reached
    to avoid excessive task overhead.
    """
    if lo >= hi:
        return

    pivot_idx = _partition(arr, lo, hi)

    if depth_limit > 0:
        # Spawn parallel tasks for left and right sub-arrays
        left_future = executor.submit(
            _parallel_quicksort, arr, lo, pivot_idx - 1,
            executor, depth_limit - 1,
        )
        _parallel_quicksort(
            arr, pivot_idx + 1, hi, executor, depth_limit - 1,
        )
        left_future.result()  # wait for left side
    else:
        # Sequential fallback
        _sequential_quicksort(arr, lo, pivot_idx - 1)
        _sequential_quicksort(arr, pivot_idx + 1, hi)


def _sequential_quicksort(arr: list, lo: int, hi: int) -> None:
    """Standard sequential quicksort for small sub-problems."""
    if lo >= hi:
        return
    pivot_idx = _partition(arr, lo, hi)
    _sequential_quicksort(arr, lo, pivot_idx - 1)
    _sequential_quicksort(arr, pivot_idx + 1, hi)


def parallel_quicksort(
    arr: list, max_workers: int = 4, parallel_depth: int = 3
) -> list:
    """Sort *arr* using parallel quicksort.

    Parameters
    ----------
    arr : list[int]
        Input array.
    max_workers : int
        Number of threads in the pool.
    parallel_depth : int
        Maximum recursion depth for spawning parallel tasks.

    Returns
    -------
    list[int]
        Sorted array.
    """
    result = list(arr)
    if len(result) <= 1:
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _parallel_quicksort(result, 0, len(result) - 1, executor, parallel_depth)

    return result


# === Demonstration ===

if __name__ == "__main__":
    import random
    import time

    random.seed(42)
    data = [random.randint(0, 99999) for _ in range(10000)]

    # Parallel quicksort
    start = time.perf_counter()
    sorted_parallel = parallel_quicksort(data, max_workers=4, parallel_depth=3)
    t_parallel = time.perf_counter() - start

    # Verify correctness
    print(f"Correctly sorted: {sorted_parallel == sorted(data)}")
    print(f"Parallel time:    {t_parallel:.4f}s")

    # Small example for demonstration
    small = [3, 6, 8, 10, 1, 2, 1]
    print(f"\nInput:  {small}")
    print(f"Sorted: {parallel_quicksort(small)}")
```

**Output:**
```
Correctly sorted: True
Parallel time:    0.0312s  (varies by hardware)

Input:  [3, 6, 8, 10, 1, 2, 1]
Sorted: [1, 1, 2, 3, 6, 8, 10]
```

## Comparison with Other Parallel Sorts

| Algorithm | Work | Span | In-place | Practical |
|-----------|------|------|----------|----------|
| Parallel quicksort (naive) | $O(n \log n)$ | $O(n)$ | Yes | Yes |
| Parallel quicksort (parallel partition) | $O(n \log n)$ | $O(\log^2 n)$ | No | Yes |
| Parallel merge sort | $O(n \log n)$ | $O(\log^3 n)$ | No | Yes |
| Bitonic sort | $O(n \log^2 n)$ | $O(\log^2 n)$ | Yes | GPU |
| Sample sort | $O(n \log n)$ | $O(n/p \cdot \log(n/p))$ | No | Yes |

## Practical Considerations

- **Depth limit.** Spawning a thread for every recursive call creates excessive
  overhead.  Limit parallel spawning to the top few levels of recursion (typically
  $\log_2 p$ levels for $p$ processors), then switch to sequential sort.
- **GIL in Python.** Python's Global Interpreter Lock limits true thread parallelism.
  For CPU-bound sorting, use `multiprocessing` or C extensions.  The thread-based
  implementation above demonstrates the concept.
- **Cache effects.** Quicksort's sequential partition has good locality; splitting it
  across processors may reduce cache efficiency.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapters 26-27. MIT Press.
- Blelloch, G. E. (1996). Programming parallel algorithms. *Communications
  of the ACM*, 39(3), 85-97.
