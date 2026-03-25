# Randomized Selection

Finding the $k$-th smallest element in an unsorted array is a fundamental problem. Sorting first takes $O(n \log n)$, but selection can be solved faster. **Randomized selection** (also called Randomized-Select or QuickSelect) uses the partitioning idea from quicksort with a random pivot to achieve $O(n)$ expected time — a significant improvement that demonstrates the power of randomization for avoiding worst-case partitions.

## Algorithm

Given an array $A[1 \ldots n]$ and a rank $k$ (where $1 \leq k \leq n$), randomized selection works as follows:

1. If $n = 1$, return $A[1]$.
2. Choose a pivot index $q$ uniformly at random from $\{1, \ldots, n\}$.
3. Partition $A$ around $A[q]$. Let $r$ be the rank of the pivot after partitioning.
4. If $k = r$, return the pivot.
5. If $k < r$, recurse on the left subarray (elements smaller than the pivot).
6. If $k > r$, recurse on the right subarray seeking rank $k - r$.

Unlike quicksort, randomized selection recurses on only **one** side of the partition, which is the key to achieving linear expected time.

## Expected Running Time Analysis

Let $T(n)$ denote the expected running time on an input of size $n$. After partitioning (which takes $\Theta(n)$ comparisons), the pivot lands at some rank $r$. By symmetry of the random choice, each rank is equally likely. The algorithm recurses on a subproblem of size $\max(r - 1, n - r)$ in the worst case.

For a more precise analysis, observe that a pivot is "good" if it lands in the middle half of the sorted order (ranks $n/4$ through $3n/4$). A good pivot reduces the subproblem size to at most $3n/4$. The probability of a good pivot is $1/2$.

The expected number of comparisons before a good pivot is found follows a geometric distribution with mean 2. Each partitioning step costs at most $cn$ comparisons. After a good pivot, the subproblem has size at most $3n/4$. Thus

$$
E[T(n)] \leq E[T(3n/4)] + E[\text{cost of partitioning steps}]
$$

The expected partitioning cost per "phase" (until a good pivot is found) is at most $2cn$. This gives

$$
E[T(n)] \leq E[T(3n/4)] + 2cn
$$

Expanding the recurrence,

$$
E[T(n)] \leq 2cn + 2c \cdot \frac{3n}{4} + 2c \cdot \left(\frac{3}{4}\right)^2 n + \cdots = 2cn \sum_{i=0}^{\infty} \left(\frac{3}{4}\right)^i = 8cn
$$

Therefore $E[T(n)] = O(n)$.

## Precise Indicator Variable Analysis

For a tighter constant, define indicator variables. Let $z_1 < z_2 < \cdots < z_n$ be the sorted elements, and suppose we seek rank $k$. Element $z_i$ is compared to the pivot if and only if $z_i$ is the first element chosen as pivot from the set of candidates that separates $z_i$ from $z_k$ (or includes both). The analysis yields

$$
E[\text{comparisons}] \leq 4n
$$

!!! tip "Single Recursion vs Double Recursion"
    The key difference from quicksort is that selection recurses on only one subarray. This is why the recurrence sums a geometric series rather than two equal-sized subproblems, yielding $O(n)$ instead of $O(n \log n)$.

## Implementation

```python
"""
Randomized selection (QuickSelect) for finding the k-th smallest element.

Expected O(n) time via random pivot selection.
"""

import random

# === Partition ===

def partition(arr, lo, hi):
    """Lomuto partition scheme around arr[hi]."""
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i

# === Randomized Select ===

def randomized_select(arr, lo, hi, k):
    """Return the k-th smallest element in arr[lo..hi] (0-indexed k)."""
    if lo == hi:
        return arr[lo]
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    q = partition(arr, lo, hi)
    rank = q - lo
    if k == rank:
        return arr[q]
    elif k < rank:
        return randomized_select(arr, lo, q - 1, k)
    else:
        return randomized_select(arr, q + 1, hi, k - rank - 1)

# === Main ===

if __name__ == "__main__":
    data = [7, 10, 4, 3, 20, 15]
    k = 3  # 0-indexed: 4th smallest
    result = randomized_select(data[:], 0, len(data) - 1, k)
    print(f"The {k+1}-th smallest element is {result}")
```

**Output:**
```
The 4th smallest element is 10
```

## Worst Case and Comparison to Deterministic Selection

The worst-case running time of randomized selection is $O(n^2)$, occurring when every pivot is the minimum or maximum. However, this event has exponentially small probability.

The deterministic **median-of-medians** algorithm guarantees $O(n)$ worst-case time, but with a larger constant factor (roughly $24n$ comparisons vs $4n$ expected for randomized selection). In practice, randomized selection is faster due to better cache behavior and simpler code.

| Algorithm | Expected time | Worst-case time | Practical speed |
|---|---|---|---|
| Randomized Select | $O(n)$ | $O(n^2)$ | Fast |
| Median of Medians | $O(n)$ | $O(n)$ | Slower (larger constant) |
| Sort then index | $O(n \log n)$ | $O(n \log n)$ | Moderate |

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.
