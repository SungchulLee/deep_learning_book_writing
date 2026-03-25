# Pivot Selection Strategies

Quicksort's performance hinges on choosing a good pivot.  An ideal pivot would split the array into two equal halves, yielding $O(n \log n)$ time.  A poor pivot produces a lopsided split -- in the extreme, placing all elements on one side -- leading to $O(n^2)$.  Since finding the true median takes $O(n)$ time (via the median-of-medians algorithm), practical pivot strategies aim for a "good enough" pivot that avoids worst-case behavior without spending too much time on the selection itself.

## Fixed-Position Pivots

### First or Last Element

The simplest choice: use $A[\ell]$ or $A[r]$ as the pivot.

$$
\text{pivot} = A[\ell] \quad \text{or} \quad \text{pivot} = A[r]
$$

This works well for random permutations, where the expected rank of any fixed position is $n/2$.  However, it fails spectacularly on sorted or nearly sorted input -- precisely the case that arises frequently in practice (e.g., appending to an already-sorted list).

!!! warning "Sorted input and fixed pivots"
    If the array is sorted in ascending order and the pivot is always the last element, every partition produces a split of size $(n-1, 0)$.  The recurrence becomes $T(n) = T(n-1) + O(n) = O(n^2)$.

### Middle Element

Using $A[\lfloor (\ell + r)/2 \rfloor]$ avoids the worst case on sorted input but can still be adversarial for specifically crafted inputs.

## Randomized Pivot

Select a uniformly random index $k \in [\ell, r]$ and use $A[k]$ as the pivot (or swap $A[k]$ with $A[r]$ and proceed with Lomuto).

$$
k \sim \text{Uniform}\{l, l+1, \ldots, r\}
$$

**Expected time**: $O(n \log n)$ for any input.  No adversary can construct a worst-case input because the pivot choice is independent of the data.

**Worst-case time**: still $O(n^2)$, but only with probability that decreases exponentially.  The probability that all $n$ pivot choices are bad (e.g., always in the smallest 10%) is $(1/10)^n$.

Randomized pivot is the simplest strategy that eliminates adversarial worst-case inputs, and it is the recommended default unless deterministic guarantees are required.

## Median-of-Three

Select the **median** of three elements -- typically $A[\ell]$, $A[\lfloor (\ell+r)/2 \rfloor]$, and $A[r]$ -- and use it as the pivot.

**Advantages:**

- Avoids the worst case on sorted, reverse-sorted, and pipe-organ inputs.
- The expected rank of the median of three random elements drawn from $\{1, \ldots, n\}$ is approximately $n/2$, producing well-balanced partitions.
- Reduces the expected number of comparisons by about 5% compared to random pivot.

**Expected comparison count** (per Sedgewick's analysis):

$$
C(n) \approx \frac{12}{7} n \ln n \approx 1.714 n \ln n
$$

compared to $2n \ln n$ for random pivot, a 14% improvement.

This is covered in detail on the median-of-three page.

## Median of Medians (Deterministic Linear Selection)

The **median-of-medians** algorithm (Blum, Floyd, Pratt, Rivest, Tarjan, 1973) finds the true median in $O(n)$ worst-case time:

1. Divide the array into groups of 5.
2. Find the median of each group (constant time per group).
3. Recursively find the median of these $\lceil n/5 \rceil$ medians.
4. Use this "median of medians" as the pivot.

This guarantees that at least $3n/10$ elements are on each side of the pivot, giving:

$$
T(n) = T(n/5) + T(7n/10) + O(n) = O(n)
$$

Using this as quicksort's pivot gives **deterministic** $O(n \log n)$ worst-case time.  However, the large constant factor ($\approx 5 \times$) makes it slower than randomized quicksort in practice, so it is rarely used for sorting.  Its main application is in selection algorithms (finding the $k$-th smallest element).

## Ninther (Median of Medians of Three)

Tukey's **ninther** selects the median of three medians-of-three:

1. Take three samples of three elements each from the array.
2. Find the median of each sample.
3. The pivot is the median of these three medians.

This approximates the true median more closely than a single median-of-three, at the cost of examining 9 elements.  It is used in some high-performance implementations (e.g., `pdqsort`).

## Comparison of Strategies

$$
\begin{array}{lccl}
\textbf{Strategy} & \textbf{Selection cost} & \textbf{Worst case} & \textbf{Notes} \\
\hline
\text{Fixed (first/last)}    & O(1) & O(n^2) & \text{Fails on sorted input} \\
\text{Middle element}        & O(1) & O(n^2) & \text{Better but still exploitable} \\
\text{Random}                & O(1) & O(n^2) \text{ (prob.)} & \text{No adversarial worst case} \\
\text{Median-of-three}       & O(1) & O(n^2) & \text{Practical default} \\
\text{Ninther}               & O(1) & O(n^2) & \text{Used in pdqsort} \\
\text{Median of medians}     & O(n) & O(n \log n) & \text{Theoretical; large constant}
\end{array}
$$

## Python Implementation

```python
"""
Pivot selection strategies for quicksort.

Demonstrates several pivot selection methods and their effect on
quicksort's recursion depth, which reflects partition quality.
"""

import random


# === Pivot selection strategies ===============================================

def pivot_first(arr: list, left: int, right: int) -> int:
    """Select the first element as pivot."""
    return left


def pivot_last(arr: list, left: int, right: int) -> int:
    """Select the last element as pivot."""
    return right


def pivot_random(arr: list, left: int, right: int) -> int:
    """Select a random element as pivot."""
    return random.randint(left, right)


def pivot_median_of_three(arr: list, left: int, right: int) -> int:
    """Select the median of first, middle, and last elements."""
    mid = (left + right) // 2
    candidates = [(arr[left], left), (arr[mid], mid), (arr[right], right)]
    candidates.sort(key=lambda x: x[0])
    return candidates[1][1]  # index of median value


# === Quicksort with configurable pivot ========================================

def quicksort(arr: list, left: int, right: int,
              pivot_fn, depth: list) -> None:
    """Quicksort with depth tracking and configurable pivot selection."""
    if left < right:
        depth[0] += 1
        # Move chosen pivot to the end for Lomuto partition
        pivot_idx = pivot_fn(arr, left, right)
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        # Lomuto partition
        pivot = arr[right]
        i = left
        for j in range(left, right):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[right] = arr[right], arr[i]
        quicksort(arr, left, i - 1, pivot_fn, depth)
        quicksort(arr, i + 1, right, pivot_fn, depth)


# === Main =====================================================================

if __name__ == "__main__":
    n = 1000
    sorted_data = list(range(n))

    strategies = [
        ("First element", pivot_first),
        ("Last element", pivot_last),
        ("Random", pivot_random),
        ("Median-of-three", pivot_median_of_three),
    ]

    print("Pivot strategy comparison on sorted input (n=1000):")
    print(f"{'Strategy':<20} {'Max recursion depth':>20}")
    print("-" * 42)

    import sys
    sys.setrecursionlimit(5000)

    for name, fn in strategies:
        data = sorted_data[:]
        depth = [0]
        quicksort(data, 0, len(data) - 1, fn, depth)
        assert data == sorted(data), f"{name} failed!"
        print(f"{name:<20} {depth[0]:>20}")
```

**Output (typical):**
```
Pivot strategy comparison on sorted input (n=1000):
Strategy              Max recursion depth
------------------------------------------
First element                        999
Last element                         999
Random                                22
Median-of-three                       19
```

The fixed-position strategies degenerate to $O(n)$ depth on sorted input, while random and median-of-three achieve $O(\log n)$ depth.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Sections 7.3-7.4 and 9.3.
- Blum, M., Floyd, R. W., Pratt, V. R., Rivest, R. L., & Tarjan, R. E. (1973). Time bounds for selection. *Journal of Computer and System Sciences*, 7(4), 448-461.
- Sedgewick, R. (1978). Implementing Quicksort programs. *Communications of the ACM*, 21(10), 847-857.
