# Comparison Tables

Choosing the right algorithm or data structure requires comparing time and space complexity across alternatives. This page collects side-by-side comparisons for the major algorithm families covered throughout the book, providing a quick reference for selecting the best tool for a given problem.

## Sorting Algorithms

All bounds refer to sorting $n$ elements. "Stable" means equal elements preserve their original relative order.

| Algorithm | Best | Average | Worst | Space | Stable |
|---|---|---|---|---|---|
| Insertion sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | Yes |
| Selection sort | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | No |
| Bubble sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | Yes |
| Shell sort | $O(n \log n)$ | depends on gap | $O(n^{3/2})$ | $O(1)$ | No |
| Merge sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes |
| Quick sort | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ | No |
| Heap sort | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ | No |
| Counting sort | $O(n + k)$ | $O(n + k)$ | $O(n + k)$ | $O(k)$ | Yes |
| Radix sort | $O(d(n + k))$ | $O(d(n + k))$ | $O(d(n + k))$ | $O(n + k)$ | Yes |
| Timsort | $O(n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Yes |

The comparison-based lower bound is $\Omega(n \log n)$. Non-comparison sorts (counting, radix) bypass this bound by exploiting key structure.

## Data Structures

Average-case complexity for basic operations on $n$ elements.

| Structure | Search | Insert | Delete | Space |
|---|---|---|---|---|
| Array (unsorted) | $O(n)$ | $O(1)$ amort. | $O(n)$ | $O(n)$ |
| Array (sorted) | $O(\log n)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| Linked list | $O(n)$ | $O(1)$ | $O(1)$* | $O(n)$ |
| Hash table | $O(1)$ exp. | $O(1)$ exp. | $O(1)$ exp. | $O(n)$ |
| BST (balanced) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| Red-black tree | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| B-tree | $O(\log_B n)$ | $O(\log_B n)$ | $O(\log_B n)$ | $O(n)$ |
| Skip list | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(n)$ exp. |

\* $O(1)$ deletion assumes a pointer to the node is given.

## Graph Algorithms

For a graph with $|V|$ vertices and $|E|$ edges.

| Algorithm | Problem | Time | Space |
|---|---|---|---|
| BFS | Traversal / shortest (unweighted) | $O(|V| + |E|)$ | $O(|V|)$ |
| DFS | Traversal / topological sort | $O(|V| + |E|)$ | $O(|V|)$ |
| Dijkstra (binary heap) | SSSP (non-negative) | $O((|V| + |E|) \log |V|)$ | $O(|V|)$ |
| Bellman-Ford | SSSP (general) | $O(|V| \cdot |E|)$ | $O(|V|)$ |
| Floyd-Warshall | APSP | $O(|V|^3)$ | $O(|V|^2)$ |
| Johnson's | APSP (sparse) | $O(|V| \cdot |E| + |V|^2 \log |V|)$ | $O(|V|^2)$ |
| Kruskal | MST | $O(|E| \log |E|)$ | $O(|V|)$ |
| Prim (binary heap) | MST | $O((|V| + |E|) \log |V|)$ | $O(|V|)$ |
| Tarjan | SCC | $O(|V| + |E|)$ | $O(|V|)$ |
| Ford-Fulkerson (Edmonds-Karp) | Max flow | $O(|V| \cdot |E|^2)$ | $O(|V|^2)$ |

## Heap Variants

| Heap Type | Insert | Extract-Min | Decrease-Key | Merge | Space |
|---|---|---|---|---|---|
| Binary heap | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| d-ary heap | $O(\log_d n)$ | $O(d \log_d n)$ | $O(\log_d n)$ | $O(n)$ | $O(n)$ |
| Binomial heap | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| Fibonacci heap | $O(1)$ amort. | $O(\log n)$ amort. | $O(1)$ amort. | $O(1)$ | $O(n)$ |
| Pairing heap | $O(1)$ | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(1)$ | $O(n)$ |

## String Algorithms

For a pattern of length $m$ and text of length $n$.

| Algorithm | Preprocessing | Search | Space |
|---|---|---|---|
| Naive | $O(1)$ | $O(nm)$ | $O(1)$ |
| KMP | $O(m)$ | $O(n)$ | $O(m)$ |
| Boyer-Moore | $O(m + |\Sigma|)$ | $O(n/m)$ best, $O(nm)$ worst | $O(m + |\Sigma|)$ |
| Rabin-Karp | $O(m)$ | $O(n)$ expected | $O(1)$ |
| Aho-Corasick ($k$ patterns) | $O(\sum m_i)$ | $O(n + z)$ | $O(\sum m_i)$ |
| Suffix array | $O(n)$ | $O(m \log n)$ | $O(n)$ |
| Suffix tree | $O(n)$ | $O(m)$ | $O(n)$ |

Here $|\Sigma|$ is the alphabet size and $z$ is the number of matches.

## How to Choose

When selecting an algorithm or data structure, consider:

1. **Input size**: Asymptotic complexity matters for large $n$; constants dominate for small $n$.
2. **Operation mix**: A hash table beats a BST if you never need ordered traversal.
3. **Worst case vs. expected**: Randomized algorithms (quicksort, skip lists, treaps) have excellent expected performance but occasionally bad worst cases.
4. **Space constraints**: In-place algorithms ($O(1)$ extra space) are preferred when memory is limited.
5. **Stability**: Required when the relative order of equal elements has meaning.

!!! tip "Practical rule of thumb"
    For $n < 50$, insertion sort often outperforms asymptotically faster algorithms due to low overhead and cache efficiency. For $n > 10^6$, always use an $O(n \log n)$ sort or better.

## Implementation

```python
"""
Algorithm Complexity Comparison -- empirical timing of sorting algorithms.

Times several sorting algorithms on random arrays of increasing size
to verify their asymptotic behavior in practice.
"""

import random
import time
from typing import Callable


# === Sorting Algorithms =======================================================

def insertion_sort(arr: list[int]) -> list[int]:
    """Sort using insertion sort. O(n^2)."""
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def merge_sort(arr: list[int]) -> list[int]:
    """Sort using merge sort. O(n log n)."""
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def python_sort(arr: list[int]) -> list[int]:
    """Sort using Python's built-in Timsort. O(n log n)."""
    return sorted(arr)


# === Benchmark ================================================================

def benchmark(func: Callable, arr: list[int]) -> float:
    """Return the time in milliseconds to sort *arr* with *func*."""
    start = time.perf_counter()
    func(arr)
    return (time.perf_counter() - start) * 1000


# === Main =====================================================================

if __name__ == "__main__":
    random.seed(42)
    sizes = [100, 500, 1000, 5000]

    print(f"{'n':>6}  {'Insertion':>12}  {'Merge':>12}  {'Timsort':>12}")
    print(f"{'':>6}  {'(ms)':>12}  {'(ms)':>12}  {'(ms)':>12}")
    print("-" * 50)

    for n in sizes:
        arr = [random.randint(0, 10 * n) for _ in range(n)]
        t_ins = benchmark(insertion_sort, arr)
        t_merge = benchmark(merge_sort, arr)
        t_tim = benchmark(python_sort, arr)
        print(f"{n:>6}  {t_ins:>12.3f}  {t_merge:>12.3f}  {t_tim:>12.3f}")
```

**Output:**

```
     n     Insertion         Merge       Timsort
               (ms)          (ms)          (ms)
--------------------------------------------------
   100         0.287         0.120         0.005
   500         6.543         0.680         0.028
  1000        26.123         1.511         0.061
  5000       645.231         9.120         0.391
```

The empirical timings confirm the theoretical analysis: insertion sort's quadratic growth is visible (roughly 25x slowdown for 5x more data), merge sort grows nearly linearly in this range, and Python's built-in Timsort is fastest due to its optimized C implementation and adaptive behavior.

## Reference

- Cormen, T.H., Leiserson, C.E., Rivest, R.L., and Stein, C. *Introduction to Algorithms*. MIT Press
- Sedgewick, R. and Wayne, K. *Algorithms*. Addison-Wesley
