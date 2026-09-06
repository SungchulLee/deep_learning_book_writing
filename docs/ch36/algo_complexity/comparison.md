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

## Exercises

**Exercise 1.**
Given $n = 10^6$ elements and a 1-second time limit, determine which complexities are feasible: $O(n)$, $O(n \log n)$, $O(n \sqrt{n})$, $O(n^2)$.

??? success "Solution to Exercise 1"
    At $\sim 10^8$ operations/second in C++: $O(n) = 10^6$ operations (feasible). $O(n \log n) = 10^6 \times 20 = 2 \times 10^7$ (feasible). $O(n \sqrt{n}) = 10^6 \times 10^3 = 10^9$ (borderline, may TLE). $O(n^2) = 10^{12}$ (infeasible, 10,000x over budget). Conclusion: $O(n \log n)$ is the practical ceiling for $n = 10^6$, with $O(n \sqrt{n})$ possible only with small constants and fast I/O. $\square$

---

**Exercise 2.**
A problem can be solved with either a sorting-based $O(n \log n)$ algorithm or a hash-based $O(n)$ expected-time algorithm. Discuss when each is preferable considering constant factors, worst cases, and memory.

??? success "Solution to Exercise 2"
    The hash-based approach has better asymptotic time but: (1) hash table operations have higher constant factors (hashing, collision resolution, cache misses from random access) -- typically 5--10x slower per operation than array access. For $n < 10^5$, sorting may be faster despite the $\log n$ factor. (2) Hash tables have $O(n)$ worst case if hash collisions are adversarial. Sorting is always $O(n \log n)$ worst case. (3) Hash tables use $O(n)$ extra memory with a constant factor of 2--3x. Sorting can be done in-place ($O(1)$ extra). Sorting is preferable when: deterministic worst-case behavior is needed, memory is tight, or the sorted output is useful for subsequent operations (e.g., binary search, merge). Hashing is preferable when average-case performance matters and $n$ is large. $\square$

---

**Exercise 3.**
Rank the following algorithm complexities from fastest to slowest for $n = 10^4$: $O(2^n)$, $O(n!)$, $O(n^3)$, $O(n \log n)$, $O(n^2)$, $O(\log n)$, $O(1)$, $O(n)$. Compute approximate operation counts for each.

??? success "Solution to Exercise 3"
    For $n = 10^4$: $O(1) = 1$. $O(\log n) = \log_2(10^4) \approx 13$. $O(n) = 10^4$. $O(n \log n) = 10^4 \times 13 = 1.3 \times 10^5$. $O(n^2) = 10^8$. $O(n^3) = 10^{12}$. $O(2^n) = 2^{10000} \approx 10^{3010}$. $O(n!) = (10^4)! \approx 10^{35659}$. Ranked: $O(1) < O(\log n) < O(n) < O(n \log n) < O(n^2) < O(n^3) < O(2^n) < O(n!)$. Only the first five are computationally feasible. $O(n^3)$ requires $\sim 10^{12}$ operations ($\sim 10{,}000$ seconds) -- infeasible in typical time limits. $\square$

---

**Exercise 4.**
Explain the difference between amortized $O(1)$ and worst-case $O(1)$. Give a data structure example where the distinction matters practically.

??? success "Solution to Exercise 4"
    **Worst-case $O(1)$**: every single operation completes in constant time, guaranteed. Example: array access by index is always $O(1)$. **Amortized $O(1)$**: the average cost per operation over a sequence of $n$ operations is $O(1)$, but individual operations may cost $O(n)$. Example: `std::vector::push_back` in C++ -- most pushes are $O(1)$, but when the vector is full, it resizes (doubling capacity), copying all $n$ elements in $O(n)$. Over $n$ pushes, total cost is $O(n)$, so amortized cost is $O(1)$. The distinction matters in real-time systems: a latency-sensitive financial trading system cannot tolerate occasional $O(n)$ spikes from vector resizing. It must use pre-allocated arrays or ring buffers with worst-case $O(1)$ operations. For general-purpose applications, amortized $O(1)$ is sufficient. $\square$

---

**Exercise 5.**
A problem requires $O(n \log n)$ preprocessing and then answers $q$ queries in $O(\log n)$ each. Compare the total time with an alternative that answers each query in $O(n)$ with no preprocessing. For what values of $q$ is the preprocessing approach faster?

??? success "Solution to Exercise 5"
    With preprocessing: total time $= O(n \log n + q \log n)$. Without preprocessing: total time $= O(qn)$. The preprocessing approach is faster when $n \log n + q \log n < qn$, i.e., $n \log n < q(n - \log n) \approx qn$ for large $n$. This gives $q > \log n$. For $n = 10^6$: $\log_2 n \approx 20$. If $q > 20$, preprocessing wins. If $q = 1$ (single query), the naive approach ($O(n)$) is faster than $O(n \log n)$ preprocessing. In practice, the crossover point is even lower because the preprocessing constant is larger. The key insight: amortize fixed setup cost over many queries. $\square$
