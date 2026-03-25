# Heapsort Analysis

Understanding *why* heapsort runs in $O(n \log n)$ time requires examining both phases independently.  The build-heap phase is surprisingly $O(n)$ -- not $O(n \log n)$ as a naive argument might suggest -- and the extraction phase contributes the dominant $O(n \log n)$ term.  This page derives both bounds and discusses the practical implications.

## Build Max-Heap Complexity

### Naive Upper Bound

Calling `sift_down` on each of the $n/2$ internal nodes appears to cost $O(\log n)$ each, giving $O(n \log n)$.  This bound is correct but not tight.

### Tight O(n) Bound

The key observation is that most nodes are near the bottom of the tree and require very few swaps.  In a heap of height $h = \lfloor \log_2 n \rfloor$:

- At depth $d$, there are at most $\lceil n / 2^{d+1} \rceil$ nodes.
- A node at depth $d$ has height $h - d$ and `sift_down` costs at most $O(h - d)$.

The total work is:

$$
W = \sum_{d=0}^{h} \left\lceil \frac{n}{2^{d+1}} \right\rceil \cdot O(h - d)
$$

Substituting $k = h - d$ (so $d = h - k$):

$$
W = \sum_{k=0}^{h} \left\lceil \frac{n}{2^{h - k + 1}} \right\rceil \cdot O(k) \leq \sum_{k=0}^{h} \frac{n}{2^{h - k + 1}} \cdot k
$$

Since $2^h \leq n$, we have $n / 2^{h+1} \leq 1$, and the sum simplifies to:

$$
W \leq n \sum_{k=0}^{\infty} \frac{k}{2^k} = n \cdot \frac{1/2}{(1 - 1/2)^2} = 2n = O(n)
$$

The identity $\sum_{k=0}^{\infty} k x^k = x / (1 - x)^2$ for $|x| < 1$ with $x = 1/2$ closes the proof.

??? note "Intuition for the O(n) bound"
    Half the nodes are leaves (height 0) and do zero work.  A quarter of the nodes have height 1 and do at most 1 swap.  An eighth have height 2 and do at most 2 swaps.  The work per level decreases geometrically, so the total is dominated by a constant times $n$.

## Extraction Phase Complexity

After building the heap, we perform $n - 1$ extract-max operations.  Each extraction:

1. Swaps the root with the last element: $O(1)$.
2. Calls `sift_down` on the new root through a heap of decreasing size: $O(\log k)$ for the $k$-th extraction.

The total cost is:

$$
\sum_{k=1}^{n-1} O(\log k) = O\!\left(\sum_{k=1}^{n} \log k\right) = O(\log(n!)) = O(n \log n)
$$

The last equality uses Stirling's approximation: $\log(n!) = \Theta(n \log n)$.

## Total Complexity

Combining both phases:

$$
T(n) = \underbrace{O(n)}_{\text{build heap}} + \underbrace{O(n \log n)}_{\text{extract all}} = O(n \log n)
$$

### Best, Average, and Worst Cases

| Case    | Time           | When it occurs |
|---------|----------------|----------------|
| Best    | $O(n \log n)$  | All elements identical (still performs all swaps) |
| Average | $O(n \log n)$  | Random permutation |
| Worst   | $O(n \log n)$  | Any input (guaranteed) |

Unlike quicksort, heapsort has **no pathological inputs**.  The $O(n \log n)$ bound holds for every input, not just in expectation.

!!! tip "Heapsort vs. the comparison lower bound"
    The comparison-based sorting lower bound is $\Omega(n \log n)$.  Heapsort matches this bound in the worst case, making it asymptotically optimal among comparison sorts.

## Space Complexity

Heapsort uses $O(1)$ auxiliary space because it operates on the array in place.  The only extra storage is a constant number of variables for indices and temporary swaps.

If `sift_down` is implemented recursively, the call stack uses $O(\log n)$ space.  An iterative implementation of `sift_down` reduces this to $O(1)$.

## Practical Considerations

### Cache Performance

Heapsort accesses array elements in a pattern determined by the heap structure: a node at index $i$ accesses indices $2i + 1$ and $2i + 2$.  For large arrays, these jumps frequently cross cache-line boundaries, leading to more cache misses than sequential-access algorithms like merge sort or quicksort.

### Comparison Count

Heapsort performs approximately $2n \log_2 n$ comparisons in the worst case (two comparisons per level of `sift_down` -- one to find the larger child, one to compare with the parent).  This is roughly twice the information-theoretic minimum of $n \log_2 n - O(n)$ comparisons.

### Stability

Heapsort is **not stable**.  The extraction phase can change the relative order of equal elements.  When stability is required, merge sort or a stable variant of heapsort (at the cost of extra space) is preferred.

## Python Demonstration

```python
"""
Heapsort analysis demonstration.

Verifies the O(n log n) behavior empirically by counting comparisons
during heapsort on arrays of increasing size.
"""

import math


# === Comparison-counting heapsort =============================================

def sift_down_counted(arr: list, i: int, heap_size: int, count: list) -> None:
    """Sift down with comparison counting."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < heap_size:
        count[0] += 1
        if arr[left] > arr[largest]:
            largest = left
    if right < heap_size:
        count[0] += 1
        if arr[right] > arr[largest]:
            largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        sift_down_counted(arr, largest, heap_size, count)


def heapsort_counted(arr: list) -> int:
    """Heapsort returning the total number of comparisons."""
    n = len(arr)
    count = [0]

    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        sift_down_counted(arr, i, n, count)

    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        sift_down_counted(arr, 0, i, count)

    return count[0]


# === Main =====================================================================

if __name__ == "__main__":
    import random

    print(f"{'n':>8}  {'comparisons':>12}  {'2n lg n':>10}  {'ratio':>6}")
    print("-" * 44)

    for n in [100, 500, 1000, 5000, 10000]:
        arr = list(range(n))
        random.shuffle(arr)
        comps = heapsort_counted(arr)
        theory = 2 * n * math.log2(n) if n > 1 else 1
        ratio = comps / theory
        print(f"{n:>8}  {comps:>12}  {theory:>10.0f}  {ratio:>6.3f}")
```

**Output (typical, varies with random seed):**
```
       n   comparisons     2n lg n   ratio
--------------------------------------------
     100          1036        1329   0.780
     500          7448        8966   0.831
    1000         17129       19932   0.859
    5000        107826      122877   0.877
   10000        234820      265754   0.884
```

The ratio remains below 1.0, confirming that the actual comparison count stays within the $2n \log_2 n$ upper bound.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Chapter 6.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley, Section 2.4.
- Floyd, R. W. (1964). Algorithm 245: Treesort 3. *Communications of the ACM*, 7(12), 701.
