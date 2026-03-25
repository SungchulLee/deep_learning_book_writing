# Build Heap O(n) Proof

At first glance, Build-Heap appears to cost $O(n \log n)$: it calls sift-down on $O(n)$ nodes, and each sift-down can take up to $O(\log n)$ time. The surprising result is that the total work is only $O(n)$. The proof exploits the fact that most nodes reside near the bottom of the tree where their subtree heights are small, so the expensive sift-down operations are rare.

## Setup

Consider a complete binary tree with $n$ nodes. The height of the tree is $h = \lfloor \log_2 n \rfloor$. A node at height $k$ (measured from the leaves, where leaves have height 0) can require at most $k$ swaps during sift-down.

**Key observation**: the number of nodes at height $k$ in a heap of size $n$ is at most

$$
\left\lceil \frac{n}{2^{k+1}} \right\rceil
$$

This bound follows because at height $k$, the nodes are the roots of subtrees of size $2^{k+1} - 1$, and these subtrees partition the heap.

## Theorem

**Build-Heap runs in $O(n)$ time.**

*Proof.* The total cost $T(n)$ of Build-Heap is the sum over all non-leaf nodes of the sift-down cost at each node. A node at height $k$ incurs at most $O(k)$ work for sift-down. Summing over all heights:

$$
T(n) = \sum_{k=0}^{\lfloor \log_2 n \rfloor} \left\lceil \frac{n}{2^{k+1}} \right\rceil \cdot O(k)
$$

Dropping the ceiling and constant factors, it suffices to show that the following sum is $O(n)$:

$$
T(n) \le c \cdot n \sum_{k=0}^{\lfloor \log_2 n \rfloor} \frac{k}{2^k}
$$

for some constant $c > 0$. We need to evaluate the series $\sum_{k=0}^{\infty} k / 2^k$.

## Evaluating the Series

We use the identity for the sum of $k x^k$. Start with the geometric series:

$$
\sum_{k=0}^{\infty} x^k = \frac{1}{1-x} \quad \text{for } |x| < 1
$$

Differentiate both sides with respect to $x$:

$$
\sum_{k=1}^{\infty} k x^{k-1} = \frac{1}{(1-x)^2}
$$

Multiply both sides by $x$:

$$
\sum_{k=1}^{\infty} k x^{k} = \frac{x}{(1-x)^2}
$$

Substituting $x = 1/2$:

$$
\sum_{k=0}^{\infty} \frac{k}{2^k} = \sum_{k=1}^{\infty} k \left(\frac{1}{2}\right)^k = \frac{1/2}{(1 - 1/2)^2} = \frac{1/2}{1/4} = 2
$$

## Completing the Proof

Since the series converges to the constant 2:

$$
T(n) \le c \cdot n \sum_{k=0}^{\infty} \frac{k}{2^k} = c \cdot n \cdot 2 = 2cn = O(n)
$$

Therefore Build-Heap runs in $\Theta(n)$ time. $\square$

!!! note "Tight Bound"
    The bound is not only $O(n)$ but $\Theta(n)$. Build-Heap must examine at least $\lfloor n/2 \rfloor$ nodes (all non-leaves), so the lower bound is $\Omega(n)$. Combined with the $O(n)$ upper bound, the complexity is $\Theta(n)$.

## Intuition

The proof works because the "heavy" operations (sift-down over many levels) happen at nodes near the root, but there are very few such nodes. The following table illustrates the distribution:

| Height $k$ | Nodes at height $k$ (approx.) | Work per node | Total work at height $k$ |
|:-----------:|:-----------------------------:|:-------------:|:------------------------:|
| 0 | $n/2$ | 0 | 0 |
| 1 | $n/4$ | 1 | $n/4$ |
| 2 | $n/8$ | 2 | $n/4$ |
| 3 | $n/16$ | 3 | $3n/16$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $\log n$ | 1 | $\log n$ | $\log n$ |

The total work column decreases geometrically. Half the nodes are leaves doing zero work, a quarter do one swap each, an eighth do two swaps each, and so on. The root is the only node that might traverse the full height $\log n$, but a single $\log n$ contribution is negligible relative to $n$.

## Empirical Validation

The following code counts the total number of comparisons made during Build-Heap and confirms the linear relationship.

```python
"""
Empirical validation of Build-Heap O(n) complexity.

Counts the exact number of comparisons made during Build-Heap
for arrays of increasing size and verifies the linear bound.
"""

import random


# === Instrumented Sift-Down ===

def sift_down_count(arr, i, n):
    """Sift down with comparison counting. Returns number of comparisons."""
    comparisons = 0
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n:
            comparisons += 1
            if arr[left] > arr[largest]:
                largest = left

        if right < n:
            comparisons += 1
            if arr[right] > arr[largest]:
                largest = right

        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

    return comparisons


# === Instrumented Build-Heap ===

def build_heap_count(arr):
    """Build a max-heap and return total comparisons."""
    n = len(arr)
    total = 0
    for i in range(n // 2 - 1, -1, -1):
        total += sift_down_count(arr, i, n)
    return total


# === Experiment ===

if __name__ == "__main__":
    print(f"{'n':>10}  {'comparisons':>12}  {'ratio c/n':>10}")
    print("-" * 36)

    for n in [100, 500, 1000, 5000, 10000, 50000, 100000]:
        # Average over multiple trials
        total_comps = 0
        trials = 10
        for _ in range(trials):
            data = list(range(n))
            random.shuffle(data)
            total_comps += build_heap_count(data)
        avg_comps = total_comps / trials
        print(f"{n:>10}  {avg_comps:>12.1f}  {avg_comps / n:>10.4f}")
```

**Output (approximate):**
```
         n   comparisons    ratio c/n
------------------------------------
       100         147.0      1.4700
       500         799.8      1.5996
      1000        1621.3      1.6213
      5000        8347.4      1.6695
     10000       16813.2      1.6813
     50000       84799.5      1.6960
    100000      169998.1      1.6999
```

The ratio of comparisons to $n$ converges to approximately 1.7, confirming that Build-Heap uses $\Theta(n)$ comparisons. The constant approaches $2$ from below, consistent with the theoretical bound derived from $\sum k/2^k = 2$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 6.3: Building a heap. MIT Press.
