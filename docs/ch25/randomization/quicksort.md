# Randomized Quicksort

Deterministic quicksort is vulnerable to adversarial inputs: if the pivot selection rule is predictable, an adversary can construct inputs that force $O(n^2)$ comparisons. **Randomized quicksort** eliminates this vulnerability by choosing pivots uniformly at random, ensuring that no input can consistently trigger worst-case behavior. The expected number of comparisons is $O(n \log n)$ for every input, and the analysis using indicator random variables is one of the most elegant applications of linearity of expectation.

## Algorithm

Given an array $A[1 \ldots n]$, randomized quicksort proceeds as follows:

1. If $n \leq 1$, return.
2. Choose a pivot index $q$ uniformly at random from $\{1, 2, \ldots, n\}$.
3. Partition $A$ around $A[q]$: elements smaller than $A[q]$ go left, elements larger go right.
4. Recurse on the left and right subarrays.

The only randomness is in step 2. The algorithm always produces a correctly sorted array (it is a Las Vegas algorithm).

## Expected Number of Comparisons

Let $z_1 < z_2 < \cdots < z_n$ be the elements of $A$ in sorted order. Define the indicator random variable

$$
X_{ij} = \begin{cases} 1 & \text{if } z_i \text{ and } z_j \text{ are compared during execution} \\ 0 & \text{otherwise} \end{cases}
$$

The total number of comparisons is

$$
X = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} X_{ij}
$$

By linearity of expectation,

$$
E[X] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \Pr[z_i \text{ and } z_j \text{ are compared}]
$$

**Key observation.** Elements $z_i$ and $z_j$ are compared if and only if one of them is the first element from the set $\{z_i, z_{i+1}, \ldots, z_j\}$ to be chosen as a pivot. If any element $z_k$ with $i < k < j$ is chosen as pivot first, then $z_i$ and $z_j$ are separated into different subarrays and never compared.

The set $\{z_i, z_{i+1}, \ldots, z_j\}$ has $j - i + 1$ elements, each equally likely to be the first pivot chosen from this set. The probability that $z_i$ or $z_j$ is chosen first is

$$
\Pr[X_{ij} = 1] = \frac{2}{j - i + 1}
$$

Therefore,

$$
E[X] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j - i + 1}
$$

Substituting $k = j - i$,

$$
E[X] = \sum_{i=1}^{n-1} \sum_{k=1}^{n-i} \frac{2}{k + 1} < \sum_{i=1}^{n-1} \sum_{k=1}^{n} \frac{2}{k} = 2(n-1) H_n
$$

where $H_n = \sum_{k=1}^{n} 1/k = \ln n + O(1)$ is the $n$-th harmonic number. Thus

$$
E[X] = 2n \ln n + O(n) = O(n \log n)
$$

!!! tip "Why This Analysis Works"
    The indicator variable technique avoids solving a recurrence entirely. Instead of reasoning about random partitions and recursive subproblem sizes, we directly count the probability that each pair of elements is compared. Linearity of expectation handles all dependencies automatically.

## Concentration Around the Mean

The expected comparison count is $\Theta(n \log n)$, but how likely is a large deviation? Using a martingale argument or careful variance analysis, one can show:

$$
\Pr[X > c \cdot n \log n] \leq n^{-\alpha}
$$

for appropriate constants $c$ and $\alpha$. The running time concentrates sharply around its expectation, making randomized quicksort reliable in practice.

## Implementation

```python
"""
Randomized quicksort with random pivot selection.

Demonstrates the Las Vegas guarantee: always correct,
with O(n log n) expected comparisons.
"""

import random

# === Partition ===

def partition(arr, lo, hi):
    """Lomuto partition around arr[hi]."""
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i

# === Randomized Quicksort ===

def randomized_quicksort(arr, lo, hi):
    """Sort arr[lo..hi] using a uniformly random pivot."""
    if lo < hi:
        pivot_idx = random.randint(lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        mid = partition(arr, lo, hi)
        randomized_quicksort(arr, lo, mid - 1)
        randomized_quicksort(arr, mid + 1, hi)

# === Main ===

if __name__ == "__main__":
    data = [3, 6, 8, 10, 1, 2, 1]
    randomized_quicksort(data, 0, len(data) - 1)
    print(data)
```

**Output:**
```
[1, 1, 2, 3, 6, 8, 10]
```

## Worst-Case Probability

Although the worst case is $O(n^2)$, the probability of approaching it is extremely small. The probability that randomized quicksort makes more than $cn \log n$ comparisons decreases exponentially in $c$, making $O(n^2)$ behavior practically impossible on any input.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.
