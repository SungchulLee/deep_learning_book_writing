# Generating Combinations

While permutations care about the **order** of elements, many problems — feature
selection, committee formation, test-case generation — care only about **which**
elements are chosen.  A $k$-combination of $n$ elements is an unordered subset of
size $k$.  Backtracking generates all $\binom{n}{k}$ combinations by making an
include-or-exclude decision for each element, with a simple constraint that exactly
$k$ elements are selected.

## Problem Statement

**Input.** A set of $n$ distinct elements $\{a_1, a_2, \ldots, a_n\}$ and an
integer $0 \leq k \leq n$.

**Output.** All $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ subsets of size $k$.

## Backtracking Formulation

### Approach 1 --- Include/Exclude Decisions

Process elements $a_1, a_2, \ldots, a_n$ in order.  At element $a_i$, decide
whether to include it in the current subset or skip it.

- **Decision $i$** ($i = 1, \ldots, n$): include or exclude $a_i$.
- **Branching factor**: 2 at every level.
- **Full tree**: $2^n$ leaves (all subsets), of which $\binom{n}{k}$ have exactly
  $k$ elements included.

**Pruning.** Two conditions allow early termination:

1. **Too many chosen**: if the number of included elements already equals $k$,
   exclude all remaining elements (no more branching needed).
2. **Too few remaining**: if the number of included elements plus the number of
   remaining elements is less than $k$, prune — it is impossible to reach $k$.

Formally, let $c$ be the count of included elements after processing $a_1, \ldots,
a_i$.  Prune when $c > k$ or $c + (n - i) < k$.

### Approach 2 --- Forward Selection

Select the $j$-th element of the combination from the elements that come after the
$(j-1)$-th selected element.  This enforces an increasing-index order, which
eliminates duplicate subsets by construction.

```
COMBINATIONS(start, combo, k, n):
    if len(combo) == k:
        output combo
        return

    for i = start to n - (k - len(combo)) + 1:
        combo.append(a[i])
        COMBINATIONS(i + 1, combo, k, n)
        combo.pop()
```

The loop upper bound `n - (k - len(combo)) + 1` prunes branches where not enough
elements remain to fill the combination.

## Python Implementation

```python
"""
Generate all k-combinations of n elements using backtracking.

Demonstrates forward-selection with pruning.
"""


# === Forward-selection approach ===============================================

def combinations(elements, k):
    """Return all k-element subsets of *elements*."""
    n = len(elements)
    results = []
    combo = []

    def backtrack(start):
        if len(combo) == k:
            results.append(combo[:])
            return

        remaining_needed = k - len(combo)
        for i in range(start, n - remaining_needed + 1):
            combo.append(elements[i])
            backtrack(i + 1)
            combo.pop()

    backtrack(0)
    return results


# === Include/exclude approach =================================================

def combinations_binary(elements, k):
    """Return all k-element subsets using include/exclude decisions."""
    n = len(elements)
    results = []
    combo = []

    def backtrack(i):
        if len(combo) == k:
            results.append(combo[:])
            return
        if i == n:
            return
        # Pruning: not enough elements remaining
        if len(combo) + (n - i) < k:
            return

        # Include a[i]
        combo.append(elements[i])
        backtrack(i + 1)
        combo.pop()

        # Exclude a[i]
        backtrack(i + 1)

    backtrack(0)
    return results


# === Main =====================================================================

if __name__ == "__main__":
    elements = [1, 2, 3, 4, 5]
    k = 3

    print(f"All {k}-combinations of {elements}:")
    for c in combinations(elements, k):
        print(f"  {c}")

    print(f"\nTotal: {len(combinations(elements, k))} combinations")
    print(f"Expected: C({len(elements)}, {k}) = "
          f"{len(combinations(elements, k))}")
```

**Output:**
```
All 3-combinations of [1, 2, 3, 4, 5]:
  [1, 2, 3]
  [1, 2, 4]
  [1, 2, 5]
  [1, 3, 4]
  [1, 3, 5]
  [1, 4, 5]
  [2, 3, 4]
  [2, 3, 5]
  [2, 4, 5]
  [3, 4, 5]

Total: 10 combinations
Expected: C(5, 3) = 10
```

## Complexity Analysis

**Time complexity.** The forward-selection approach generates exactly $\binom{n}{k}$
leaves.  At each leaf, copying the combination costs $O(k)$.  The total number of
internal nodes is bounded by

$$
\sum_{j=0}^{k} \binom{n}{j}
$$

and each internal node does $O(1)$ work (one append and one pop).  The overall time
is

$$
T = O\!\left(k \cdot \binom{n}{k}\right)
$$

which is output-optimal since the output itself has size $\Theta\!\left(k \cdot \binom{n}{k}\right)$.

**Space complexity.** The recursion depth is at most $\min(k, n)$ and the combination
buffer holds at most $k$ elements, giving $O(k)$ space beyond the output.

## Generating All Subsets

Setting $k$ to range over $0, 1, \ldots, n$ generates all $2^n$ subsets.  An
equivalent approach uses a single binary decision tree (include/exclude) without
fixing $k$:

```
ALL_SUBSETS(i, subset):
    if i == n:
        output subset
        return

    // Include a[i]
    subset.append(a[i])
    ALL_SUBSETS(i + 1, subset)
    subset.pop()

    // Exclude a[i]
    ALL_SUBSETS(i + 1, subset)
```

This generates all $2^n$ subsets in $O(n \cdot 2^n)$ time and $O(n)$ space.

## Relationship to Permutations

Combinations and permutations are connected by

$$
P(n, k) = k! \cdot \binom{n}{k}
$$

where $P(n, k) = n! / (n - k)!$ is the number of $k$-permutations.  To generate
all $k$-permutations, first generate all $k$-combinations and then permute each
combination.

## Reference

- Skiena, *The Algorithm Design Manual*, Chapter 9: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)
