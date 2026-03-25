# Generating Permutations

Many combinatorial problems — scheduling, assignment, routing — require examining
all possible orderings of a set of elements.  A **permutation** of $n$ elements is a
bijection from $\{1, \ldots, n\}$ to itself, and there are $n!$ such bijections.
Backtracking generates all $n!$ permutations systematically by building each
ordering one position at a time, ensuring no element is reused.

## Problem Statement

**Input.** A set of $n$ distinct elements $\{a_1, a_2, \ldots, a_n\}$.

**Output.** All $n!$ permutations of the set.

## Backtracking Formulation

### State Space Tree

- **Decision $k$** ($k = 1, \ldots, n$): choose which of the remaining unused
  elements occupies position $k$.
- **Branching factor**: $n - k + 1$ at level $k$ (the number of unused elements).
- **Full tree**: the tree has $n!$ leaves, one per permutation.

### Feasibility Check

The only constraint is that each element appears exactly once.  The feasibility
check reduces to: "Has element $a_i$ already been used in positions $1, \ldots, k-1$?"
A Boolean array `used[i]` answers this in $O(1)$.

## Approach 1 --- Selection-Based

At each level $k$, iterate over all $n$ elements and select those not yet used.

```
PERMUTATIONS(perm, k, n):
    if k == n:
        output perm
        return

    for i = 0 to n - 1:
        if not used[i]:
            perm[k] = a[i]
            used[i] = True
            PERMUTATIONS(perm, k + 1, n)
            used[i] = False
```

### Complexity

The algorithm generates exactly $n!$ leaves.  At each internal node, it scans the
`used` array in $O(n)$ to find unused elements.  The total number of internal nodes
is

$$
\sum_{k=0}^{n-1} \frac{n!}{(n - k)!}
$$

and each node does $O(n)$ work, giving a total time of $O(n \cdot n!)$.  The output
itself has size $\Theta(n \cdot n!)$, so the algorithm is output-optimal up to a
constant factor.

## Approach 2 --- Swap-Based (Heap-style)

An alternative avoids the `used` array entirely by swapping elements into position.
At level $k$, swap each of the elements in positions $k, k+1, \ldots, n-1$ into
position $k$, recurse, and swap back.

```
PERMUTE_SWAP(arr, k, n):
    if k == n:
        output arr
        return

    for i = k to n - 1:
        swap(arr[k], arr[i])
        PERMUTE_SWAP(arr, k + 1, n)
        swap(arr[k], arr[i])       // undo
```

This approach uses no auxiliary data structure and permutes the array in place.
The per-node work is $O(1)$ (a single swap), giving $O(n!)$ total work excluding
output.

## Python Implementation

```python
"""
Generate all permutations of a list using backtracking.

Demonstrates both the selection-based and swap-based approaches.
"""


# === Selection-based approach =================================================

def permutations_select(elements):
    """Generate all permutations using a used-array."""
    n = len(elements)
    results = []
    perm = [None] * n
    used = [False] * n

    def backtrack(k):
        if k == n:
            results.append(perm[:])
            return
        for i in range(n):
            if not used[i]:
                perm[k] = elements[i]
                used[i] = True
                backtrack(k + 1)
                used[i] = False

    backtrack(0)
    return results


# === Swap-based approach =====================================================

def permutations_swap(elements):
    """Generate all permutations by swapping elements in place."""
    arr = list(elements)
    n = len(arr)
    results = []

    def backtrack(k):
        if k == n:
            results.append(arr[:])
            return
        for i in range(k, n):
            arr[k], arr[i] = arr[i], arr[k]
            backtrack(k + 1)
            arr[k], arr[i] = arr[i], arr[k]

    backtrack(0)
    return results


# === Main =====================================================================

if __name__ == "__main__":
    elements = [1, 2, 3]

    print("Selection-based permutations:")
    for p in permutations_select(elements):
        print(f"  {p}")

    print(f"\nSwap-based permutations:")
    for p in permutations_swap(elements):
        print(f"  {p}")

    print(f"\nTotal: {len(permutations_select(elements))} permutations of "
          f"{len(elements)} elements")
```

**Output:**
```
Selection-based permutations:
  [1, 2, 3]
  [1, 3, 2]
  [2, 1, 3]
  [2, 3, 1]
  [3, 1, 2]
  [3, 2, 1]

Swap-based permutations:
  [1, 2, 3]
  [1, 3, 2]
  [2, 1, 3]
  [2, 3, 1]
  [3, 2, 1]
  [3, 1, 2]

Total: 6 permutations of 3 elements
```

!!! note "Ordering difference"

    The two approaches produce permutations in different orders.  The
    selection-based method generates them in lexicographic order when the
    elements are initially sorted.  The swap-based method generates a different
    (but still complete) ordering.

## Complexity Analysis

| Metric | Selection-based | Swap-based |
|--------|----------------|------------|
| Leaves | $n!$ | $n!$ |
| Internal nodes | $\sum_{k=0}^{n-1} n!/(n-k)!$ | same |
| Per-node work | $O(n)$ | $O(1)$ |
| Total time | $O(n \cdot n!)$ | $O(n!)$ + $O(n \cdot n!)$ output |
| Space | $O(n)$ | $O(n)$ (recursion stack) |

Both approaches are output-optimal because generating $n!$ permutations of length
$n$ requires $\Omega(n \cdot n!)$ time just to write the output.

## Lexicographic Generation

When only the **next** permutation is needed (rather than all at once), the
next-permutation algorithm generates permutations in lexicographic order in
amortized $O(1)$ time per permutation:

1. Find the largest index $i$ such that $a[i] < a[i + 1]$.  If none exists, the
   current permutation is the last one.
2. Find the largest index $j > i$ such that $a[i] < a[j]$.
3. Swap $a[i]$ and $a[j]$.
4. Reverse the suffix $a[i+1], \ldots, a[n-1]$.

This iterative approach uses $O(1)$ extra space and avoids the recursion overhead
of backtracking.

## Reference

- Skiena, *The Algorithm Design Manual*, Chapter 9: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)
- Sedgewick, "Permutation Generation Methods," *ACM Computing Surveys*, 1977
