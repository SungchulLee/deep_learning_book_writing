# Subset Sum

The subset sum problem asks whether a given set of integers contains a subset whose
elements sum to a specified target.  It is one of Karp's 21 NP-complete problems
and appears frequently as a sub-problem in scheduling, cryptography, and resource
allocation.  Backtracking solves it by making an include-or-exclude decision for
each element, with two powerful pruning rules that dramatically reduce the search
space compared to brute-force enumeration of all $2^n$ subsets.

## Problem Statement

**Input.** A set of $n$ positive integers $S = \{a_1, a_2, \ldots, a_n\}$ and a
target value $T > 0$.

**Output.** A subset $A \subseteq S$ such that $\sum_{a \in A} a = T$, or a report
that no such subset exists.

!!! info "Positive integers assumption"

    Restricting to positive integers enables the "over-target" pruning rule below.
    The general problem with arbitrary (possibly negative) integers is also
    NP-complete, but the pruning analysis changes.

## Backtracking Formulation

### State Space Tree

- **Decision $i$** ($i = 1, \ldots, n$): include $a_i$ in the subset
  ($x_i = 1$) or exclude it ($x_i = 0$).
- **Branching factor**: 2 at every level.
- **Full tree**: $2^n$ leaves, each corresponding to a distinct subset.

### Feasibility Checks (Pruning)

Let $\text{sum}_k = \sum_{i=1}^{k} x_i \, a_i$ be the running sum after the first
$k$ decisions, and let $\text{remaining}_k = \sum_{i=k+1}^{n} a_i$ be the sum of
elements not yet decided.

Two pruning conditions apply:

1. **Over-target**: if $\text{sum}_k > T$, prune.  Adding more positive integers
   can only increase the sum further.

2. **Under-target**: if $\text{sum}_k + \text{remaining}_k < T$, prune.  Even
   including all remaining elements cannot reach the target.

Both conditions are evaluated in $O(1)$ when a precomputed suffix-sum array is
available.

### Sorting Heuristic

Sorting the elements in **decreasing order** before searching provides two benefits:

1. The over-target prune triggers earlier because large elements push the running
   sum past $T$ quickly.
2. The under-target prune also triggers earlier because the suffix sums decrease
   faster.

## Algorithm

```
SUBSET_SUM(i, current_sum, target, suffix_sum):
    if current_sum == target:
        report solution
        return True

    if i == n:
        return False

    // Pruning
    if current_sum > target:
        return False                   // over-target
    if current_sum + suffix_sum[i] < target:
        return False                   // under-target

    // Include a[i]
    if SUBSET_SUM(i + 1, current_sum + a[i], target, suffix_sum):
        return True

    // Exclude a[i]
    if SUBSET_SUM(i + 1, current_sum, target, suffix_sum):
        return True

    return False
```

## Python Implementation

```python
"""
Subset sum solver using backtracking with over-target and under-target pruning.

Given a set of positive integers and a target, finds a subset that sums
to the target (or reports that none exists).
"""


# === Solver ===================================================================

def subset_sum(numbers, target):
    """Find a subset of *numbers* that sums to *target*.

    Returns the subset as a list, or None if no solution exists.
    Elements are assumed to be positive integers.
    """
    nums = sorted(numbers, reverse=True)
    n = len(nums)

    # Precompute suffix sums for the under-target prune
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + nums[i]

    result = []

    def backtrack(i, current):
        if current == target:
            return True
        if i == n:
            return False
        if current > target:                # over-target prune
            return False
        if current + suffix[i] < target:    # under-target prune
            return False

        # Include nums[i]
        result.append(nums[i])
        if backtrack(i + 1, current + nums[i]):
            return True
        result.pop()

        # Exclude nums[i]
        if backtrack(i + 1, current):
            return True

        return False

    if backtrack(0, 0):
        return result
    return None


# === Find all solutions ======================================================

def subset_sum_all(numbers, target):
    """Find all subsets of *numbers* that sum to *target*."""
    nums = sorted(numbers, reverse=True)
    n = len(nums)

    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + nums[i]

    solutions = []
    current_subset = []

    def backtrack(i, current):
        if current == target:
            solutions.append(current_subset[:])
            return                      # do not return early — find all
        if i == n:
            return
        if current > target:
            return
        if current + suffix[i] < target:
            return

        # Include nums[i]
        current_subset.append(nums[i])
        backtrack(i + 1, current + nums[i])
        current_subset.pop()

        # Exclude nums[i]
        backtrack(i + 1, current)

    backtrack(0, 0)
    return solutions


# === Main =====================================================================

if __name__ == "__main__":
    numbers = [3, 7, 1, 8, 4, 12, 5]
    target = 15

    print(f"Numbers: {numbers}")
    print(f"Target:  {target}\n")

    result = subset_sum(numbers, target)
    if result is not None:
        print(f"One solution: {result}  (sum = {sum(result)})")
    else:
        print("No solution exists.")

    all_results = subset_sum_all(numbers, target)
    print(f"\nAll solutions ({len(all_results)} total):")
    for s in all_results:
        print(f"  {s}  (sum = {sum(s)})")
```

**Output:**
```
Numbers: [3, 7, 1, 8, 4, 12, 5]
Target:  15

One solution: [12, 3]  (sum = 15)

All solutions (5 total):
  [12, 3]  (sum = 15)
  [8, 7]  (sum = 15)
  [8, 4, 3]  (sum = 15)
  [7, 5, 3]  (sum = 15)
  [7, 4, 3, 1]  (sum = 15)
```

## Complexity Analysis

**Time complexity.** In the worst case, both pruning rules fail to eliminate any
branches, and the algorithm visits all $2^n$ leaves:

$$
T(n) = O(2^n)
$$

With the sorting heuristic and both pruning rules active, the practical running
time is much lower for most instances.  However, no polynomial-time algorithm is
known (the problem is NP-complete).

**Space complexity.** The recursion depth is $n$, and the suffix-sum array uses
$O(n)$ space.  Total space is $O(n)$.

## Comparison with Dynamic Programming

When the target $T$ is not too large, the subset sum problem admits a
pseudo-polynomial-time dynamic programming solution with time $O(nT)$ and space
$O(T)$.  The backtracking approach is preferable when:

- $T$ is very large (the DP table would be too big).
- Only one solution is needed (backtracking can stop after finding the first).
- The pruning rules are effective (many branches are cut early).

| Method | Time | Space | Best when |
|--------|------|-------|-----------|
| Backtracking | $O(2^n)$ worst | $O(n)$ | Small $n$, large $T$, strong pruning |
| DP | $O(nT)$ | $O(T)$ | Moderate $n$ and $T$ |

## Reference

- Karp, "Reducibility among Combinatorial Problems," 1972
- Skiena, *The Algorithm Design Manual*, Chapter 9: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)
