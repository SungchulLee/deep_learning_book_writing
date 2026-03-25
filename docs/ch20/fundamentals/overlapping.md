# Overlapping Subproblems

A recursive algorithm for an optimization problem often solves the same subproblem many times.  When this redundancy is present, the problem has **overlapping subproblems** — the second of two key properties (alongside optimal substructure) that make dynamic programming effective.  Understanding this property explains why naive recursion is exponential while memoization and tabulation reduce the work to polynomial time.

## Definition

A problem has **overlapping subproblems** if a recursive algorithm for the problem solves the same subproblems repeatedly rather than always generating new ones.

This stands in contrast to divide-and-conquer algorithms, where each recursive call works on a disjoint portion of the input:

| Paradigm | Subproblem overlap | Example |
|----------|-------------------|---------|
| Divide and conquer | Subproblems are disjoint | Merge sort splits the array into non-overlapping halves |
| Dynamic programming | Subproblems recur many times | Fibonacci recursion recomputes $F(k)$ exponentially often |

## Fibonacci as a Case Study

The Fibonacci recurrence provides the clearest illustration.  The naive recursive definition is

$$
F(n) = F(n-1) + F(n-2), \quad F(0) = 0, \quad F(1) = 1
$$

Drawing the recursion tree for $F(5)$ reveals massive redundancy:

```
                    F(5)
                /         \
            F(4)           F(3)
           /    \         /    \
        F(3)   F(2)    F(2)   F(1)
       /  \    / \     / \
    F(2) F(1) F(1) F(0) F(1) F(0)
    / \
 F(1) F(0)
```

In this tree, $F(2)$ is computed 3 times, $F(1)$ is computed 5 times, and $F(0)$ is computed 3 times.  The total number of calls grows as $O(\phi^n)$ where $\phi = (1 + \sqrt{5})/2 \approx 1.618$, even though there are only $n + 1$ distinct subproblems: $F(0), F(1), \ldots, F(n)$.

## Counting Distinct Subproblems

The gap between the number of **distinct** subproblems and the number of **total** recursive calls determines how much dynamic programming helps.

| Problem | Distinct subproblems | Naive recursive calls | Speedup from DP |
|---------|--------------------|-----------------------|-----------------|
| Fibonacci | $n + 1$ | $O(\phi^n)$ | Exponential to linear |
| Rod cutting of length $n$ | $n$ | $O(2^n)$ | Exponential to quadratic |
| LCS of strings of length $m, n$ | $O(mn)$ | $O(2^{m+n})$ | Exponential to polynomial |

When the number of distinct subproblems is polynomial but naive recursion makes exponentially many calls, dynamic programming provides an exponential speedup.

## Overlap vs No Overlap

Not every recursive problem benefits from dynamic programming.  Consider binary search: at each step, the algorithm recurses on exactly one half of the array.  No subproblem is ever revisited, so there is no overlap and no benefit from memoization.

Similarly, merge sort divides the array into two halves, recursively sorts each, and merges the results.  Each recursive call operates on a distinct subarray.  The subproblems never overlap, so merge sort is a divide-and-conquer algorithm, not a dynamic programming algorithm.

The key diagnostic question is: **does the recursion tree contain repeated nodes?**  If yes, the problem has overlapping subproblems and is a candidate for dynamic programming.

## How Dynamic Programming Eliminates Redundancy

Once overlapping subproblems are identified, two strategies eliminate the redundant computation:

**Memoization (top-down)** keeps the recursive structure but stores each result in a lookup table.  Before computing a subproblem, the algorithm checks whether the answer is already cached.  Each distinct subproblem is solved exactly once.

**Tabulation (bottom-up)** iterates through all subproblems in a fixed order, filling a table from the smallest subproblems upward.  When a subproblem is needed, its answer is already in the table.

Both approaches reduce the total work from the number of recursive calls to the number of *distinct* subproblems, multiplied by the work per subproblem.

!!! tip "Identifying overlap in practice"
    When analyzing a new recurrence, draw the recursion tree for a small input.  If you see the same function arguments appearing at multiple nodes, the problem has overlapping subproblems.  The number of distinct argument tuples gives the size of the DP table.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
