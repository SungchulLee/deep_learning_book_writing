# Pattern Recognition Guide

Most algorithmic problems fall into a small number of recurring patterns. Recognizing
which pattern applies to a given problem is the most important skill for interviews and
contests, because it immediately narrows the solution space from hundreds of algorithms
to two or three candidates.

## Two Pointers

Use two pointers when the problem involves a sorted array or a linked list and asks
for a pair, subarray, or partition.

| Variant | Setup | Typical Complexity | Example Problems |
|---|---|---|---|
| Opposite ends | `lo = 0, hi = n-1` | $O(n)$ | Two Sum (sorted), Container With Most Water |
| Same direction | `slow, fast` | $O(n)$ | Remove duplicates, linked list cycle |
| Partitioning | `lo, hi` swap | $O(n)$ | Dutch National Flag, Quick Sort partition |

**When to use**: the array is sorted (or can be sorted without losing information),
and you need to find elements satisfying a condition involving their sum, difference,
or relative position.

## Sliding Window

Sliding window maintains a contiguous subarray or substring that expands and contracts
to satisfy a constraint.

| Variant | When to Use | Complexity |
|---|---|---|
| Fixed size $k$ | Maximum/minimum of all windows of size $k$ | $O(n)$ |
| Variable size | Shortest/longest subarray with a property | $O(n)$ |
| With hash map | Substring with character frequency constraints | $O(n)$ |

**Key insight**: if adding an element to the right maintains or violates the
constraint monotonically, sliding window works. If the constraint can be violated and
then restored by removing from the left, the window is valid.

## Binary Search

Binary search applies whenever the answer lies in a sorted or monotone search space.

| Variant | Setup | Complexity | Example |
|---|---|---|---|
| Index search | Sorted array, find target | $O(\log n)$ | Lower/upper bound |
| Answer search | Monotone predicate on answer space | $O(\log R \cdot f(n))$ | Minimum capacity, maximum distance |
| Fractional | Real-valued answer | $O(\log(R/\epsilon) \cdot f(n))$ | Geometric optimization |

**When to use**: the problem asks "find the minimum $x$ such that condition $C(x)$
holds" and $C$ is monotone (once true, stays true for all larger $x$).

## Dynamic Programming

DP applies when a problem has optimal substructure and overlapping subproblems.

| Pattern | State Space | Time | Example |
|---|---|---|---|
| Linear DP | $dp[i]$ | $O(n)$ to $O(n^2)$ | House Robber, LIS |
| Grid DP | $dp[i][j]$ | $O(mn)$ | Unique Paths, Edit Distance |
| Interval DP | $dp[i][j]$ for range $[i, j]$ | $O(n^3)$ | Matrix Chain, Burst Balloons |
| Knapsack | $dp[i][w]$ | $O(nW)$ | 0/1 Knapsack, Coin Change |
| Bitmask DP | $dp[\text{mask}]$ | $O(2^n \cdot n)$ | TSP, Assignment |
| Tree DP | $dp[\text{node}]$ | $O(n)$ | Max Independent Set |

**When to use**: the brute-force solution recomputes the same subproblems, and the
problem asks for an optimal value (min, max, count).

## Greedy

Greedy algorithms make locally optimal choices that lead to a globally optimal solution.

| Signal | Strategy | Example |
|---|---|---|
| Interval scheduling | Sort by end time, pick non-overlapping | Activity Selection |
| Huffman-style merging | Always merge the two smallest | Huffman Coding |
| Exchange argument | Swapping any two elements does not improve | Fractional Knapsack |
| Matroid structure | Greedy on a matroid yields optimal | MST (Kruskal's) |

**When to use**: a greedy choice property can be proven (exchange argument or matroid
theory), and the problem has optimal substructure.

## Backtracking

Backtracking systematically explores all candidates and prunes branches that cannot
lead to a valid solution.

| Signal | Approach | Complexity |
|---|---|---|
| "Find all" / "list all" | Generate and test with pruning | Exponential |
| Constraint satisfaction | Place, check, backtrack | $O(k^n)$ in the worst case |
| Optimization with constraints | Branch and bound | Exponential with pruning |

**When to use**: the problem asks for all valid configurations, the search space is
small ($n \le 20$), or effective pruning drastically reduces the actual search.

## Graph Patterns

| Problem Type | Algorithm | Complexity |
|---|---|---|
| Shortest path (unweighted) | BFS | $O(V + E)$ |
| Shortest path (weighted, non-negative) | Dijkstra | $O((V + E) \log V)$ |
| Shortest path (negative weights) | Bellman-Ford | $O(VE)$ |
| Connected components | DFS / Union-Find | $O(V + E)$ |
| Topological order | DFS / Kahn's | $O(V + E)$ |
| Cycle detection | DFS coloring | $O(V + E)$ |
| MST | Kruskal / Prim | $O(E \log V)$ |

## Pattern Selection Flowchart

When facing a new problem, ask these questions in order:

1. **Is the input sorted or can sorting help?** Try two pointers or binary search.
2. **Does it involve contiguous subarrays/substrings?** Try sliding window.
3. **Does it ask for an optimal value with overlapping subproblems?** Try DP.
4. **Can you prove a greedy choice property?** Try greedy.
5. **Does it involve a graph or tree structure?** Try BFS/DFS.
6. **Does it ask for all solutions?** Try backtracking.
7. **None of the above?** Start with brute force, then optimize.

!!! tip "Pattern Combinations"
    Many problems combine patterns. For example, "minimum cost path in a grid"
    combines graph (grid as graph) with DP (optimal substructure). "Longest
    substring without repeating characters" combines sliding window with a hash
    set.

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Skiena, S. *The Algorithm Design Manual*. 3rd ed. Springer, 2020.
