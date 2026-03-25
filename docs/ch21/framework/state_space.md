# State Space Tree

When solving a combinatorial problem, we must systematically explore all possible
configurations to find a valid or optimal solution.  A **state space tree** organizes
this exploration: each node represents a partial solution built from a sequence of
decisions, and each edge corresponds to extending that partial solution by one more
choice.  Understanding state space trees is the first step toward backtracking,
branch-and-bound, and other exhaustive search strategies covered in this chapter.

## Definitions

A **state space tree** is a rooted tree that encodes every possible sequence of
decisions for a given problem instance.

- **Root**: the empty partial solution (no decisions made yet).
- **Internal node**: a partial solution obtained after $k$ of $n$ total decisions.
- **Edge**: a single decision that extends a partial solution by one step.
- **Leaf (solution node)**: a complete assignment of all $n$ decisions.  A leaf is
  **feasible** if it satisfies all problem constraints, and **infeasible** otherwise.

For a problem with $n$ decisions where decision $i$ has $d_i$ possible values, the
state space tree has at most

$$
\prod_{i=1}^{n} d_i
$$

leaves, and the total number of nodes (including internal nodes) can be even larger.

??? example "4-Queens placement"

    Place 4 queens on a $4 \times 4$ board so that no two attack each other.  Each
    decision $i$ ($i = 1, 2, 3, 4$) selects the column for the queen in row $i$.
    Each decision has $d_i = 4$ choices, so the full tree has $4^4 = 256$ leaves —
    most of which are infeasible.

## Nodes and Levels

A node at **level** (depth) $k$ represents a partial solution with $k$ decisions
already made.  At each level the branching factor equals the number of remaining
choices for the next decision.

| Level | Decisions made | Remaining decisions | Node meaning |
|-------|---------------|---------------------|--------------|
| 0     | 0             | $n$                 | Empty (root) |
| $k$   | $k$           | $n - k$             | Partial solution |
| $n$   | $n$           | 0                   | Complete solution (leaf) |

Two important node categories drive search algorithms:

- **Live node**: a node that has been generated but whose children have not yet
  been fully explored.
- **Dead node**: a node whose children have all been explored, or a node that has
  been **pruned** (determined to lead only to infeasible or suboptimal solutions).

## Traversal Strategies

Different algorithms explore the state space tree using different traversal orders.
The choice of traversal strategy determines which nodes are expanded first and how
the set of live nodes is managed.

| Strategy | Data structure | Expansion order | Algorithm family |
|----------|---------------|-----------------|------------------|
| BFS      | Queue         | Level by level  | Breadth-first search |
| DFS      | Stack         | Deepest node first | Backtracking |
| Best-first | Priority queue | Most promising first | Branch-and-bound |

### Backtracking as DFS with Pruning

Backtracking performs a **depth-first traversal** of the state space tree but skips
(prunes) subtrees that cannot lead to a valid solution.  Rather than generating the
entire tree and then checking solutions, backtracking checks constraints at every
internal node:

1. **Extend**: choose the next decision value and move to a child node.
2. **Check feasibility**: does the partial solution still satisfy all constraints?
3. **Prune**: if infeasible, mark the node dead and do not generate its children.
4. **Backtrack**: return to the parent and try the next sibling.

This DFS-plus-pruning approach transforms an exhaustive search with exponential cost
into a practical algorithm whenever the pruning eliminates large subtrees early.

### Branch-and-Bound as Best-First Search

Branch-and-bound extends the idea of pruning by attaching a **bound** (an optimistic
estimate of the best objective value achievable from a node) to each live node.
Nodes are expanded in order of their bound value using a priority queue:

1. **Branch**: split a live node into children.
2. **Bound**: compute an upper (or lower) bound for each child.
3. **Prune**: discard any child whose bound is worse than the best known solution.
4. **Select**: pick the live node with the best bound for the next expansion.

## Implicit vs. Explicit Trees

The state space tree is almost never built in full.  Instead, it is generated
**implicitly** — nodes are created on the fly as the search proceeds and discarded
after backtracking.  Only the path from the root to the current node (of length at
most $n$) needs to reside in memory at any time, giving a space complexity of
$O(n)$ for DFS-based approaches.

An **explicit** tree, by contrast, stores all generated nodes simultaneously.  BFS
and best-first search may require explicit storage of all live nodes, leading to
space complexity that can be exponential in the worst case.

!!! tip "Memory trade-off"

    DFS (backtracking) uses $O(n)$ space but may miss the optimal node ordering.
    Best-first search finds optimal solutions sooner but can require $O(b^n)$ space,
    where $b$ is the branching factor.  Many practical algorithms combine elements
    of both — for example, iterative deepening DFS achieves BFS-level completeness
    with DFS-level memory.

## Size and Complexity

The size of the state space tree determines the worst-case running time of any
exhaustive search.  Common problem structures lead to well-known tree sizes:

| Problem | Decisions | Branching | Tree leaves | Tree nodes |
|---------|-----------|-----------|-------------|------------|
| Binary strings of length $n$ | $n$ | 2 | $2^n$ | $2^{n+1} - 1$ |
| Permutations of $n$ elements | $n$ | $n, n{-}1, \ldots, 1$ | $n!$ | $\sum_{k=0}^{n} \frac{n!}{k!}$ |
| Subsets of $n$ elements | $n$ | 2 | $2^n$ | $2^{n+1} - 1$ |
| $m$-coloring of $n$ vertices | $n$ | $m$ | $m^n$ | $\frac{m^{n+1} - 1}{m - 1}$ |

The exponential (or factorial) growth underscores why pruning techniques —
backtracking and branch-and-bound — are essential for making exhaustive search
practical.

## Reference

- Abdul Bari, "Introduction to Backtracking — Brute Force Approach,"
  [YouTube](https://www.youtube.com/watch?v=DKCbsiDBN6c&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=63)
