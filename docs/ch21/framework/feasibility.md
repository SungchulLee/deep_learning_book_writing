# Feasibility Check

In the backtracking template, the `is_valid` function is called at every node of
the state space tree before extending the partial solution.  This **feasibility
check** is the single most important factor in determining how much of the tree the
algorithm actually visits.  A fast, tight feasibility check prunes large subtrees
early; a slow or loose one forces the search to explore nodes that can never lead to
a solution.  This page examines how to design feasibility checks, analyzes their
cost, and shows how the same abstract idea specializes to several classic problems.

## Formal Definition

Given a problem with $n$ decisions, let $S_k = (x_1, x_2, \ldots, x_k)$ denote the
partial solution after $k$ decisions.  A **feasibility function**

$$
\text{feasible}: S_k \to \{\text{True},\, \text{False}\}
$$

returns True exactly when $S_k$ can be extended to at least one complete solution
$S_n$ that satisfies all problem constraints.

Computing the exact feasibility function is often as hard as solving the problem
itself.  In practice we use a **necessary condition** — a predicate that returns
True whenever $S_k$ is extensible, but may also return True for some non-extensible
partial solutions.  The tighter this necessary condition approximates the true
feasibility function, the more nodes are pruned.

!!! info "Necessary vs. sufficient conditions"

    A necessary condition that is too weak (always returns True) provides no
    pruning.  A condition that is too strong (sometimes returns False for
    extensible partial solutions) is incorrect — it causes the algorithm to miss
    valid solutions.  The goal is a necessary condition that is as close to
    sufficient as possible while remaining efficient to evaluate.

## Design Principles

### Incremental Evaluation

Because the feasibility check is called at every node, its running time multiplies
the total node count.  The check should therefore be **incremental**: when decision
$x_k$ is added to $S_{k-1}$, only evaluate constraints involving $x_k$, not all
constraints from scratch.

| Approach | Cost per node | Total for $N$ nodes |
|----------|--------------|---------------------|
| Full constraint check | $O(k)$ or $O(k^2)$ | $O(N \cdot n^2)$ |
| Incremental check | $O(1)$ or $O(k)$ | $O(N \cdot n)$ or $O(N)$ |

The incremental approach requires maintaining auxiliary data structures that are
updated in `make_move` and restored in `undo_move`.

### Auxiliary Data Structures

A well-designed feasibility check often relies on auxiliary state that summarizes
constraint information:

- **Conflict counters**: an integer that counts how many constraints are violated.
  If the counter exceeds zero, the partial solution is infeasible.
- **Availability sets**: a set (or bitmask) of values still legal for each
  remaining decision.  If any set becomes empty, the partial solution is infeasible.
- **Projection arrays**: row, column, or diagonal markers for grid-based problems
  (e.g., N-Queens).

These structures must be updated in $O(1)$ during `make_move` and `undo_move` so
that the per-node feasibility check stays cheap.

## Problem-Specific Feasibility Checks

### N-Queens

**Problem.** Place $n$ queens on an $n \times n$ board so that no two queens share a
row, column, or diagonal.

**Decision.** Place one queen per row.  Decision $k$ selects the column $c_k \in
\{0, 1, \ldots, n-1\}$ for the queen in row $k$.

**Feasibility check.** After placing the queen in row $k$ at column $c_k$, check:

1. **Column conflict**: is $c_k$ already used by a queen in rows $0, \ldots, k-1$?
2. **Diagonal conflict**: does $|c_k - c_j| = |k - j|$ for any $j < k$?

With three Boolean arrays — `col_used[c]`, `diag1_used[k - c + n - 1]`, and
`diag2_used[k + c]` — each check runs in $O(1)$.

### Graph Coloring

**Problem.** Assign one of $m$ colors to each of $n$ vertices so that no two
adjacent vertices share the same color.

**Feasibility check.** After assigning color $c$ to vertex $v_k$, iterate over
$v_k$'s neighbors.  If any already-colored neighbor has color $c$, the assignment
is infeasible.

With an adjacency list, this check runs in $O(\deg(v_k))$.  To make it $O(1)$,
maintain a per-vertex set of forbidden colors; when $v_k$ receives color $c$, add $c$
to the forbidden set of every uncolored neighbor.

### Subset Sum

**Problem.** Given a set of positive integers $\{a_1, \ldots, a_n\}$ and a target
$T$, find a subset whose elements sum to exactly $T$.

**Feasibility check.** Let $\text{sum}_k$ be the running sum after $k$ decisions.
Two conditions enable pruning:

1. **Over-target**: $\text{sum}_k > T$ (prune — adding more positive integers only
   increases the sum).
2. **Under-target**: $\text{sum}_k + \sum_{i=k+1}^{n} a_i < T$ (prune — even
   including all remaining elements cannot reach the target).

Both checks run in $O(1)$ if a precomputed suffix-sum array is maintained.

### Sudoku

**Problem.** Fill a $9 \times 9$ grid so that every row, column, and $3 \times 3$ box
contains the digits 1 through 9 exactly once.

**Feasibility check.** After placing digit $d$ in cell $(r, c)$, check three
constraints:

1. Is $d$ already present in row $r$?
2. Is $d$ already present in column $c$?
3. Is $d$ already present in the $3 \times 3$ box containing $(r, c)$?

With three arrays of bitmasks (`row_used[r]`, `col_used[c]`, `box_used[b]`), each
check is a single bitwise AND in $O(1)$.

## Constraint Propagation

A pure feasibility check asks: "Is the current partial solution still valid?"
**Constraint propagation** goes further by deducing forced values — decisions that
are uniquely determined given the current partial solution.  After each move:

1. Update the availability sets for all remaining decisions.
2. If any availability set has exactly one element, that decision is forced —
   assign it immediately (this is called **unit propagation** in SAT terminology).
3. If any availability set is empty, prune — the partial solution is infeasible.

Constraint propagation transforms the feasibility check from a passive filter into an
active inference engine that can dramatically reduce the branching factor.

!!! tip "Constraint propagation in Sudoku"

    Naked-singles and hidden-singles propagation can solve many Sudoku puzzles
    without any branching at all.  When branching is needed, propagation after
    each guess reduces the remaining possibilities so aggressively that even
    hard puzzles require exploring only a handful of nodes.

## Cost vs. Pruning Power

The feasibility check presents a fundamental trade-off:

$$
\text{Total cost} = (\text{nodes visited}) \times (\text{cost per node})
$$

A more expensive check may prune more nodes, but the per-node overhead grows.  The
optimal balance depends on the problem:

| Check complexity | Pruning power | Best when |
|-----------------|---------------|-----------|
| $O(1)$ | Low–moderate | Branching factor is small; tree is shallow |
| $O(k)$ | Moderate–high | Each decision interacts with all previous ones |
| $O(k^2)$ or higher | High | Tight pruning eliminates exponentially many nodes |

In practice, start with the cheapest correct check and add more sophisticated tests
only if profiling shows that the search still visits too many nodes.

## Reference

- Skiena, *The Algorithm Design Manual*, Chapter 9: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)
