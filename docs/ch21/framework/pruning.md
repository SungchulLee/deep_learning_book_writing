# Pruning Strategies

A state space tree for a problem with $n$ decisions and branching factor $b$ contains
$O(b^n)$ nodes.  Visiting every node is prohibitively expensive for all but the
smallest instances.  **Pruning** is the technique of recognizing — as early as
possible — that an entire subtree cannot contain a valid or optimal solution, and
skipping it entirely.  Effective pruning is what makes backtracking and
branch-and-bound practical: it preserves correctness while dramatically reducing the
portion of the tree that is actually explored.

## Why Pruning Works

Consider a state space tree of depth $n$ with a uniform branching factor $b$.
Without pruning the search visits

$$
\sum_{k=0}^{n} b^k = \frac{b^{n+1} - 1}{b - 1} = O(b^n)
$$

nodes.  If a pruning rule eliminates a fraction $p$ of children at every node, the
effective branching factor drops to $b' = b(1 - p)$, and the number of visited nodes
falls to $O(b'^n)$.  Because the tree size is exponential in $n$, even a modest
reduction in $b'$ produces an enormous speedup.

??? example "Quantitative impact of pruning"

    Suppose $b = 4$ and $n = 20$.  Without pruning the tree has roughly
    $4^{20} \approx 1.1 \times 10^{12}$ leaves.  If pruning eliminates 50 % of
    branches at each level, the effective branching factor is $b' = 2$ and the
    number of leaves drops to $2^{20} \approx 10^6$ — a million-fold reduction.

## Types of Pruning

### Feasibility Pruning

Feasibility pruning checks whether the current partial solution can be extended to a
**valid** (constraint-satisfying) complete solution.  If not, the subtree rooted at
the current node is skipped.

**Mechanism.** After adding decision $x_k$ to the partial solution
$(x_1, \ldots, x_{k-1})$, evaluate a predicate

$$
\text{feasible}(x_1, \ldots, x_k) \in \{\text{True}, \text{False}\}
$$

If the predicate returns False, prune immediately — do not generate any children.

**Examples:**

- **N-Queens**: after placing a queen in row $k$, check whether it attacks any
  previously placed queen.  If it does, prune.
- **Graph coloring**: after assigning color $c$ to vertex $v_k$, check whether any
  neighbor of $v_k$ already has color $c$.  If so, prune.
- **Subset sum**: if the running sum already exceeds the target $T$, prune.

### Optimality Pruning (Bounding)

Optimality pruning applies to **optimization** problems.  It computes an optimistic
bound on the best objective value achievable from the current node and compares it to
the best complete solution found so far (the **incumbent**).

**Mechanism.** Let $\text{bound}(x_1, \ldots, x_k)$ be an upper bound on the
objective for a maximization problem (or a lower bound for minimization).  Let
$z^*$ be the incumbent value.  Prune when

$$
\text{bound}(x_1, \ldots, x_k) \leq z^*
$$

because no descendant of this node can improve the incumbent.

The quality of the bound determines how much of the tree is eliminated.  A tighter
bound prunes more aggressively but is usually more expensive to compute.  The
best pruning strategies balance bound tightness against computation cost.

!!! tip "Bound computation trade-off"

    A bound that takes $O(1)$ to compute but prunes 20 % of nodes can be more
    effective than a bound that takes $O(n^2)$ to compute but prunes 50 %, because
    the cheaper bound is evaluated at many more nodes.

### Symmetry Pruning

Many combinatorial problems have **symmetries** — transformations (rotations,
reflections, relabelings) that map one solution to another.  Symmetry pruning
eliminates redundant branches by enforcing a canonical ordering:

- **Subsets**: require elements to be chosen in increasing order to avoid generating
  the same subset in different orderings.
- **N-Queens**: fix the first queen's column to be in the left half of the board,
  eliminating mirror-image solutions.
- **Graph coloring**: assign colors to the first vertex in a fixed order (color 1
  first, then color 2, etc.) to break color-permutation symmetry.

Symmetry pruning can reduce the search space by a factor equal to the size of the
symmetry group — up to $n!$ for permutation symmetries.

### Dominance Pruning

A partial solution $A$ **dominates** partial solution $B$ if every completion of $B$
that is optimal can be matched or improved by a corresponding completion of $A$.
When dominance can be detected efficiently, the entire subtree rooted at $B$ is
pruned.

**Example.** In the 0/1 Knapsack problem, suppose two partial solutions have
considered the same items $\{1, \ldots, k\}$ and have the same remaining capacity,
but solution $A$ has a higher total value.  Then $A$ dominates $B$, and $B$'s subtree
can be pruned.

## Pruning Order Matters

The effectiveness of pruning depends on the **order** in which decisions are made and
the **order** in which candidate values are tried:

| Strategy | Effect |
|----------|--------|
| **Most-constrained variable first** | Choose the decision with the fewest remaining legal values.  This maximizes the chance of early failure and deeper pruning. |
| **Least-constraining value first** | Among candidate values, try the one that rules out the fewest options for future decisions.  This increases the chance of finding a solution quickly. |
| **Best-first ordering** | Sort candidates by their bounding value so that the best incumbent is found early, enabling stronger optimality pruning. |

These heuristics do not change the worst-case complexity, but they can reduce the
average-case search effort by orders of magnitude.

## Measuring Pruning Effectiveness

Two metrics quantify how well pruning performs on a given instance:

**Pruning ratio.** The fraction of the full state space tree that is never generated:

$$
r = 1 - \frac{\text{nodes visited with pruning}}{\text{nodes in full tree}}
$$

A ratio close to 1 indicates effective pruning.

**Effective branching factor.** If the search visits $N$ nodes for a tree of depth
$n$, the effective branching factor $b^*$ satisfies

$$
N = \sum_{k=0}^{n} (b^*)^k \approx (b^*)^n
$$

so $b^* \approx N^{1/n}$.  Comparing $b^*$ to the nominal branching factor $b$ shows
how much pruning reduces the search.

## Summary

| Pruning type | Applicable to | Prunes when |
|-------------|--------------|-------------|
| Feasibility | Constraint satisfaction | Partial solution violates a constraint |
| Optimality (bounding) | Optimization | Bound is worse than incumbent |
| Symmetry | Problems with structural symmetries | Equivalent solutions would be generated |
| Dominance | Problems with comparable partial solutions | One partial solution is provably no worse |

All four pruning types can be combined.  In practice, the strongest backtracking
solvers apply feasibility pruning at every node, optimality pruning when an
objective function exists, symmetry breaking to remove redundant branches, and
dominance checks when the partial-solution comparison is cheap.

## Reference

- Skiena, *The Algorithm Design Manual*, Chapter 9: Combinatorial Search,
  [algorist.com](https://www.algorist.com/)
