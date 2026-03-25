# Optimal Substructure

When solving an optimization problem, a natural question arises: can the solution to the whole problem be built from solutions to its parts?  Optimal substructure answers this affirmatively for many classical problems.  A problem exhibits optimal substructure if an optimal solution to the problem contains within it optimal solutions to subproblems.  Together with overlapping subproblems, optimal substructure is one of the two hallmarks that make dynamic programming applicable.

## Definition

A problem has **optimal substructure** if the following property holds:

> An optimal solution to the problem can be constructed from optimal solutions to its subproblems.

More precisely, suppose an optimal solution $S^*$ to a problem of size $n$ makes a choice that leaves one or more subproblems to solve.  If the restriction of $S^*$ to each subproblem is itself an optimal solution to that subproblem, then the problem has optimal substructure.

Formally, let $\text{OPT}(P)$ denote the value of an optimal solution to problem $P$, and suppose $P$ decomposes into subproblems $P_1, P_2, \ldots, P_k$ after an initial decision.  Optimal substructure means

$$
\text{OPT}(P) = f\bigl(\text{OPT}(P_1), \text{OPT}(P_2), \ldots, \text{OPT}(P_k)\bigr)
$$

for some combining function $f$ that depends on the problem structure.

## Proving Optimal Substructure

Establishing optimal substructure typically follows a **cut-and-paste** argument:

1. **Assume** an optimal solution $S^*$ to the original problem.
2. **Identify** the subproblem solutions contained within $S^*$.
3. **Suppose for contradiction** that one of those subproblem solutions is not optimal.
4. **Cut** the suboptimal piece out and **paste** in a truly optimal subproblem solution.
5. **Show** that the resulting solution is strictly better than $S^*$, contradicting its optimality.

This proof pattern applies broadly across DP problems and provides a systematic way to verify that a proposed recurrence is correct.

## Example: Rod Cutting

Consider a rod of length $n$ and a price table $p_i$ for pieces of length $i$.  The goal is to cut the rod into pieces to maximize total revenue.

Suppose the optimal solution makes a first cut of length $i$, leaving a rod of length $n - i$.  The remaining rod must be cut optimally — otherwise we could replace its cutting plan with a better one and increase total revenue, contradicting optimality.

This yields the recurrence

$$
r(n) = \max_{1 \le i \le n} \bigl(p_i + r(n - i)\bigr)
$$

with base case $r(0) = 0$.  The fact that $r(n-i)$ appears in the recurrence is a direct consequence of optimal substructure.

## Example: Shortest Paths

Let $\delta(u, v)$ denote the weight of a shortest path from vertex $u$ to vertex $v$ in a weighted graph.  If vertex $w$ lies on a shortest path from $u$ to $v$, then

$$
\delta(u, v) = \delta(u, w) + \delta(w, v)
$$

The sub-paths from $u$ to $w$ and from $w$ to $v$ must each be shortest paths.  If either sub-path were not shortest, we could substitute a shorter one and obtain a $u$-to-$v$ path with smaller total weight — a contradiction.

!!! warning "Optimal substructure does not hold for all problems"
    The **longest simple path** problem in a general graph does *not* exhibit optimal substructure.  A longest simple path from $u$ to $v$ passing through $w$ does not necessarily consist of a longest simple path from $u$ to $w$ followed by one from $w$ to $v$, because the simplicity constraint (no repeated vertices) creates dependencies between subproblems.

## Recognizing Optimal Substructure

When analyzing a new problem, look for these indicators:

1. **The problem asks for an optimum** (minimum, maximum, longest, shortest, or count).
2. **A choice reduces the problem** to one or more smaller instances of the same type.
3. **Subproblem solutions combine independently** — the choice for one subproblem does not constrain the choices available in another.

When indicator 3 fails — for example, when subproblems share resources or constraints — optimal substructure may not hold, and dynamic programming may not apply directly.

## Relationship to Greedy Algorithms

Both dynamic programming and greedy algorithms exploit optimal substructure.  The difference lies in how they handle the initial choice:

| Aspect | Dynamic Programming | Greedy |
|--------|-------------------|--------|
| Choices considered | All possible choices at each step | A single locally optimal choice |
| Subproblems solved | All subproblems arising from each choice | Only the subproblem after the greedy choice |
| Correctness guarantee | Always correct if optimal substructure holds | Requires an additional greedy-choice property |

Dynamic programming is more general: it examines all choices and picks the best, while greedy algorithms commit to one choice and never reconsider.  Greedy algorithms are faster when applicable, but they require a separate proof that the greedy choice is safe.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
