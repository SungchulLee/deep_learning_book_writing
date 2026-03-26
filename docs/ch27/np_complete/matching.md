# 3-Dimensional Matching

While 2D matching (bipartite matching) is solvable in polynomial time, extending to three dimensions makes the problem NP-complete. **3-Dimensional Matching (3DM)** is one of Karp's original 21 NP-complete problems and serves as a key stepping stone in NP-completeness reductions, particularly toward problems like Partition and Subset Sum.

## Problem Definition

!!! tip "Definition: 3-Dimensional Matching"
    Given three disjoint sets $X$, $Y$, $Z$, each of size $n$, and a set of triples $T \subseteq X \times Y \times Z$, a **3-dimensional matching** is a subset $M \subseteq T$ of size $n$ such that every element of $X \cup Y \cup Z$ appears in exactly one triple of $M$.

    The **3DM decision problem** asks: does a perfect 3D matching exist?

## Contrast with 2D Matching

In bipartite matching, we have two sets $X$ and $Y$ and edges $E \subseteq X \times Y$. A perfect matching pairs each element of $X$ with a distinct element of $Y$. This is solvable in $O(\sqrt{n} \cdot m)$ time (Hopcroft-Karp).

The jump from 2 to 3 dimensions fundamentally changes the problem's complexity. The extra dimension prevents the flow-based techniques that make 2D matching tractable.

## NP-Completeness

!!! tip "Theorem (Karp, 1972)"
    3-Dimensional Matching is NP-complete.

**Membership in NP.** A matching $M$ of size $n$ is a certificate. Verification checks that $|M| = n$ and every element of $X \cup Y \cup Z$ appears exactly once, taking $O(n)$ time.

**NP-Hardness: Reduction from 3-SAT.** Given a 3-SAT formula with variables $x_1, \ldots, x_p$ and clauses $C_1, \ldots, C_q$:

### Construction

1. **Variable gadgets.** For each variable $x_i$, create $2q$ triples arranged in a cycle that allows selecting either "true" or "false" triples (but not both). The true triples correspond to setting $x_i = 1$ and the false triples to $x_i = 0$.

2. **Clause gadgets.** For each clause $C_j$, introduce fresh $Y$- and $Z$-elements that must be matched. Connect them to triples corresponding to the clause's literals. At least one literal must be "true" to complete the matching.

3. **Cleanup gadgets.** Add triples to ensure all remaining unmatched elements can be covered.

The total construction is polynomial. A satisfying assignment yields a 3D matching by selecting the corresponding truth/false triples, and vice versa. $\square$

## Related Problems

### Exact Cover by 3-Sets

A closely related problem where we have a universe $U$ with $|U| = 3n$ and a collection $\mathcal{S}$ of 3-element subsets. We ask: can we select $n$ sets from $\mathcal{S}$ that partition $U$?

This is a generalization of 3DM (which requires the partition structure across three dimensions) and is also NP-complete.

### Numerical 3DM

A restricted version where elements have numerical values and triples must sum to a target. This variant is **strongly NP-complete** and is used to prove hardness of problems like 3-Partition.

## Applications and Reductions

3DM serves as a critical intermediate problem in the NP-completeness reduction chain:

$$
\text{3-SAT} \to \text{3DM} \to \text{Partition} \to \text{Subset Sum}
$$

$$
\text{3-SAT} \to \text{3DM} \to \text{Exact Cover} \to \text{Set Cover}
$$

### Reduction to Partition

Given a 3DM instance, construct integers such that a subset summing to half the total exists if and only if a perfect 3D matching exists. This uses the classic technique of encoding set membership in the binary representation of carefully chosen integers.

## Tractable Special Cases

While 3DM is NP-complete in general, restricted versions are tractable:

| Restriction | Complexity | Notes |
|-------------|-----------|-------|
| Bounded occurrences ($\leq 3$ per element) | NP-complete | Still hard |
| Bounded occurrences ($\leq 2$ per element) | P | Reduces to 2-SAT |
| Planar instances | NP-complete | Geometric restrictions do not help |

## Algorithms

### Exact Algorithms

| Algorithm | Time | Notes |
|-----------|------|-------|
| Brute force | $O(\binom{|T|}{n})$ | Try all size-$n$ subsets |
| Inclusion-exclusion | $O(2^n \cdot |T|)$ | Count matching covers |
| Color-coding | $O(2^{3n} \cdot |T|)$ | Randomized FPT |

### Heuristics

For practical instances, **greedy matching** (select triples that cover the most unmatched elements) combined with **backtracking** often finds solutions quickly, though without worst-case guarantees.

??? example "Example: Small 3DM Instance"
    **Sets:** $X = \{x_1, x_2\}$, $Y = \{y_1, y_2\}$, $Z = \{z_1, z_2\}$.

    **Triples:**

    - $t_1 = (x_1, y_1, z_1)$
    - $t_2 = (x_1, y_2, z_2)$
    - $t_3 = (x_2, y_1, z_2)$
    - $t_4 = (x_2, y_2, z_1)$

    **Solution:** Select $\{t_1, t_3\}$: covers $x_1, y_1, z_1$ and $x_2, y_1, z_2$. But $y_1$ appears twice --- invalid.

    Try $\{t_2, t_3\}$: covers $x_1, y_2, z_2$ and $x_2, y_1, z_2$. But $z_2$ appears twice --- invalid.

    Try $\{t_1, t_4\}$: covers $x_1, y_1, z_1$ and $x_2, y_2, z_1$. But $z_1$ appears twice --- invalid.

    Try $\{t_2, t_4\}$: covers $x_1, y_2, z_2$ and $x_2, y_2, z_1$. But $y_2$ appears twice --- invalid.

    **No perfect matching exists** for this instance. Every pair of triples shares an element.

## Reference

- Karp, R. M. (1972). Reducibility among combinatorial problems. In *Complexity of Computer Computations*, Plenum Press.
- Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability*. W. H. Freeman.
- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
