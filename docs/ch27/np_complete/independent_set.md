# Independent Set

An **independent set** in a graph is a set of vertices with no edges between them. Finding the largest independent set is NP-hard, and the problem is closely related to Clique and Vertex Cover through graph complementation. Understanding these relationships reveals how NP-completeness reductions form a web of equivalent problems.

## Problem Definition

!!! tip "Definition: Independent Set"
    Given an undirected graph $G = (V, E)$, an **independent set** is a subset $S \subseteq V$ such that no two vertices in $S$ are adjacent:

    $$
    \forall\, u, v \in S : (u, v) \notin E
    $$

    The **Maximum Independent Set (MIS)** problem asks for the largest such $S$. The decision version asks: does $G$ have an independent set of size $\geq k$?

The **independence number** $\alpha(G)$ is the size of a maximum independent set.

## Relationship to Clique and Vertex Cover

Three problems are tightly connected through complementation:

**Complement graph.** The complement $\bar{G}$ has the same vertex set as $G$ but edge $(u,v) \in \bar{G}$ if and only if $(u,v) \notin G$.

!!! tip "Theorem: IS-Clique Equivalence"
    $S$ is an independent set in $G$ if and only if $S$ is a clique in $\bar{G}$.

**Proof.** $S$ is independent in $G$ means no edge of $G$ connects vertices in $S$. In $\bar{G}$, all pairs in $S$ are edges, making $S$ a clique. $\square$

!!! tip "Theorem: IS-Vertex Cover Complement"
    $S$ is an independent set in $G$ if and only if $V \setminus S$ is a vertex cover in $G$.

**Proof.** If $S$ is independent, then for every edge $(u,v) \in E$, at least one of $u, v$ is not in $S$ --- so at least one is in $V \setminus S$. Hence $V \setminus S$ covers every edge. The converse is identical. $\square$

**Corollary:** $\alpha(G) + \tau(G) = |V|$, where $\tau(G)$ is the minimum vertex cover size.

## NP-Completeness

!!! tip "Theorem"
    Independent Set is NP-complete.

**Membership in NP.** The set $S$ is a certificate, verifiable in $O(|S|^2)$ time by checking no pair is an edge.

**NP-Hardness via Clique.** Since Independent Set in $G$ is equivalent to Clique in $\bar{G}$, and Clique is NP-complete (by reduction from 3-SAT), Independent Set is NP-hard.

### Direct Reduction from 3-SAT

For a direct proof, reduce from 3-SAT. Given formula $\phi$ with $m$ clauses, each with 3 literals:

1. For each clause $C_j = (\ell_1 \lor \ell_2 \lor \ell_3)$, create 3 vertices (one per literal).
2. **Clause edges:** Connect all 3 vertices within each clause (forming a triangle).
3. **Conflict edges:** Connect $x_i$ in one clause to $\bar{x}_i$ in any other clause.
4. Set $k = m$.

**Correctness:**

- ($\Rightarrow$) If $\phi$ is satisfiable, pick one true literal per clause. The $m$ chosen vertices form an independent set: clause edges prevent picking two from the same clause, conflict edges prevent picking a variable and its negation.
- ($\Leftarrow$) An independent set of size $m$ must pick exactly one vertex per clause (triangle constraint). No conflicts means the assignment is consistent. $\square$

## Inapproximability

Independent Set is one of the hardest problems to approximate:

!!! warning "Theorem (Zuckerman, 2007)"
    For any $\epsilon > 0$, it is NP-hard to approximate MIS within a factor of $n^{1-\epsilon}$.

This means no polynomial-time algorithm can find an independent set whose size is within a factor of $n^{0.99}$ of optimal. Even distinguishing whether $\alpha(G) = 1$ or $\alpha(G) \geq n^{0.01}$ is hard.

## Special Graph Classes

| Graph Class | MIS Complexity | Notes |
|-------------|---------------|-------|
| Bipartite | P | Via Konig's theorem: $\alpha = n - \nu$ (matching number) |
| Planar | NP-hard | But PTAS exists |
| Trees | P | DP on the tree structure |
| Interval | P | Greedy by right endpoints |
| Perfect | P | $\alpha = \theta$ by definition, computable via SDP |
| Chordal | P | Via perfect elimination ordering |

## Algorithms

### Exact Algorithms

| Algorithm | Time | Technique |
|-----------|------|-----------|
| Brute force | $O(2^n \cdot n)$ | Try all subsets |
| Bron-Kerbosch (for complement) | $O(3^{n/3})$ | Maximal clique enumeration |
| Measure and conquer | $O(1.1996^n)$ | Branch on low-degree vertices |

### Greedy Heuristic

Repeatedly add the vertex with minimum degree, then remove it and its neighbors. This gives no approximation guarantee in general but produces a maximal independent set.

??? example "Example: Independent Set from 3-SAT"
    **Formula:** $(x_1 \lor x_2 \lor x_3) \land (\bar{x}_1 \lor \bar{x}_2 \lor x_3) \land (x_1 \lor x_2 \lor \bar{x}_3)$.

    **Graph construction:** 9 vertices (3 per clause), 3 clause triangles, and conflict edges connecting $x_1$-clause1 to $\bar{x}_1$-clause2, $x_2$-clause1 to $\bar{x}_2$-clause2, etc.

    **Setting** $x_1 = T, x_2 = T, x_3 = T$: pick $x_1$ from clause 1, $x_3$ from clause 2, $x_1$ from clause 3. But $x_1$ appears twice --- pick $x_2$ from clause 3 instead.

    **Independent set:** $\{x_1^{(1)}, x_3^{(2)}, x_2^{(3)}\}$, size 3 $= m$. No clause edges (different clauses), no conflict edges (no variable and its negation). Confirms satisfiability.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
- Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability*. W. H. Freeman.
