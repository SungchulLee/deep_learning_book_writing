# Graph Coloring

**Graph coloring** assigns labels (colors) to vertices such that no two adjacent vertices share the same color. The minimum number of colors needed is the **chromatic number** $\chi(G)$. Determining whether $k$ colors suffice ($k$-Coloring) is one of the classic NP-complete problems, with connections to scheduling, register allocation, and map coloring.

## Problem Definition

!!! tip "Definition: k-Coloring"
    Given an undirected graph $G = (V, E)$ and integer $k$, a **proper $k$-coloring** is a function $c : V \to \{1, 2, \ldots, k\}$ such that $c(u) \neq c(v)$ for every edge $(u, v) \in E$.

    The **$k$-Coloring** decision problem asks: does $G$ have a proper $k$-coloring?

The **chromatic number** $\chi(G) = \min\{k : G \text{ has a proper } k\text{-coloring}\}$.

## Easy Cases

- **$k = 1$:** $G$ is 1-colorable if and only if $E = \emptyset$. Trivially decidable.
- **$k = 2$:** $G$ is 2-colorable if and only if $G$ is **bipartite** (no odd cycles). Decidable in $O(|V| + |E|)$ via BFS.

## NP-Completeness of 3-Coloring

!!! tip "Theorem"
    3-Coloring is NP-complete.

**Membership in NP.** A coloring $c : V \to \{1, 2, 3\}$ serves as a polynomial-size certificate verifiable in $O(|E|)$ time.

**NP-Hardness: Reduction from 3-SAT.** Given a 3-SAT formula $\phi$ with variables $x_1, \ldots, x_n$ and clauses $C_1, \ldots, C_m$, construct a graph $G$:

### Gadget Construction

**Variable gadgets.** Create vertices $v_i$ and $\bar{v}_i$ for each variable $x_i$. Connect them with an edge. Also create three special vertices: $T$ (True), $F$ (False), $B$ (Base), forming a triangle. Connect each $v_i$ and $\bar{v}_i$ to $B$.

This forces each $v_i$ to be colored either $T$'s color or $F$'s color (encoding True/False).

**Clause gadgets.** For each clause $C_j = (\ell_1 \lor \ell_2 \lor \ell_3)$, construct a small gadget (the "OR gadget") that is 3-colorable if and only if at least one literal vertex has the True color.

The OR gadget uses 5 additional vertices per clause, connected so that 3-colorability requires at least one input to have the True color.

### Correctness

- If $\phi$ is satisfiable, the satisfying assignment determines a coloring of variable vertices. Each clause gadget is 3-colorable because at least one literal is True.
- If $G$ is 3-colorable, the colors of $v_i$ encode a satisfying assignment, and the clause gadgets ensure each clause has a True literal.

The construction is polynomial in $|\phi|$, completing the reduction. $\square$

## k-Coloring for k >= 3

For every fixed $k \geq 3$, $k$-Coloring is NP-complete. This follows by reduction from 3-Coloring: given a 3-Coloring instance $G$, add $k - 3$ new vertices connected to all of $V$ and to each other. The resulting graph is $k$-colorable if and only if $G$ is 3-colorable.

## Chromatic Number Properties

**Bounds:**

$$
\omega(G) \leq \chi(G) \leq \Delta(G) + 1
$$

where $\omega(G)$ is the clique number (largest clique) and $\Delta(G)$ is the maximum degree.

**Brooks' Theorem:** If $G$ is connected and not a complete graph or odd cycle, then $\chi(G) \leq \Delta(G)$.

**Greedy bound:** The greedy coloring algorithm (color vertices in order, using the smallest available color) uses at most $\Delta(G) + 1$ colors.

## Inapproximability

!!! warning "Hardness of Approximation"
    For any $\epsilon > 0$, it is NP-hard to approximate the chromatic number within a factor of $n^{1 - \epsilon}$ (Zuckerman, 2007). This means no polynomial-time algorithm can distinguish graphs with $\chi(G) = 3$ from those with $\chi(G) = n^{1-\epsilon}$.

## Special Graph Classes

| Graph Class | Chromatic Number | Complexity |
|-------------|-----------------|-----------|
| Bipartite | $\leq 2$ | P |
| Planar | $\leq 4$ (Four Color Theorem) | P for 4-coloring |
| Perfect | $= \omega(G)$ | P (via SDP) |
| Interval | $= \omega(G)$ (perfect) | P |
| Chordal | $= \omega(G)$ (perfect) | P |

The **Four Color Theorem** states that every planar graph is 4-colorable, but 3-Coloring remains NP-complete even for planar graphs.

??? example "Example: 3-Coloring a Small Graph"
    **Graph:** $K_4$ minus one edge. Vertices $\{a, b, c, d\}$, edges $\{(a,b), (a,c), (a,d), (b,c), (b,d)\}$ (edge $(c,d)$ is missing).

    **Is this 3-colorable?** Assign $c(a) = 1$. Then $b, c, d$ must avoid color 1. Since $b$ is adjacent to $c$ and $d$, give $c(b) = 2$. Now $c$ must avoid 1 and 2: $c(c) = 3$. And $d$ must avoid 1 and 2: $c(d) = 3$.

    **Check:** $(c, d)$ is not an edge, so $c(c) = c(d) = 3$ is allowed. Valid 3-coloring.

    **Chromatic number:** $\chi = 3$ (not 2-colorable since $\{a, b, c\}$ forms a triangle).

## Reference

- Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability*. W. H. Freeman.
- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
- Diestel, R. (2017). *Graph Theory* (5th ed.). Springer.
