# Christofides Algorithm for Metric TSP

The Traveling Salesman Problem (TSP) asks for the shortest tour visiting every city exactly once and returning to the start. While general TSP is NP-hard and inapproximable, the **metric** variant --- where distances satisfy the triangle inequality --- admits constant-factor approximation. The Christofides-Serdyukov algorithm achieves the best known ratio of $3/2$ for metric TSP, a bound that stood since 1976.

## Problem Definition

Given a complete graph $G = (V, E)$ with edge weights $w : E \to \mathbb{R}_{\geq 0}$ satisfying the **triangle inequality**:

$$
w(u, v) \leq w(u, x) + w(x, v) \quad \forall\, u, v, x \in V
$$

find a Hamiltonian cycle (tour) of minimum total weight. Let $\text{OPT}$ denote the cost of an optimal tour.

## Algorithm Steps

The Christofides algorithm combines three graph-theoretic ingredients: minimum spanning trees, minimum-weight perfect matchings, and Eulerian circuits.

**Input:** Complete graph $G = (V, E)$ with metric weights $w$.

1. **Minimum Spanning Tree.** Compute a minimum spanning tree $T$ of $G$.
2. **Odd-Degree Vertices.** Let $O \subseteq V$ be the set of vertices with odd degree in $T$. By the handshaking lemma, $|O|$ is even.
3. **Minimum-Weight Perfect Matching.** Compute a minimum-weight perfect matching $M$ on the complete subgraph induced by $O$.
4. **Eulerian Multigraph.** Form the multigraph $H = T \cup M$. Every vertex in $H$ has even degree, so $H$ is Eulerian.
5. **Euler Tour.** Find an Eulerian circuit of $H$.
6. **Shortcutting.** Convert the Euler tour to a Hamiltonian cycle by skipping previously visited vertices (the triangle inequality ensures this does not increase cost).

**Output:** A Hamiltonian cycle whose cost is at most $\frac{3}{2} \cdot \text{OPT}$.

## Approximation Guarantee

!!! tip "Theorem (Christofides 1976, Serdyukov 1978)"
    The Christofides algorithm is a $\frac{3}{2}$-approximation for metric TSP.

**Proof.** We bound the costs of $T$ and $M$ separately.

**Step 1: Bounding $w(T)$.** Removing any edge from the optimal tour $\text{OPT}$ yields a spanning tree. Since $T$ is a *minimum* spanning tree:

$$
w(T) \leq \text{OPT}
$$

**Step 2: Bounding $w(M)$.** Consider the optimal tour restricted to the odd-degree vertices $O = \{o_1, o_2, \ldots, o_{2k}\}$ in tour order. The shortcut tour on $O$ has cost at most $\text{OPT}$ (by the triangle inequality). This shortcut tour decomposes into two perfect matchings:

$$
M_1 = \{(o_1, o_2), (o_3, o_4), \ldots\}, \quad M_2 = \{(o_2, o_3), (o_4, o_5), \ldots\}
$$

Since $w(M_1) + w(M_2) \leq \text{OPT}$, the cheaper matching satisfies:

$$
w(M) \leq \min(w(M_1), w(M_2)) \leq \frac{\text{OPT}}{2}
$$

**Step 3: Combining.** The Euler tour on $H = T \cup M$ has cost $w(T) + w(M)$. Shortcutting does not increase cost (triangle inequality), so the final tour costs at most:

$$
w(T) + w(M) \leq \text{OPT} + \frac{\text{OPT}}{2} = \frac{3}{2} \cdot \text{OPT}
$$

$\square$

## Tightness of the Bound

The $3/2$ ratio is tight. Consider a path graph with $n$ vertices and unit-weight edges, completed into a metric by shortest-path distances. The MST cost is $n - 1$, the matching cost approaches $(n-1)/2$, and the optimal tour costs $n$. As $n$ grows, the ratio approaches $3/2$.

## Running Time

| Step | Algorithm | Time |
|------|-----------|------|
| MST | Prim / Kruskal | $O(n^2 \log n)$ |
| Odd vertices | Degree scan | $O(n)$ |
| Matching | Edmonds' blossom | $O(n^3)$ |
| Euler tour | Hierholzer | $O(n)$ |
| Shortcutting | Linear scan | $O(n)$ |

The bottleneck is the minimum-weight perfect matching step, giving overall complexity $O(n^3)$.

## Comparison with Other Approaches

| Algorithm | Ratio | Time | Notes |
|-----------|-------|------|-------|
| Nearest Neighbor | Unbounded (for metric: $O(\log n)$) | $O(n^2)$ | Greedy heuristic |
| Double-Tree | $2$ | $O(n^2 \log n)$ | MST + shortcut |
| Christofides | $3/2$ | $O(n^3)$ | Best classical ratio |

The double-tree algorithm uses the MST directly (traverse and shortcut) to achieve a 2-approximation. Christofides improves this by adding the matching step to handle odd-degree vertices more efficiently.

??? example "Worked Example: 5-City Instance"
    Consider 5 cities with distances forming a metric space:

    | | A | B | C | D | E |
    |---|---|---|---|---|---|
    | A | 0 | 2 | 5 | 7 | 3 |
    | B | 2 | 0 | 4 | 6 | 4 |
    | C | 5 | 4 | 0 | 3 | 5 |
    | D | 7 | 6 | 3 | 0 | 4 |
    | E | 3 | 4 | 5 | 4 | 0 |

    **Step 1:** MST edges: $\{(A,B,2), (B,C,4), (C,D,3), (A,E,3)\}$, cost = 12.

    **Step 2:** Odd-degree vertices: $O = \{C, D, E, B\}$ (each with degree 1 or 3).

    **Step 3:** Minimum matching on $O$: $\{(B,C,4), (D,E,4)\}$, cost = 8.

    **Step 4:** Eulerian multigraph $H$ has edges from both $T$ and $M$.

    **Step 5:** Euler tour: $A \to B \to C \to B \to C \to D \to E \to A$ (traversing all edges).

    **Step 6:** Shortcut: $A \to B \to C \to D \to E \to A$, cost = $2 + 4 + 3 + 4 + 3 = 16$.

    The algorithm produces a tour of cost 16. The bound gives $\frac{3}{2} \cdot \text{OPT}$, confirming the approximation guarantee.

## Reference

- Christofides, N. (1976). *Worst-case analysis of a new heuristic for the travelling salesman problem*. Technical Report 388, Graduate School of Industrial Administration, CMU.
- Serdyukov, A. I. (1978). On some extremal walks in graphs. *Upravlyaemye Sistemy*, 17, 76--79.
- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
