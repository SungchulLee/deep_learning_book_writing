# Complete and Planar Graphs

Certain graph families arise so frequently in algorithm design and combinatorics that they deserve dedicated study. Complete graphs represent the densest possible structure, where every vertex connects to every other. Planar graphs, on the other hand, represent structures that can be drawn flat without crossings -- a property with deep consequences for algorithm efficiency and topological reasoning.

## Complete Graphs

A **complete graph** $K_n$ is a simple undirected graph on $n$ vertices in which every pair of distinct vertices is connected by exactly one edge. The edge count follows directly from choosing 2 vertices out of $n$:

$$
|E(K_n)| = \binom{n}{2} = \frac{n(n-1)}{2}
$$

!!! example "Small Complete Graphs"
    - $K_1$: a single vertex, no edges.
    - $K_2$: two vertices, one edge.
    - $K_3$: a triangle with 3 edges.
    - $K_4$: four vertices with $\binom{4}{2} = 6$ edges.
    - $K_5$: five vertices with 10 edges.

### Properties of Complete Graphs

- **Degree.** Every vertex in $K_n$ has degree $n - 1$, so $K_n$ is $(n-1)$-regular.
- **Connectivity.** $K_n$ is $(n-1)$-vertex-connected, meaning it remains connected after removing any $n - 2$ vertices.
- **Hamiltonian.** $K_n$ contains a Hamiltonian cycle for $n \geq 3$.
- **Chromatic number.** $\chi(K_n) = n$ because every vertex is adjacent to every other.
- **Clique number.** The clique number of $K_n$ is $n$ itself.

### Complete Bipartite Graphs

The complete bipartite graph $K_{m,n}$ connects every vertex in a set of size $m$ to every vertex in a set of size $n$, yielding $m \cdot n$ edges. The graph $K_{3,3}$ plays a special role in planarity theory (see below).

## Planar Graphs

A graph $G$ is **planar** if it can be drawn in the plane so that no two edges cross except at shared endpoints. Such a crossing-free drawing is called a **planar embedding** or **plane graph**. The regions bounded by edges in a planar embedding are called **faces**, including one unbounded outer face.

!!! example "Planar and Non-Planar Examples"
    - $K_4$ is planar: it can be drawn as a triangle with one vertex inside.
    - $K_5$ is **not** planar: no crossing-free drawing exists.
    - $K_{3,3}$ is **not** planar: this is the "three utilities" problem.

### Euler's Formula

The most fundamental result about planar graphs relates vertices, edges, and faces.

!!! tip "Theorem: Euler's Formula for Connected Planar Graphs"
    If $G$ is a connected planar graph with $v$ vertices, $e$ edges, and $f$ faces (including the outer face), then

$$
v - e + f = 2
$$

**Proof sketch.** Start with a spanning tree of $G$, which has $v$ vertices, $v - 1$ edges, and 1 face (the entire plane). Each additional edge added back creates exactly one new face by splitting an existing face. After adding all $e - (v - 1)$ remaining edges, the face count is $1 + e - v + 1 = e - v + 2$, so $f = e - v + 2$, which rearranges to $v - e + f = 2$. $\square$

### Edge Bound for Planar Graphs

Euler's formula yields a tight upper bound on the number of edges in a planar graph.

!!! tip "Corollary: Planar Edge Bound"
    If $G$ is a simple planar graph with $v \geq 3$ vertices and $e$ edges, then

$$
e \leq 3v - 6
$$

**Proof.** Every face is bounded by at least 3 edges (since $G$ is simple and has no multi-edges or self-loops). Counting edge-face incidences, each edge borders at most 2 faces, so $3f \leq 2e$. Substituting $f = 2 - v + e$ from Euler's formula gives $3(2 - v + e) \leq 2e$, which simplifies to $e \leq 3v - 6$. $\square$

This bound immediately proves that $K_5$ is non-planar: with $v = 5$, the bound gives $e \leq 9$, but $K_5$ has 10 edges.

### Kuratowski's Theorem

The definitive characterization of planarity connects the concept to two specific graphs.

!!! tip "Theorem: Kuratowski's Characterization"
    A graph $G$ is planar if and only if $G$ contains no subgraph that is a subdivision of $K_5$ or $K_{3,3}$.

A **subdivision** of a graph $H$ is obtained by replacing edges of $H$ with paths of one or more edges. Equivalently, by Wagner's theorem, $G$ is planar if and only if it has no $K_5$ or $K_{3,3}$ minor.

### Algorithmic Implications

Planarity has significant algorithmic consequences:

- **Planarity testing** can be performed in $O(V + E)$ time using algorithms by Hopcroft-Tarjan or Boyer-Myrvold.
- **Sparse structure.** The edge bound $e \leq 3v - 6$ guarantees that planar graphs are sparse: $|E| = O(|V|)$. Many algorithms that are expensive on dense graphs become efficient on planar graphs.
- **Four Color Theorem.** Every planar graph can be properly colored with at most 4 colors, so $\chi(G) \leq 4$ for all planar $G$.
- **Separator theorems.** Planar graphs admit $O(\sqrt{n})$-size separators, enabling efficient divide-and-conquer algorithms.

## Summary Comparison

| Property | Complete $K_n$ | Planar |
|---|---|---|
| Edge count | $\frac{n(n-1)}{2}$ | $\leq 3v - 6$ |
| Density | Maximum | Sparse ($O(V)$ edges) |
| Chromatic number | $n$ | $\leq 4$ |
| $K_5$ subgraph | Yes (for $n \geq 5$) | Never |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
- West, D. B. (2001). *Introduction to Graph Theory* (2nd ed.). Prentice Hall. Sections 1.1, 6.1-6.3.
