# Kruskal's Algorithm

Kruskal's algorithm takes a direct, edge-centric approach to building a minimum spanning tree: sort all edges by weight, then greedily add each edge as long as it does not create a cycle. This simple strategy produces an MST because every added edge is the lightest edge crossing some cut, satisfying the cut property. The algorithm is particularly well-suited for sparse graphs where the number of edges is close to the number of vertices.

## Algorithm Overview

Given a connected, undirected graph $G = (V, E)$ with weight function $w : E \to \mathbb{R}$, Kruskal's algorithm proceeds as follows:

1. **Sort** all edges in $E$ by non-decreasing weight.
2. **Initialize** a forest $F$ where each vertex is its own tree (component).
3. **Iterate** through edges in sorted order. For each edge $(u, v)$:
    - If $u$ and $v$ are in different components of $F$, add $(u, v)$ to $F$ (merging the two components).
    - If $u$ and $v$ are in the same component, skip the edge (it would create a cycle).
4. **Terminate** when $F$ has $|V| - 1$ edges (a spanning tree).

## Pseudocode

```
KRUSKAL(G, w):
    sort edges of G by non-decreasing weight
    F = ∅
    for each vertex v ∈ V:
        MAKE-SET(v)
    for each edge (u, v) in sorted order:
        if FIND-SET(u) ≠ FIND-SET(v):
            F = F ∪ {(u, v)}
            UNION(u, v)
    return F
```

The `MAKE-SET`, `FIND-SET`, and `UNION` operations form the **Union-Find** (disjoint set) data structure, covered in detail in the Union-Find section of this chapter.

## Worked Example

Consider a graph on vertices $\{A, B, C, D, E\}$ with edges sorted by weight:

| Order | Edge | Weight | Action |
|-------|------|--------|--------|
| 1 | (A, C) | 1 | Add -- connects $\{A\}$ and $\{C\}$ |
| 2 | (B, D) | 2 | Add -- connects $\{B\}$ and $\{D\}$ |
| 3 | (B, C) | 3 | Add -- connects $\{A, C\}$ and $\{B, D\}$ |
| 4 | (A, B) | 4 | Skip -- $A$ and $B$ already connected |
| 5 | (C, D) | 5 | Skip -- $C$ and $D$ already connected |
| 6 | (D, E) | 6 | Add -- connects $\{A, B, C, D\}$ and $\{E\}$ |

The MST consists of edges $\{(A,C), (B,D), (B,C), (D,E)\}$ with total weight $1 + 2 + 3 + 6 = 12$.

After adding 4 edges ($|V| - 1 = 5 - 1$), the algorithm terminates.

## Correctness

Kruskal's algorithm is correct by the cut property. When the algorithm adds edge $(u, v)$, the vertices $u$ and $v$ are in different components. Let $S$ be the vertex set of the component containing $u$. Then $(S, V \setminus S)$ is a cut that respects the current edge set $F$ (no edge in $F$ crosses between these specific components in a way that would violate the condition, since $F$ forms a forest of disjoint trees).

Edge $(u, v)$ is a light edge crossing this cut because all lighter edges have already been processed. Any lighter edge crossing this cut would have already been added (connecting the two components earlier) or would have connected vertices already in the same component. Since $(u, v)$ is the lightest available crossing edge, the cut property guarantees it is safe.

## Complexity Analysis

**Sorting**: sorting $|E|$ edges takes $O(E \log E)$ time. Since $|E| \le |V|^2$, we have $\log E \le 2 \log V = O(\log V)$, so sorting is $O(E \log V)$.

**Union-Find operations**: with union by rank and path compression, each `FIND-SET` and `UNION` operation takes amortized $O(\alpha(V))$ time, where $\alpha$ is the inverse Ackermann function. Over $O(E)$ operations, this contributes $O(E \cdot \alpha(V))$.

**Total**:

$$
T(V, E) = O(E \log E) + O(E \cdot \alpha(V)) = O(E \log E)
$$

Since $\alpha(V) \le \log V \le \log E$ for connected graphs, the sorting step dominates.

**Space**: $O(V + E)$ for the graph representation and Union-Find structure.

## When to Use Kruskal's

Kruskal's algorithm is preferred when:

- The graph is **sparse** ($|E| = O(|V|)$ or $|E| = O(|V| \log |V|)$).
- Edges are provided as a **list** rather than an adjacency matrix.
- Edges arrive in a **stream** and can be sorted externally.

For dense graphs where $|E| = \Theta(|V|^2)$, Prim's algorithm with a binary heap achieves $O(E \log V)$, which is asymptotically the same, but Prim's with a Fibonacci heap achieves $O(E + V \log V)$, which is faster.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Kruskal, J. B. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. *Proceedings of the AMS*, 7(1), 48--50.
- [Kruskal's algorithm -- Wikipedia](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
