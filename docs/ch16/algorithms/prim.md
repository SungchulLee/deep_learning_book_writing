# Prim's Algorithm

While Kruskal's algorithm is edge-centric -- sorting and scanning all edges globally -- Prim's algorithm takes a vertex-centric approach. It grows a single tree from an arbitrary starting vertex, repeatedly attaching the lightest edge that connects the tree to a vertex not yet included. This mirrors how one might physically lay cable: start at one location and always extend to the nearest unconnected site.

## Algorithm Overview

Given a connected, undirected graph $G = (V, E)$ with weight function $w : E \to \mathbb{R}$, Prim's algorithm proceeds as follows:

1. **Initialize**: pick an arbitrary starting vertex $r$. Set the key of $r$ to 0 and the key of every other vertex to $\infty$. Each vertex's key represents the minimum weight of any edge connecting it to the growing tree.
2. **Repeat** $|V|$ times:
    - Extract the vertex $u$ with minimum key from the set of vertices not yet in the tree.
    - Add $u$ to the tree.
    - For each neighbor $v$ of $u$ not yet in the tree: if $w(u, v) < \text{key}[v]$, update $\text{key}[v] = w(u, v)$ and record $u$ as the parent of $v$.
3. **Terminate** when all vertices are in the tree.

## Pseudocode

```
PRIM(G, w, r):
    for each vertex u ∈ V:
        key[u] = ∞
        parent[u] = NIL
        in_tree[u] = False
    key[r] = 0
    Q = priority queue of all vertices, keyed by key[]
    while Q is not empty:
        u = EXTRACT-MIN(Q)
        in_tree[u] = True
        for each neighbor v of u:
            if not in_tree[v] and w(u, v) < key[v]:
                key[v] = w(u, v)
                parent[v] = u
                DECREASE-KEY(Q, v, key[v])
    return {(parent[v], v) : v ∈ V, v ≠ r}
```

## Worked Example

Consider a graph on $\{A, B, C, D, E\}$. Start from vertex $A$:

| Step | Extract | key[A] | key[B] | key[C] | key[D] | key[E] | Edge added |
|------|---------|--------|--------|--------|--------|--------|------------|
| Init | -- | 0 | inf | inf | inf | inf | -- |
| 1 | A | **0** | 4 | 1 | inf | inf | -- |
| 2 | C | -- | 3 | **1** | 5 | inf | (A, C) |
| 3 | B | -- | **3** | -- | 2 | inf | (C, B) |
| 4 | D | -- | -- | -- | **2** | 6 | (B, D) |
| 5 | E | -- | -- | -- | -- | **6** | (D, E) |

MST edges: $\{(A,C), (C,B), (B,D), (D,E)\}$ with total weight $1 + 3 + 2 + 6 = 12$.

## Correctness

At each step, Prim's algorithm maintains a tree $T$ on a subset $S \subseteq V$ of vertices. The cut $(S, V \setminus S)$ respects the current edge set because every edge in $T$ has both endpoints in $S$. The algorithm selects the lightest edge crossing this cut (the vertex with minimum key and its connecting edge). By the cut property, this edge is safe to add to the MST.

Since the algorithm adds $|V| - 1$ safe edges, the result is an MST.

## Complexity Analysis

The running time depends on the priority queue implementation:

| Priority Queue | EXTRACT-MIN | DECREASE-KEY | Total |
|---------------|-------------|--------------|-------|
| Array (unsorted) | $O(V)$ | $O(1)$ | $O(V^2)$ |
| Binary heap | $O(\log V)$ | $O(\log V)$ | $O(E \log V)$ |
| Fibonacci heap | $O(\log V)$ amortized | $O(1)$ amortized | $O(E + V \log V)$ |

**Array implementation**: iterating $V$ times, each EXTRACT-MIN scans $V$ entries. DECREASE-KEY is $O(1)$ (direct array access). Total: $O(V^2)$. This is optimal for dense graphs where $E = \Theta(V^2)$.

**Binary heap**: covered in detail on the next page (Prim with Heap).

**Fibonacci heap**: achieves $O(E + V \log V)$, which is the best known bound for Prim's algorithm and is asymptotically faster than Kruskal's $O(E \log E)$ when $E = \omega(V)$.

## Comparison with Kruskal's

| Aspect | Prim's | Kruskal's |
|--------|--------|-----------|
| Strategy | Grow one tree vertex by vertex | Merge forest edge by edge |
| Data structure | Priority queue | Union-Find |
| Best for | Dense graphs | Sparse graphs |
| Best complexity | $O(E + V \log V)$ (Fibonacci heap) | $O(E \log E)$ |
| Parallelizable | Less naturally | More naturally |

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Prim, R. C. (1957). Shortest connection networks and some generalizations. *Bell System Technical Journal*, 36(6), 1389--1401.
- [Prim's algorithm -- Wikipedia](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
