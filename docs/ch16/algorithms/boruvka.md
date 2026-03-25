# Boruvka's Algorithm

Boruvka's algorithm, published in 1926, is the oldest known MST algorithm -- predating both Kruskal's (1956) and Prim's (1957). Its key idea is that every component can independently and simultaneously select its cheapest outgoing edge, and all such selections are safe by the cut property. Because every component participates in each round, the number of components at least halves with each iteration, yielding at most $O(\log V)$ rounds. This inherent parallelism makes Boruvka's algorithm the foundation of modern parallel and distributed MST algorithms.

## Algorithm Overview

Given a connected, undirected graph $G = (V, E)$ with distinct edge weights:

1. **Initialize**: each vertex is its own component (a forest of $|V|$ singleton trees).
2. **Repeat** until only one component remains:
    - For each component, find the **cheapest edge** connecting it to a different component.
    - Add all such cheapest edges to the MST (some may be found by both endpoints' components -- add each edge only once).
    - Merge the connected components.
3. **Return** the collected edges as the MST.

## Pseudocode

```
BORUVKA(G, w):
    Initialize each vertex as its own component (using Union-Find)
    mst = ∅
    while number of components > 1:
        cheapest = array of size |V|, initialized to NIL
        for each edge (u, v, w) ∈ E:
            comp_u = FIND(u)
            comp_v = FIND(v)
            if comp_u ≠ comp_v:
                if cheapest[comp_u] is NIL or w < weight(cheapest[comp_u]):
                    cheapest[comp_u] = (u, v, w)
                if cheapest[comp_v] is NIL or w < weight(cheapest[comp_v]):
                    cheapest[comp_v] = (u, v, w)
        for each component c:
            if cheapest[c] is not NIL:
                (u, v, w) = cheapest[c]
                if FIND(u) ≠ FIND(v):
                    mst = mst ∪ {(u, v, w)}
                    UNION(u, v)
    return mst
```

## Why Components Halve Each Round

In each round, every component selects at least one outgoing edge. When two components are linked by their cheapest edges, they merge. Since every component merges with at least one other component, the number of components after a round is at most half the number before:

$$
C_{i+1} \le \frac{C_i}{2}
$$

Starting with $C_0 = |V|$ components, after $k$ rounds we have at most $|V| / 2^k$ components. The algorithm terminates when one component remains, so

$$
k \le \lceil \log_2 |V| \rceil
$$

## Worked Example

Consider a graph on $\{A, B, C, D\}$ with edges:

| Edge | Weight |
|------|--------|
| (A, B) | 4 |
| (A, C) | 1 |
| (B, C) | 3 |
| (B, D) | 2 |
| (C, D) | 5 |

**Round 1** (4 components: {A}, {B}, {C}, {D}):

- Component {A}: cheapest outgoing edge is (A, C) with weight 1.
- Component {B}: cheapest outgoing edge is (B, D) with weight 2.
- Component {C}: cheapest outgoing edge is (A, C) with weight 1.
- Component {D}: cheapest outgoing edge is (B, D) with weight 2.

Add edges (A, C) and (B, D). Components become {A, C} and {B, D}.

**Round 2** (2 components: {A, C}, {B, D}):

- Component {A, C}: cheapest outgoing edge is (B, C) with weight 3.
- Component {B, D}: cheapest outgoing edge is (B, C) with weight 3.

Add edge (B, C). One component remains: {A, B, C, D}.

MST: $\{(A, C), (B, D), (B, C)\}$ with total weight $1 + 2 + 3 = 6$.

## Complexity Analysis

**Per round**: scanning all $|E|$ edges to find the cheapest outgoing edge per component takes $O(E)$ time (with Union-Find operations contributing $O(E \cdot \alpha(V))$).

**Number of rounds**: at most $\lceil \log_2 V \rceil$.

**Total time**:

$$
T(V, E) = O(E \log V)
$$

**Space**: $O(V + E)$ for the graph and Union-Find structure.

## Correctness

The correctness follows directly from the cut property. In each round, for each component $C$, the cheapest outgoing edge crosses the cut $(C, V \setminus C)$. Since no MST edge in the current forest crosses this cut (all current MST edges have both endpoints within some component), the cut respects the current edge set. The cheapest crossing edge is therefore safe.

## Parallel Potential

Unlike Kruskal's (which requires a global sort) and Prim's (which grows a single tree sequentially), Boruvka's algorithm processes all components independently in each round. This makes it naturally parallelizable:

- Each round's edge scanning can be distributed across processors.
- With $p$ processors, each round takes $O(E / p)$ time.
- Total parallel time: $O((E \log V) / p)$.

This parallel structure has made Boruvka's algorithm the basis for many modern parallel MST algorithms, including those used in distributed computing and GPU implementations.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Boruvka, O. (1926). O jistem problemu minimalnim. *Prace Moravske Prirodovedecke Spolecnosti*, 3, 37--58.
