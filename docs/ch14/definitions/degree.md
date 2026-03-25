# Degree

The degree of a vertex counts how many edges touch it, making degree one of the simplest yet most informative local measurements in a graph. Degree distributions reveal whether a network is regular, whether hubs dominate its structure, and whether certain algorithmic shortcuts are available. In directed graphs, splitting degree into incoming and outgoing components exposes the flow of information or resources through the network.

## Degree in Undirected Graphs

For an undirected graph $G = (V, E)$, the **degree** of a vertex $v$, denoted $\deg(v)$, is the number of edges incident to $v$. A self-loop at $v$ contributes 2 to $\deg(v)$ (both endpoints of the edge are $v$).

$$
\deg(v) = |\{e \in E : v \in e\}|
$$

A vertex with $\deg(v) = 0$ is called an **isolated vertex**, and a vertex with $\deg(v) = 1$ is a **leaf** (or pendant vertex).

!!! example "Degree Computation"
    Consider a graph on vertices $\{a, b, c, d\}$ with edges $\{(a,b), (a,c), (a,d), (b,c)\}$. Then $\deg(a) = 3$, $\deg(b) = 2$, $\deg(c) = 2$, and $\deg(d) = 1$. The vertex $d$ is a leaf.

## Degree in Directed Graphs

In a directed graph (digraph) $G = (V, E)$, each edge $(u, v)$ has a direction from $u$ to $v$. This leads to two separate degree measures for each vertex.

The **in-degree** of $v$, denoted $\deg^-(v)$, counts the edges directed into $v$:

$$
\deg^-(v) = |\{(u, v) \in E : u \in V\}|
$$

The **out-degree** of $v$, denoted $\deg^+(v)$, counts the edges directed out of $v$:

$$
\deg^+(v) = |\{(v, w) \in E : w \in V\}|
$$

The total degree of $v$ in a digraph is $\deg(v) = \deg^-(v) + \deg^+(v)$.

!!! example "In-Degree and Out-Degree"
    In a digraph with edges $\{(a,b), (a,c), (b,c), (c,a)\}$: vertex $a$ has $\deg^+(a) = 2$ (edges to $b$ and $c$) and $\deg^-(a) = 1$ (edge from $c$). Vertex $c$ has $\deg^+(c) = 1$ and $\deg^-(c) = 2$.

## Relationship to Edge Count

The sum of all vertex degrees relates directly to the number of edges. For undirected graphs, each edge contributes exactly 1 to the degree of each of its two endpoints, giving the [Handshaking Lemma](handshaking.md):

$$
\sum_{v \in V} \deg(v) = 2|E|
$$

For directed graphs, each directed edge contributes 1 to one vertex's out-degree and 1 to another's in-degree:

$$
\sum_{v \in V} \deg^-(v) = \sum_{v \in V} \deg^+(v) = |E|
$$

## Degree Sequence

The **degree sequence** of a graph is the list of vertex degrees sorted in non-increasing order. Two graphs with different degree sequences cannot be isomorphic, making the degree sequence a basic graph invariant.

!!! example "Degree Sequence Example"
    A path graph $P_4$ on vertices $\{1, 2, 3, 4\}$ with edges $\{(1,2), (2,3), (3,4)\}$ has degree sequence $(2, 2, 1, 1)$.

Not every non-increasing sequence of non-negative integers is realizable as the degree sequence of a simple graph. The **Erdos-Gallai theorem** provides a necessary and sufficient condition: a non-increasing sequence $d_1 \geq d_2 \geq \cdots \geq d_n$ is graphical (realizable) if and only if the sum $\sum d_i$ is even, and for each $k \in \{1, \ldots, n\}$:

$$
\sum_{i=1}^{k} d_i \leq k(k-1) + \sum_{i=k+1}^{n} \min(d_i, k)
$$

## Regular Graphs

A graph is **$k$-regular** if every vertex has degree exactly $k$. Special cases include:

| Regularity | Name | Example |
|---|---|---|
| 0-regular | Empty graph | $n$ isolated vertices |
| 1-regular | Perfect matching | Edge-disjoint pairs |
| 2-regular | Disjoint cycles | Union of cycles covering all vertices |
| 3-regular | Cubic graph | Petersen graph |
| $(n-1)$-regular | Complete graph | $K_n$ |

For a $k$-regular graph on $n$ vertices, the handshaking lemma gives $|E| = kn/2$, so $kn$ must be even.

## Computing Degrees

```python
"""
Degree computation for undirected and directed graphs.

Demonstrates how to compute vertex degrees from adjacency list
representations, including degree sequences and regularity checks.
"""


# === Undirected Degree ===

def compute_degrees_undirected(adj, n):
    """Compute degree of each vertex in an undirected graph."""
    degrees = [0] * n
    for u in range(n):
        degrees[u] = len(adj[u])
    return degrees


# === Directed Degree ===

def compute_degrees_directed(adj, n):
    """Compute in-degree and out-degree for a directed graph."""
    in_deg = [0] * n
    out_deg = [0] * n
    for u in range(n):
        out_deg[u] = len(adj[u])
        for v in adj[u]:
            in_deg[v] += 1
    return in_deg, out_deg


# === Degree Sequence ===

def degree_sequence(adj, n):
    """Return the degree sequence in non-increasing order."""
    degrees = compute_degrees_undirected(adj, n)
    return sorted(degrees, reverse=True)


# === Regularity Check ===

def is_k_regular(adj, n):
    """Check if the graph is k-regular. Return k or -1."""
    degrees = compute_degrees_undirected(adj, n)
    if len(set(degrees)) == 1:
        return degrees[0]
    return -1


# === Main ===

if __name__ == "__main__":
    # Undirected graph: triangle with pendant
    adj_undirected = [[1, 2], [0, 2, 3], [0, 1], [1]]
    degrees = compute_degrees_undirected(adj_undirected, 4)
    print(f"Degrees: {degrees}")
    print(f"Degree sequence: {degree_sequence(adj_undirected, 4)}")
    print(f"Sum of degrees: {sum(degrees)}, 2*|E| = {2 * 4}")

    # Directed graph
    adj_directed = [[1, 2], [2], [0], []]
    in_d, out_d = compute_degrees_directed(adj_directed, 4)
    print(f"\nDirected in-degrees:  {in_d}")
    print(f"Directed out-degrees: {out_d}")

    # Regular graph: cycle on 4 vertices
    adj_cycle = [[1, 3], [0, 2], [1, 3], [2, 0]]
    k = is_k_regular(adj_cycle, 4)
    print(f"\nCycle C4 is {k}-regular")
```

**Output:**
```
Degrees: [2, 3, 2, 1]
Degree sequence: [3, 2, 2, 1]
Sum of degrees: 8, 2*|E| = 8
Directed in-degrees:  [1, 1, 2, 0]
Directed out-degrees: [2, 1, 1, 0]
Cycle C4 is 2-regular
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
- West, D. B. (2001). *Introduction to Graph Theory* (2nd ed.). Prentice Hall. Section 1.3.
