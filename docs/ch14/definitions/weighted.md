# Weighted Graphs

Many real-world networks carry numerical information on their connections: road networks have distances, communication networks have bandwidth capacities, and financial networks have transaction amounts. Weighted graphs formalize this by attaching a numerical value to each edge, transforming the question "is there a connection?" into "how strong, costly, or long is the connection?" This simple extension opens the door to shortest-path algorithms, minimum spanning trees, and network flow problems.

## Definition

A **weighted graph** is a graph $G = (V, E)$ together with a weight function $w : E \to \mathbb{R}$ that assigns a real number to each edge. The triple $(V, E, w)$ fully specifies the weighted graph.

For an edge $e = (u, v)$ or $e = \{u, v\}$, the value $w(e)$ is called the **weight** (or cost, length, capacity) of the edge.

An **unweighted graph** can be viewed as a special case where every edge has weight 1:

$$
w(e) = 1 \quad \text{for all } e \in E
$$

## Types of Weights

Different problem domains assign different interpretations to edge weights.

| Interpretation | Example | Optimization Goal |
|---|---|---|
| Distance | Road network | Minimize total path weight |
| Cost | Transportation pricing | Minimize cost |
| Capacity | Network bandwidth | Maximize flow |
| Similarity | Correlation between assets | Maximize total weight |
| Probability | Reliability network | Maximize product of weights |

!!! warning "Negative Weights"
    When edges can have negative weights, algorithms like Dijkstra's fail because they assume adding an edge never decreases path cost. The [Bellman-Ford algorithm](../../ch15/single/bellman_ford.md) handles negative weights correctly, but negative-weight **cycles** make shortest paths undefined (path cost can be decreased indefinitely).

## Weighted Path Length

In an unweighted graph, the length of a path is simply the number of edges. In a weighted graph, the **weighted path length** (or path cost) sums the weights of all edges along the path.

For a path $p = v_0, v_1, \ldots, v_k$:

$$
w(p) = \sum_{i=0}^{k-1} w(v_i, v_{i+1})
$$

The **shortest-path distance** from $u$ to $v$ in a weighted graph is

$$
\delta(u, v) = \min\{w(p) : p \text{ is a path from } u \text{ to } v\}
$$

If no path exists, $\delta(u, v) = \infty$.

## Representation

Weighted graphs require storing weights alongside the edge structure. The three standard representations each handle weights differently.

### Adjacency Matrix

For a weighted graph on $n$ vertices, the adjacency matrix $A$ stores weights directly:

$$
A[i][j] = \begin{cases} w(i, j) & \text{if } (i, j) \in E \\ \infty \text{ (or 0)} & \text{if } (i, j) \notin E \end{cases}
$$

The choice between $\infty$ and 0 for non-edges depends on the algorithm. Shortest-path algorithms use $\infty$ (non-edge means infinite distance), while some matrix-based algorithms use 0.

### Adjacency List

Each adjacency list entry stores a pair (neighbor, weight):

$$
\text{adj}[u] = [(v_1, w_1), (v_2, w_2), \ldots]
$$

This is the most common representation for sparse weighted graphs.

### Edge List

Each edge is stored as a triple $(u, v, w)$. This representation is natural for algorithms that process edges one at a time, such as [Kruskal's MST algorithm](../../ch16/algorithms/kruskal.md).

## Implementation

```python
"""
Weighted graph representation and basic operations.

Demonstrates adjacency list and adjacency matrix representations
for weighted graphs, along with weighted path computation.
"""


# === Adjacency List (Weighted) ===

def build_weighted_adj_list(n, edges):
    """
    Build a weighted adjacency list.

    Each edge is (u, v, w) representing an undirected edge
    from u to v with weight w.
    """
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# === Adjacency Matrix (Weighted) ===

def build_weighted_matrix(n, edges):
    """
    Build a weighted adjacency matrix.

    Non-edges are represented as float('inf').
    Diagonal entries are 0 (distance from a vertex to itself).
    """
    INF = float('inf')
    matrix = [[INF] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for u, v, w in edges:
        matrix[u][v] = w
        matrix[v][u] = w
    return matrix


# === Path Weight ===

def path_weight(adj, path):
    """
    Compute the total weight of a path given as a vertex list.

    Returns the sum of edge weights along the path, or None if
    any edge in the path does not exist.
    """
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        found = False
        for neighbor, w in adj[u]:
            if neighbor == v:
                total += w
                found = True
                break
        if not found:
            return None
    return total


# === Main ===

if __name__ == "__main__":
    # Weighted graph: 4 vertices, 5 edges
    edges = [
        (0, 1, 4), (0, 2, 1), (1, 2, 2),
        (1, 3, 5), (2, 3, 8)
    ]
    n = 4

    # Adjacency list
    adj = build_weighted_adj_list(n, edges)
    print("Weighted adjacency list:")
    for v in range(n):
        print(f"  {v}: {adj[v]}")

    # Adjacency matrix
    matrix = build_weighted_matrix(n, edges)
    print("\nWeighted adjacency matrix:")
    for row in matrix:
        print(f"  {[x if x != float('inf') else 'inf' for x in row]}")

    # Path weights
    path1 = [0, 1, 3]  # weight = 4 + 5 = 9
    path2 = [0, 2, 1, 3]  # weight = 1 + 2 + 5 = 8
    print(f"\nPath {path1} weight: {path_weight(adj, path1)}")
    print(f"Path {path2} weight: {path_weight(adj, path2)}")
    print(f"Shorter path: {path2}")
```

**Output:**
```
Weighted adjacency list:
  0: [(1, 4), (2, 1)]
  1: [(0, 4), (2, 2), (3, 5)]
  2: [(0, 1), (1, 2), (3, 8)]
  3: [(1, 5), (2, 8)]
Weighted adjacency matrix:
  [0, 4, 1, 'inf']
  [4, 0, 2, 5]
  [1, 2, 0, 8]
  ['inf', 5, 8, 0]
Path [0, 1, 3] weight: 9
Path [0, 2, 1, 3] weight: 8
Shorter path: [0, 2, 1, 3]
```

## Key Algorithms for Weighted Graphs

| Algorithm | Problem | Time Complexity |
|---|---|---|
| [Dijkstra](../../ch15/single/dijkstra.md) | Single-source shortest paths (non-negative weights) | $O((V + E) \log V)$ |
| [Bellman-Ford](../../ch15/single/bellman_ford.md) | Single-source shortest paths (any weights) | $O(VE)$ |
| [Floyd-Warshall](../../ch15/all_pairs/floyd_warshall.md) | All-pairs shortest paths | $O(V^3)$ |
| [Prim](../../ch16/algorithms/prim.md) / [Kruskal](../../ch16/algorithms/kruskal.md) | Minimum spanning tree | $O(E \log V)$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapters 22-25.
