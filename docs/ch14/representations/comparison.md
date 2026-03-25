# Representation Comparison

Choosing the right graph representation is one of the first and most consequential decisions in graph algorithm design. The adjacency matrix, adjacency list, and edge list each offer different trade-offs in space usage and operation speed. The optimal choice depends on the graph's density, the operations the algorithm performs most frequently, and whether the graph is static or dynamic.

## Sparse vs Dense Graphs

The distinction between sparse and dense graphs drives the representation choice.

A graph on $|V|$ vertices can have at most $O(|V|^2)$ edges. A graph is **sparse** when $|E| = O(|V|)$ or more generally $|E| \ll |V|^2$. It is **dense** when $|E| = \Theta(|V|^2)$.

Most real-world graphs are sparse: social networks, road maps, and biological networks typically have each vertex connected to a small fraction of all other vertices.

## Operation Complexity Comparison

The following table compares the three standard representations across common operations.

| Operation | Adjacency Matrix | Adjacency List | Edge List |
|---|---|---|---|
| **Space** | $O(V^2)$ | $O(V + E)$ | $O(E)$ |
| **Check edge $(u,v)$** | $O(1)$ | $O(\deg(u))$ | $O(E)$ |
| **Iterate neighbors of $u$** | $O(V)$ | $O(\deg(u))$ | $O(E)$ |
| **Iterate all edges** | $O(V^2)$ | $O(V + E)$ | $O(E)$ |
| **Add edge** | $O(1)$ | $O(1)$ | $O(1)$ |
| **Remove edge** | $O(1)$ | $O(\deg(u))$ | $O(E)$ |
| **Add vertex** | $O(V)$ (resize) | $O(1)$ | $O(1)$ |
| **Check if graph is dense** | Natural fit | Wasteful | Wasteful |

## Detailed Analysis

### Adjacency Matrix

The [adjacency matrix](matrix.md) stores a $|V| \times |V|$ matrix $A$ where $A[i][j] = 1$ (or the edge weight) if edge $(i,j)$ exists.

**Strengths:**

- $O(1)$ edge existence queries -- critical for algorithms like Floyd-Warshall that check $A[i][j]$ repeatedly.
- Simple implementation with 2D arrays.
- Matrix operations (multiplication, transitive closure) apply directly.

**Weaknesses:**

- $O(V^2)$ space regardless of edge count, wasteful for sparse graphs.
- Iterating neighbors of a vertex always takes $O(V)$, even if the vertex has few neighbors.
- Adding a vertex requires resizing the entire matrix.

**Best for:** Dense graphs, algorithms using matrix operations, small graphs where $O(V^2)$ space is acceptable.

### Adjacency List

The [adjacency list](list.md) stores, for each vertex $u$, a list of vertices adjacent to $u$ (with optional edge weights).

**Strengths:**

- $O(V + E)$ space, proportional to the actual graph size.
- Iterating neighbors of $u$ takes $O(\deg(u))$, which is optimal.
- Most graph traversal algorithms (BFS, DFS) naturally iterate over neighbor lists, making this the default choice.

**Weaknesses:**

- Edge existence queries take $O(\deg(u))$ in the worst case (can be improved to $O(1)$ with hash sets instead of lists).
- Slightly more complex implementation than a matrix.

**Best for:** Sparse graphs (the most common case), BFS/DFS-based algorithms, dynamic graphs.

### Edge List

The [edge list](edge_list.md) stores edges as a flat list of tuples $(u, v)$ or $(u, v, w)$.

**Strengths:**

- $O(E)$ space, the most compact representation when $E \ll V$.
- Simple to iterate over all edges, natural for Kruskal's algorithm and other edge-processing algorithms.
- Easy to sort edges by weight.

**Weaknesses:**

- Neighbor queries and edge existence checks require $O(E)$ scans.
- Not suitable for algorithms that repeatedly query adjacency.

**Best for:** Edge-centric algorithms (Kruskal, Bellman-Ford), input/output formats, very sparse graphs.

## Decision Guide

```python
"""
Graph representation selection guide.

Demonstrates when to choose each representation based on graph
density and the primary operations needed.
"""


# === Representation Selection ===

def recommend_representation(n_vertices, n_edges, primary_ops):
    """
    Recommend a graph representation based on graph properties
    and required operations.

    Parameters:
        n_vertices: number of vertices
        n_edges: number of edges
        primary_ops: list of primary operations needed
            ('edge_query', 'neighbor_iter', 'all_edges', 'matrix_ops')
    """
    density = n_edges / max(1, n_vertices * (n_vertices - 1) / 2)
    recommendations = []

    if 'matrix_ops' in primary_ops:
        recommendations.append(("Adjacency Matrix",
                                "matrix operations required"))
    elif 'edge_query' in primary_ops and density > 0.5:
        recommendations.append(("Adjacency Matrix",
                                f"dense graph ({density:.1%}), "
                                f"O(1) edge queries"))
    elif 'all_edges' in primary_ops and 'neighbor_iter' not in primary_ops:
        recommendations.append(("Edge List",
                                "only need to iterate all edges"))
    else:
        recommendations.append(("Adjacency List",
                                f"sparse graph ({density:.1%}), "
                                f"efficient neighbor iteration"))

    return recommendations


# === Space Comparison ===

def compare_space(n_vertices, n_edges):
    """Compare space usage of all three representations."""
    matrix_space = n_vertices ** 2
    adj_list_space = n_vertices + 2 * n_edges  # undirected
    edge_list_space = 2 * n_edges

    return {
        "Adjacency Matrix": matrix_space,
        "Adjacency List": adj_list_space,
        "Edge List": edge_list_space,
    }


# === Main ===

if __name__ == "__main__":
    # Sparse graph: 1000 vertices, 3000 edges
    print("=== Sparse Graph (V=1000, E=3000) ===")
    space = compare_space(1000, 3000)
    for name, s in space.items():
        print(f"  {name}: {s:,} entries")
    recs = recommend_representation(1000, 3000, ['neighbor_iter'])
    print(f"  Recommendation: {recs[0][0]} ({recs[0][1]})")

    # Dense graph: 100 vertices, 4000 edges
    print("\n=== Dense Graph (V=100, E=4000) ===")
    space = compare_space(100, 4000)
    for name, s in space.items():
        print(f"  {name}: {s:,} entries")
    recs = recommend_representation(100, 4000, ['edge_query'])
    print(f"  Recommendation: {recs[0][0]} ({recs[0][1]})")

    # Edge-centric: Kruskal's MST
    print("\n=== Edge-Centric (Kruskal's MST) ===")
    recs = recommend_representation(1000, 5000, ['all_edges'])
    print(f"  Recommendation: {recs[0][0]} ({recs[0][1]})")
```

**Output:**
```
=== Sparse Graph (V=1000, E=3000) ===
  Adjacency Matrix: 1,000,000 entries
  Adjacency List: 7,000 entries
  Edge List: 6,000 entries
  Recommendation: Adjacency List (sparse graph (0.6%), efficient neighbor iteration)
=== Dense Graph (V=100, E=4000) ===
  Adjacency Matrix: 10,000 entries
  Adjacency List: 8,100 entries
  Edge List: 8,000 entries
  Recommendation: Adjacency Matrix (dense graph (80.8%), O(1) edge queries)
=== Edge-Centric (Kruskal's MST) ===
  Recommendation: Edge List (only need to iterate all edges)
```

## Hybrid Approaches

In practice, several hybrid strategies combine the advantages of multiple representations:

- **Adjacency list with hash sets.** Replace each vertex's neighbor list with a hash set to get $O(1)$ edge existence queries while maintaining $O(V + E)$ space.
- **Compressed sparse row (CSR).** Store the adjacency list in contiguous arrays for cache-friendly neighbor iteration. Common in high-performance graph libraries.
- **Implicit representations.** For graphs defined by rules (grids, game states), neighbors are computed on the fly without explicit storage. See [Implicit Graphs](implicit.md).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
