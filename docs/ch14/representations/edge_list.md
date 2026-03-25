# Edge List

The edge list is the most straightforward graph representation: store every edge as a tuple. While it lacks the fast adjacency queries of matrices and adjacency lists, its simplicity and compact storage make it ideal for algorithms that process edges sequentially -- such as Kruskal's minimum spanning tree algorithm, Bellman-Ford shortest paths, and many input/output formats. Understanding when the edge list shines (and when it falls short) clarifies the trade-offs among graph representations.

## Definition

An **edge list** represents a graph $G = (V, E)$ as a collection of edges, where each edge is stored as a tuple:

- **Unweighted:** $(u, v)$ for each edge.
- **Weighted:** $(u, v, w)$ where $w = w(u, v)$ is the edge weight.

For an undirected graph, each edge appears once (either as $(u, v)$ or $(v, u)$, but not both). For a directed graph, each directed edge $(u, v)$ is stored with its direction.

## Complexity Analysis

| Operation | Time | Notes |
|---|---|---|
| Space | $O(E)$ | One entry per edge |
| Check if edge $(u,v)$ exists | $O(E)$ | Linear scan required |
| Find all neighbors of $u$ | $O(E)$ | Scan all edges for $u$ |
| Iterate all edges | $O(E)$ | Natural, sequential access |
| Add edge | $O(1)$ | Append to list |
| Remove edge | $O(E)$ | Find and remove |
| Sort by weight | $O(E \log E)$ | Needed for Kruskal's |

The $O(E)$ cost for adjacency queries is the main drawback. For algorithms that repeatedly check "is $(u, v)$ an edge?", an adjacency list or matrix is far more efficient.

## When to Use Edge Lists

Edge lists are the representation of choice in several scenarios:

1. **Edge-processing algorithms.** Kruskal's MST algorithm sorts edges by weight and processes them sequentially. Bellman-Ford relaxes every edge in each iteration. Both naturally consume an edge list.

2. **Input parsing.** Graph problems are commonly specified as a list of edges. Storing them directly avoids the overhead of building an adjacency structure when it is not needed.

3. **Very sparse graphs.** When $|E| \ll |V|$, storing an adjacency list with $|V|$ empty lists wastes space. An edge list uses only $O(E)$ space.

4. **Immutable graphs.** When the graph does not change after construction, and the algorithm only iterates over edges, the edge list is the most cache-friendly sequential structure.

## Implementation

```python
"""
Edge list representation for graphs.

Demonstrates edge list construction, basic operations, and
conversion to an adjacency list for algorithms requiring
neighbor queries.
"""


# === Edge List Class ===

class EdgeListGraph:
    """Graph represented as a list of edges."""

    def __init__(self, n_vertices, directed=False):
        self.n = n_vertices
        self.directed = directed
        self.edges = []

    def add_edge(self, u, v, weight=None):
        """Add an edge to the graph."""
        if weight is not None:
            self.edges.append((u, v, weight))
        else:
            self.edges.append((u, v))

    def has_edge(self, u, v):
        """Check if edge (u, v) exists. O(E) time."""
        for edge in self.edges:
            eu, ev = edge[0], edge[1]
            if eu == u and ev == v:
                return True
            if not self.directed and eu == v and ev == u:
                return True
        return False

    def neighbors(self, u):
        """Find all neighbors of vertex u. O(E) time."""
        result = []
        for edge in self.edges:
            eu, ev = edge[0], edge[1]
            if eu == u:
                result.append(ev)
            elif not self.directed and ev == u:
                result.append(eu)
        return result

    def to_adjacency_list(self):
        """Convert to adjacency list representation. O(E) time."""
        adj = [[] for _ in range(self.n)]
        for edge in self.edges:
            u, v = edge[0], edge[1]
            w = edge[2] if len(edge) == 3 else None
            adj[u].append((v, w) if w is not None else v)
            if not self.directed:
                adj[v].append((u, w) if w is not None else u)
        return adj

    def sorted_by_weight(self):
        """Return edges sorted by weight (for Kruskal's)."""
        return sorted(self.edges, key=lambda e: e[2])


# === Main ===

if __name__ == "__main__":
    # Weighted undirected graph
    g = EdgeListGraph(5, directed=False)
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 2, 2)
    g.add_edge(1, 3, 5)
    g.add_edge(2, 4, 3)

    print("Edge list:")
    for e in g.edges:
        print(f"  {e}")

    print(f"\nEdge (0,1) exists: {g.has_edge(0, 1)}")
    print(f"Edge (3,4) exists: {g.has_edge(3, 4)}")
    print(f"Neighbors of 2: {g.neighbors(2)}")

    print("\nEdges sorted by weight (for Kruskal's):")
    for e in g.sorted_by_weight():
        print(f"  {e[0]}-{e[1]} weight={e[2]}")

    print("\nConverted to adjacency list:")
    adj = g.to_adjacency_list()
    for v in range(5):
        print(f"  {v}: {adj[v]}")
```

**Output:**
```
Edge list:
  (0, 1, 4)
  (0, 2, 1)
  (1, 2, 2)
  (1, 3, 5)
  (2, 4, 3)
Edge (0,1) exists: True
Edge (3,4) exists: False
Neighbors of 2: [0, 1, 4]
Edges sorted by weight (for Kruskal's):
  0-2 weight=1
  1-2 weight=2
  2-4 weight=3
  0-1 weight=4
  1-3 weight=5
Converted to adjacency list:
  0: [(1, 4), (2, 1)]
  1: [(0, 4), (2, 2), (3, 5)]
  2: [(0, 1), (1, 2), (4, 3)]
  3: [(1, 5)]
  4: [(2, 3)]
```

## Comparison with Other Representations

For a detailed trade-off analysis across all operations, see [Representation Comparison](comparison.md). The key insight is that edge lists trade query speed for storage simplicity and sequential-access efficiency.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
