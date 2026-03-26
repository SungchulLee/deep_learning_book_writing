# Graph Algorithm Complexities

Graphs model relationships between entities, and the complexity of graph algorithms
depends heavily on the representation (adjacency list vs. matrix) and the structure of
the graph (sparse vs. dense, weighted vs. unweighted). This page collects the time and
space bounds for every major graph algorithm, so you can quickly estimate feasibility
for a given input size.

## Traversal Algorithms

Traversals visit every vertex and edge exactly once, making them the foundation for
many higher-level algorithms.

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| BFS | $O(V + E)$ | $O(V)$ | Queue-based; finds shortest path in unweighted graphs |
| DFS | $O(V + E)$ | $O(V)$ | Stack/recursion; basis for topological sort, SCC |
| IDDFS | $O(b^d)$ | $O(d)$ | Combines BFS optimality with DFS space; $b$ = branching factor, $d$ = depth |

Here $V$ is the number of vertices and $E$ is the number of edges. Both BFS and DFS
assume an adjacency list representation. With an adjacency matrix, time becomes $O(V^2)$.

## Shortest Path Algorithms

Different algorithms apply depending on edge weight constraints.

| Algorithm | Time | Space | Constraints |
|---|---|---|---|
| BFS | $O(V + E)$ | $O(V)$ | Unweighted edges |
| Dijkstra (binary heap) | $O((V + E) \log V)$ | $O(V)$ | Non-negative weights |
| Dijkstra (Fibonacci heap) | $O(V \log V + E)$ | $O(V)$ | Non-negative weights |
| Bellman-Ford | $O(VE)$ | $O(V)$ | Handles negative weights |
| SPFA | $O(VE)$ worst, $O(E)$ avg | $O(V)$ | Bellman-Ford variant |
| DAG relaxation | $O(V + E)$ | $O(V)$ | DAG only (topological order) |
| Floyd-Warshall | $O(V^3)$ | $O(V^2)$ | All-pairs; handles negative weights |
| Johnson's | $O(V^2 \log V + VE)$ | $O(V^2)$ | All-pairs; sparse graphs |

!!! tip "Choosing the Right Algorithm"
    For sparse graphs ($E \approx V$), Dijkstra with a binary heap runs in
    $O(V \log V)$. For dense graphs ($E \approx V^2$), Floyd-Warshall's $O(V^3)$
    may be simpler to implement with comparable performance.

## Minimum Spanning Tree

MST algorithms find the least-weight subset of edges that connects all vertices in an
undirected, connected, weighted graph.

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Kruskal's | $O(E \log E)$ | $O(V)$ | Sort edges, use Union-Find |
| Prim's (binary heap) | $O((V + E) \log V)$ | $O(V)$ | Better for dense graphs |
| Prim's (Fibonacci heap) | $O(E + V \log V)$ | $O(V)$ | Theoretically optimal |
| Boruvka's | $O(E \log V)$ | $O(V)$ | Parallelizable |

Kruskal's algorithm is dominated by the sort step. The Union-Find operations (with
path compression and union by rank) run in amortized $O(\alpha(n))$ per operation,
where $\alpha$ is the inverse Ackermann function.

## Topological Sort and SCC

These algorithms apply to directed graphs and rely on DFS as a subroutine.

| Algorithm | Time | Space | Purpose |
|---|---|---|---|
| Topological sort (DFS) | $O(V + E)$ | $O(V)$ | Linear ordering of DAG |
| Kahn's algorithm | $O(V + E)$ | $O(V)$ | BFS-based topological sort |
| Kosaraju's SCC | $O(V + E)$ | $O(V)$ | Two DFS passes |
| Tarjan's SCC | $O(V + E)$ | $O(V)$ | Single DFS pass |

## Network Flow

Flow algorithms find maximum flow or minimum cost flow in directed networks.

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Ford-Fulkerson (DFS) | $O(E \cdot f^*)$ | $O(V + E)$ | $f^*$ = max flow value |
| Edmonds-Karp (BFS) | $O(VE^2)$ | $O(V + E)$ | Polynomial bound |
| Dinic's | $O(V^2 E)$ | $O(V + E)$ | $O(E\sqrt{V})$ for unit-capacity |
| Push-Relabel | $O(V^2 E)$ | $O(V + E)$ | Often faster in practice |
| Hungarian | $O(V^3)$ | $O(V^2)$ | Bipartite matching |

!!! warning "Ford-Fulkerson Pitfall"
    Ford-Fulkerson with DFS may not terminate on irrational capacities and can be
    exponentially slow on integer capacities. Always prefer Edmonds-Karp or Dinic's
    for reliable polynomial-time behavior.

## Complexity by Graph Density

The relationship between $V$ and $E$ determines which algorithms are practical.

- **Sparse graphs** ($E = O(V)$): Adjacency list is essential. Dijkstra with a
  heap gives $O(V \log V)$. Bellman-Ford gives $O(V^2)$.
- **Dense graphs** ($E = O(V^2)$): Adjacency matrix is viable. Floyd-Warshall's
  $O(V^3)$ competes with running Dijkstra from each vertex, which gives
  $O(V^3 \log V)$ with a binary heap or $O(V^3)$ with a Fibonacci heap.

| Input size $V$ | Max $E$ (dense) | BFS/DFS | Dijkstra (heap) | Floyd-Warshall |
|---|---|---|---|---|
| $10^3$ | $10^6$ | fast | fast | fast |
| $10^4$ | $10^8$ | fast | moderate | slow |
| $10^5$ | $10^{10}$ | moderate | moderate | infeasible |
| $10^6$ | $10^{12}$ | moderate | slow | infeasible |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.
