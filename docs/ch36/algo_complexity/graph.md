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

## 최소 뻗은 나무

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

## Exercises

**Exercise 1.**
Compare the time complexities of BFS, DFS, Dijkstra, and Bellman-Ford for a graph with $V$ vertices and $E$ edges. When is each algorithm the appropriate choice?

??? success "Solution to Exercise 1"
    BFS: $O(V + E)$. Use for unweighted shortest paths. DFS: $O(V + E)$. Use for connectivity, topological sort, cycle detection. Dijkstra: $O((V + E) \log V)$ with a binary heap, $O(V^2)$ with an array. Use for non-negative weighted shortest paths. Bellman-Ford: $O(VE)$. Use when negative edge weights exist (but no negative cycles). For dense graphs ($E \approx V^2$): Dijkstra with array is $O(V^2)$, matching BFS asymptotically but with higher constants. Bellman-Ford becomes $O(V^3)$. For sparse graphs ($E \approx V$): Dijkstra with heap is $O(V \log V)$, much faster than Bellman-Ford's $O(V^2)$. $\square$

---

**Exercise 2.**
Explain why Dijkstra's algorithm fails with negative edge weights. Give a concrete 3-node example.

??? success "Solution to Exercise 2"
    Dijkstra greedily finalizes the shortest distance to each node: once a node is extracted from the priority queue, its distance is never updated. With negative edges, a finalized distance can be wrong. Example: nodes A, B, C. Edges: A->B (weight 1), A->C (weight 5), B->C (weight -10). Dijkstra from A: extract A (dist 0), update B=1, C=5. Extract B (dist 1), update C = min(5, 1 + (-10)) = -9. Extract C (dist -9). In this case Dijkstra happens to work because C is extracted after B. But change to: A->B (weight 5), A->C (weight 2), C->B (weight -4). Extract A, update B=5, C=2. Extract C (dist 2), update B = min(5, 2-4) = -2. Extract B (dist -2). Correct! However, with A->B(1), B->C(2), A->C(10), C->B(-8): extract A, update B=1, C=10. Extract B(1), update C=3. Extract C(3). But C->B has weight -8, giving B = 3-8 = -5 < 1. B was already finalized at 1, so Dijkstra misses the better path. $\square$

---

**Exercise 3.**
The Floyd-Warshall algorithm computes all-pairs shortest paths in $O(V^3)$. Compare this with running Dijkstra from every vertex on a sparse graph ($E = O(V)$).

??? success "Solution to Exercise 3"
    Floyd-Warshall: always $O(V^3)$ regardless of graph density. Dijkstra from every vertex: $V \times O((V + E) \log V)$. For sparse graphs ($E = O(V)$): $V \times O(V \log V) = O(V^2 \log V)$, which is faster than $O(V^3)$ by a factor of $V / \log V$. For dense graphs ($E = O(V^2)$): $V \times O(V^2) = O(V^3)$, matching Floyd-Warshall. Floyd-Warshall is preferable when: (1) the graph has negative edges (Dijkstra requires non-negative); (2) the graph is dense; (3) simplicity of implementation matters (Floyd-Warshall is 5 lines of code). Repeated Dijkstra is preferable for sparse graphs with non-negative weights. Johnson's algorithm handles negative edges with sparse graphs in $O(V^2 \log V + VE)$ by reweighting edges. $\square$

---

**Exercise 4.**
A minimum spanning tree can be found using Kruskal's ($O(E \log E)$) or Prim's ($O(E \log V)$ with a heap). For which graph densities is each faster?

??? success "Solution to Exercise 4"
    Kruskal's: sort all edges ($O(E \log E)$) then process them with union-find ($O(E \alpha(V))$). Total: $O(E \log E) = O(E \log V)$ since $E \le V^2$. Prim's with a binary heap: $O((V + E) \log V)$. Prim's with a Fibonacci heap: $O(E + V \log V)$. For sparse graphs ($E = O(V)$): Kruskal = $O(V \log V)$, Prim (binary heap) = $O(V \log V)$, Prim (Fibonacci) = $O(V \log V)$. All equivalent. For dense graphs ($E = O(V^2)$): Kruskal = $O(V^2 \log V)$, Prim (binary heap) = $O(V^2 \log V)$, Prim (Fibonacci) = $O(V^2)$. Prim with Fibonacci heap wins on dense graphs. In practice, Kruskal is preferred for sparse graphs (simpler, good cache behavior with sorted edge list) and Prim with a heap for dense graphs. $\square$

---

**Exercise 5.**
Topological sort runs in $O(V + E)$. Prove this and explain why it is optimal.

??? success "Solution to Exercise 5"
    Kahn's algorithm: (1) compute in-degrees of all vertices: $O(V + E)$ (scan all edges). (2) Enqueue all vertices with in-degree 0: $O(V)$. (3) While queue is non-empty: dequeue vertex $u$ ($O(1)$), output $u$, decrement in-degree of each neighbor ($O(\text{out-degree}(u))$). Total work in step 3: each vertex dequeued once ($O(V)$) and each edge processed once ($O(E)$). Total: $O(V + E)$. Optimality: any algorithm must examine every vertex (to output it) and every edge (to determine ordering constraints). This requires $\Omega(V + E)$ time. Therefore, $O(V + E)$ is optimal. $\square$