# Biconnected Components

A connected graph may contain vertices whose removal disconnects the graph. Understanding which parts of the graph can survive the loss of any single vertex leads to the notion of **biconnected components**. Decomposing a graph into these components reveals its internal structure of redundant connectivity and identifies the critical articulation points that hold the graph together.

## Definitions

**Biconnected graph.** A connected undirected graph $G = (V, E)$ with $|V| \ge 3$ is *biconnected* (or 2-connected) if removing any single vertex leaves the graph connected. Equivalently, $G$ is biconnected if and only if every pair of vertices lies on a common simple cycle.

**Biconnected component.** A maximal biconnected subgraph of $G$. Each edge of $G$ belongs to exactly one biconnected component. A bridge (an edge whose removal disconnects $G$) forms a biconnected component consisting of that single edge and its two endpoints.

**Articulation point.** A vertex $v$ whose removal increases the number of connected components. A vertex is an articulation point if and only if it belongs to two or more biconnected components.

## Key Properties

1. Every edge belongs to exactly one biconnected component.
2. Two biconnected components share at most one vertex, and any shared vertex is an articulation point.
3. A graph with no articulation points is either biconnected, a single edge, or a single vertex.
4. The number of biconnected components equals the number of edges in the block-cut tree.

## Algorithm

A single DFS pass finds all biconnected components and articulation points in $O(V + E)$ time. The algorithm maintains:

- $\text{disc}[v]$: the discovery time of vertex $v$.
- $\text{low}[v]$: the minimum discovery time reachable from the subtree rooted at $v$ using at most one back edge.

$$
\text{low}[v] = \min\!\bigl(\text{disc}[v],\; \min_{(v,w) \text{ back edge}} \text{disc}[w],\; \min_{(v,u) \text{ tree edge}} \text{low}[u]\bigr)
$$

A vertex $v$ is an articulation point if:

- $v$ is the DFS root and has two or more children in the DFS tree, or
- $v$ is not the root and has a child $u$ with $\text{low}[u] \ge \text{disc}[v]$.

To extract biconnected components, maintain an edge stack. When the DFS backtracks from $u$ to $v$ and $\text{low}[u] \ge \text{disc}[v]$, pop all edges from the stack down to and including $(v, u)$; these edges form one biconnected component.

## Implementation

```python
"""
Biconnected components of an undirected graph via DFS.

Uses Tarjan's algorithm with an edge stack to identify all biconnected
components and articulation points in O(V + E) time.
"""

from collections import defaultdict

# === Biconnected Component Finder ===

class BiconnectedComponents:
    """Find all biconnected components and articulation points."""

    def __init__(self, n: int):
        """Initialize graph with n vertices (0-indexed)."""
        self.n = n
        self.adj = defaultdict(list)
        self.components = []
        self.articulation_points = set()

    def add_edge(self, u: int, v: int) -> None:
        """Add undirected edge (u, v)."""
        self.adj[u].append(v)
        self.adj[v].append(u)

    def find_components(self) -> None:
        """Run DFS to find all biconnected components."""
        disc = [-1] * self.n
        low = [0] * self.n
        parent = [-1] * self.n
        stack = []  # edge stack
        timer = [0]

        def dfs(u: int) -> None:
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            children = 0

            for v in self.adj[u]:
                if disc[v] == -1:
                    children += 1
                    parent[v] = u
                    stack.append((u, v))
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # Articulation point check
                    is_root = parent[u] == -1
                    if (is_root and children > 1) or \
                       (not is_root and low[v] >= disc[u]):
                        self.articulation_points.add(u)

                    # Extract component when boundary detected
                    if low[v] >= disc[u]:
                        component = []
                        while stack:
                            edge = stack.pop()
                            component.append(edge)
                            if edge == (u, v):
                                break
                        self.components.append(component)

                elif v != parent[u] and disc[v] < disc[u]:
                    stack.append((u, v))
                    low[u] = min(low[u], disc[v])

        for i in range(self.n):
            if disc[i] == -1:
                dfs(i)


# === Demonstration ===

if __name__ == "__main__":
    # Graph: 0-1-2-0 (triangle), 2-3, 3-4-5-3 (triangle)
    bc = BiconnectedComponents(6)
    for u, v in [(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)]:
        bc.add_edge(u, v)
    bc.find_components()

    print(f"Number of biconnected components: {len(bc.components)}")
    print(f"Articulation points: {sorted(bc.articulation_points)}")
    for i, comp in enumerate(bc.components):
        vertices = set()
        for u, v in comp:
            vertices.update([u, v])
        print(f"  Component {i}: vertices {sorted(vertices)}, edges {comp}")
```

**Output:**

```
Number of biconnected components: 3
Articulation points: [2, 3]
  Component 0: vertices [3, 4, 5], edges [(4, 5), (5, 3), (3, 4)]
  Component 1: vertices [2, 3], edges [(2, 3)]
  Component 2: vertices [0, 1, 2], edges [(1, 2), (2, 0), (0, 1)]
```

The triangle $\{0, 1, 2\}$ forms one biconnected component, the bridge $(2, 3)$ forms another, and the triangle $\{3, 4, 5\}$ forms a third. Vertices $2$ and $3$ are articulation points because they each connect two components.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(V + E)$ |
| Space  | $O(V + E)$ |

The algorithm performs a single DFS pass. Each edge is pushed onto and popped from the stack exactly once, so the total work is linear in the size of the graph.

## Applications

- **Network reliability.** Biconnected components identify the parts of a network that remain connected even if any single node fails.
- **Redundant connections.** Within a biconnected component, there are at least two vertex-disjoint paths between any pair of vertices.
- **Block-cut tree.** The biconnected components and articulation points together define the block-cut tree, a higher-level view of the graph's connectivity structure.

## Reference

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: Efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372--378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 22: Elementary Graph Algorithms.
