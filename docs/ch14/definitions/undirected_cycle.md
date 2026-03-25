# Detect Cycle in Undirected Graph

Cycle detection in undirected graphs answers a basic structural question: is the graph a forest (a collection of trees), or does it contain at least one cycle? Trees have exactly $|V| - 1$ edges for each connected component and no redundant connections, while the presence of a cycle means at least one edge is "extra." Two standard approaches exist: DFS-based parent tracking and Union-Find, each running in near-linear time.

## Cycle Definition in Undirected Graphs

A **cycle** in an undirected graph $G = (V, E)$ is a closed path $v_0, v_1, \ldots, v_{k} = v_0$ where $k \geq 3$ and all vertices $v_0, v_1, \ldots, v_{k-1}$ are distinct. The constraint $k \geq 3$ avoids counting a single edge as a "cycle" (traversing an edge back and forth is not a cycle in a simple graph).

!!! tip "Theorem: Tree Characterization"
    A connected undirected graph $G$ on $n$ vertices is a tree if and only if $G$ has exactly $n - 1$ edges. Equivalently, $G$ is a tree if and only if $G$ is connected and acyclic.

Adding any edge to a tree creates exactly one cycle, and removing any edge from a tree disconnects it.

## Method 1: DFS with Parent Tracking

During DFS on an undirected graph, each edge is explored from both endpoints. When visiting vertex $u$ and examining neighbor $v$, there are three cases:

1. $v$ is unvisited: continue DFS from $v$ with $u$ as parent.
2. $v$ is visited and $v$ is the parent of $u$: this is the same edge we arrived on -- skip it.
3. $v$ is visited and $v$ is **not** the parent of $u$: a cycle exists. The path from $v$ to $u$ in the DFS tree, combined with edge $\{u, v\}$, forms a cycle.

### Complexity

Each vertex and edge is examined at most twice (once from each endpoint), giving total time $O(V + E)$ and space $O(V)$ for the visited array and recursion stack.

$$
T(V, E) = O(V + E)
$$

```python
"""
Cycle detection in undirected graphs using DFS with parent tracking.

When DFS encounters a visited neighbor that is not the current
vertex's parent, a cycle has been found.
"""


# === DFS-Based Cycle Detection ===

def has_cycle_dfs(adj, n):
    """
    Detect a cycle in an undirected graph using DFS.

    Returns True if any cycle exists, False otherwise.
    Handles disconnected graphs by iterating over all components.
    """
    visited = [False] * n

    def dfs(u, parent):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                if dfs(v, u):
                    return True
            elif v != parent:
                return True  # back edge to non-parent => cycle
        return False

    for u in range(n):
        if not visited[u]:
            if dfs(u, -1):
                return True
    return False


# === Main ===

if __name__ == "__main__":
    # Graph with cycle: 0-1-2-0
    adj_cycle = [[1, 2], [0, 2], [0, 1]]
    print(f"Triangle has cycle: {has_cycle_dfs(adj_cycle, 3)}")

    # Tree: 0-1, 1-2, 1-3
    adj_tree = [[1], [0, 2, 3], [1], [1]]
    print(f"Tree has cycle: {has_cycle_dfs(adj_tree, 4)}")

    # Disconnected: tree + isolated vertex
    adj_disconnected = [[1], [0, 2], [1], []]
    print(f"Disconnected tree has cycle: "
          f"{has_cycle_dfs(adj_disconnected, 4)}")
```

**Output:**
```
Triangle has cycle: True
Tree has cycle: False
Disconnected tree has cycle: False
```

## Method 2: Union-Find

The Union-Find (disjoint set) approach processes edges one at a time. For each edge $\{u, v\}$:

- If $u$ and $v$ are in different sets, merge them (union).
- If $u$ and $v$ are already in the same set, this edge creates a cycle.

With union by rank and path compression, each operation takes amortized $O(\alpha(n))$ time, where $\alpha$ is the inverse Ackermann function. The total time is $O(E \cdot \alpha(V))$, which is effectively linear.

```python
"""
Cycle detection in undirected graphs using Union-Find.

Processes edges one by one; if both endpoints are already in the
same component, the edge would create a cycle.
"""


# === Union-Find ===

class UnionFind:
    """Disjoint set with union by rank and path compression."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank. Returns False if already same set (cycle)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # cycle detected
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# === Cycle Detection ===

def has_cycle_union_find(n, edges):
    """
    Detect a cycle using Union-Find.

    Returns True if any edge connects two vertices already in the
    same component.
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False


# === Main ===

if __name__ == "__main__":
    # Graph with cycle
    print(f"Triangle has cycle: "
          f"{has_cycle_union_find(3, [(0,1), (1,2), (2,0)])}")

    # Tree
    print(f"Tree has cycle: "
          f"{has_cycle_union_find(4, [(0,1), (1,2), (1,3)])}")
```

**Output:**
```
Triangle has cycle: True
Tree has cycle: False
```

## Method Comparison

| Aspect | DFS | Union-Find |
|---|---|---|
| Time complexity | $O(V + E)$ | $O(E \cdot \alpha(V))$ |
| Space | $O(V)$ | $O(V)$ |
| Finds cycle path | Yes (with parent tracking) | No (only detects existence) |
| Handles disconnected | Yes (iterate all vertices) | Yes (naturally) |
| Online (streaming edges) | No (needs full adjacency list) | Yes |

!!! warning "Multi-Edges and Self-Loops"
    The DFS parent-tracking approach assumes a simple graph. If multi-edges exist between the same pair of vertices, the parent check must track the specific edge index rather than just the parent vertex. Self-loops are immediately cycles.

## Connection to Trees and Forests

The cycle detection results connect directly to forest and tree properties:

- A connected graph on $n$ vertices is a tree if and only if it has exactly $n - 1$ edges and no cycle.
- A graph is a **forest** if and only if every connected component is a tree -- equivalently, the graph is acyclic.
- For [directed cycle detection](directed_cycle.md), the algorithm differs because edge direction matters: DFS uses three-color marking instead of parent tracking.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Sections 22.3, 21.1-21.3.
