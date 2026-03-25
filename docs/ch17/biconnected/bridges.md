# Finding Bridges

While [articulation points](articulation.md) are vertices whose removal disconnects a graph, **bridges** are edges with the same property. A bridge represents a single critical link -- if it fails, communication between two parts of the network is severed. Finding bridges uses the same DFS framework as articulation point detection, with a slightly stricter condition on the low-link values.

## Definition

!!! note "Formal Definition"
    An edge $(u, v)$ in a connected undirected graph $G = (V, E)$ is a **bridge** if removing $(u, v)$ disconnects $G$. Equivalently, $(u, v)$ is a bridge if and only if it does not lie on any cycle.

An edge on a cycle can be removed without disconnecting the graph because the cycle provides an alternative path. Conversely, an edge not on any cycle is the unique path between its endpoints' respective components, making it a bridge.

## DFS-Based Detection

The bridge-finding algorithm mirrors the [articulation point algorithm](articulation.md) but uses a strict inequality.

For a tree edge $(u, v)$ in the DFS tree (where $u$ is the parent of $v$):

- $(u, v)$ is a bridge if and only if $\text{low}[v] > \text{disc}[u]$.

!!! tip "Strict vs Non-Strict Inequality"
    For articulation points, the condition is $\text{low}[v] \geq \text{disc}[u]$ (non-strict). For bridges, it is $\text{low}[v] > \text{disc}[u]$ (strict). The difference arises because a back edge from $v$'s subtree to $u$ itself saves $u$ from being a bridge endpoint (the cycle through $u$ provides an alternate path), but it does not save $u$ from being an articulation point.

## Complexity

A single DFS pass finds all bridges:

$$
T(V, E) = O(V + E)
$$

Space complexity is $O(V)$ for the discovery times, low-link values, and parent array.

## Implementation

```python
"""
Finding bridges (cut edges) in an undirected graph.

Uses DFS with discovery times and low-link values. An edge (u, v) is
a bridge if no vertex in v's subtree can reach u or any of u's
ancestors through a back edge.
"""


# === Find Bridges ===
def find_bridges(graph, n):
    """
    Find all bridges in an undirected graph.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of an undirected graph with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[tuple[int, int]]
        List of bridge edges.
    """
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    bridges = []
    timer = [0]

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v in graph.get(u, []):
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # Strict inequality: bridge condition
                if low[v] > disc[u]:
                    bridges.append((u, v))

            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return bridges


# === Main ===
if __name__ == "__main__":
    # Graph with bridges: 3-4, 4-5, 5-6
    graph = {
        0: [1],
        1: [0, 2, 3],
        2: [1, 3],
        3: [1, 2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5],
    }
    bridges = find_bridges(graph, 7)
    print(f"Bridges: {bridges}")

    # Triangle graph (no bridges)
    triangle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    print(f"Triangle bridges: {find_bridges(triangle, 3)}")
```

**Output:**
```
Bridges: [(3, 4), (4, 5), (5, 6)]
Triangle bridges: []
```

The edges 3-4, 4-5, and 5-6 are bridges because they form a chain with no alternative paths. The triangle has no bridges because every edge lies on a cycle.

## Relationship to Articulation Points

Every bridge has at least one articulation point as an endpoint:

- If $(u, v)$ is a bridge and both $u$ and $v$ have degree greater than 1, then both $u$ and $v$ are articulation points.
- If $(u, v)$ is a bridge and $v$ is a leaf (degree 1), then $u$ is an articulation point (unless $u$ is also degree 1, in which case the graph has only two vertices and one edge).

However, an articulation point does not necessarily have a bridge incident to it. For example, consider vertex 1 in a graph shaped like two triangles sharing vertex 1: it is an articulation point, but no edge incident to it is a bridge.

## Bridge Trees

Contracting each [biconnected component](components.md) (2-edge-connected component) into a single vertex produces the **bridge tree** (or **block-cut tree** restricted to bridges). This tree captures the hierarchical bridge structure of the graph and is useful for:

- Answering queries about whether two vertices are separated by a bridge.
- Computing the minimum number of edges to add to eliminate all bridges.
- Finding the number of bridges on any path in $O(\log V)$ time with LCA queries.

## Applications

- **Network reliability:** Bridges represent single points of failure in communication networks.
- **Transportation:** Bridge edges in a road network are roads whose closure disconnects regions.
- **Biology:** In protein interaction networks, bridge interactions may indicate essential regulatory pathways.

## Reference

- Tarjan, R. E. (1974). A note on finding the bridges of a graph. *Information Processing Letters*, 2(6), 160-161.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
