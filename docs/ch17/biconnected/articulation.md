# Finding Articulation Points

In a communication network, certain nodes are critical -- if they fail, the network splits into disconnected parts. These vulnerable nodes are called **articulation points** (or **cut vertices**). Identifying them is essential for assessing network robustness, designing fault-tolerant systems, and understanding the structure of [biconnected components](components.md). A single DFS pass can find all articulation points in linear time.

## Definition

!!! note "Formal Definition"
    A vertex $v$ in a connected undirected graph $G = (V, E)$ is an **articulation point** if removing $v$ (and all edges incident to $v$) disconnects $G$. Equivalently, the graph $G - v$ has more connected components than $G$.

A graph with no articulation points is called **biconnected** -- every pair of vertices has at least two vertex-disjoint paths between them.

## DFS-Based Algorithm

The algorithm uses DFS discovery times and low-link values, similar to [Tarjan's SCC algorithm](../scc/tarjan.md) but applied to undirected graphs.

For each vertex $u$, define:

- $\text{disc}[u]$: the discovery time of $u$ in the DFS.
- $\text{low}[u]$: the minimum discovery time reachable from $u$ through the DFS subtree of $u$, including back edges.

$$
\text{low}[u] = \min\!\Big(\text{disc}[u],\ \min_{\substack{v \text{ child of } u}} \text{low}[v],\ \min_{\substack{(u,w) \text{ back edge}}} \text{disc}[w]\Big)
$$

A vertex $u$ is an articulation point if and only if one of these conditions holds:

1. **Root condition:** $u$ is the root of the DFS tree and has two or more children.
2. **Non-root condition:** $u$ is not the root and has a child $v$ such that $\text{low}[v] \geq \text{disc}[u]$.

!!! tip "Intuition Behind the Conditions"
    The non-root condition says: if no vertex in $v$'s subtree can reach an ancestor of $u$ through a back edge, then removing $u$ disconnects $v$'s subtree from the rest of the graph. The root condition handles the special case where the root has no parent -- it is an articulation point only if it has multiple independent subtrees.

## Complexity

The algorithm performs a single DFS traversal:

$$
T(V, E) = O(V + E)
$$

Space complexity is $O(V)$ for the discovery times, low-link values, and parent tracking.

## Implementation

```python
"""
Finding articulation points (cut vertices) in an undirected graph.

Uses a single DFS pass with discovery times and low-link values to
identify vertices whose removal disconnects the graph.
"""


# === Find Articulation Points ===
def find_articulation_points(graph, n):
    """
    Find all articulation points in an undirected graph.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of an undirected graph with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[int]
        List of articulation point vertices.
    """
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    is_ap = [False] * n
    timer = [0]

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0

        for v in graph.get(u, []):
            if disc[v] == -1:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # Root with 2+ children
                if parent[u] == -1 and children > 1:
                    is_ap[u] = True

                # Non-root: subtree of v cannot reach above u
                if parent[u] != -1 and low[v] >= disc[u]:
                    is_ap[u] = True

            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return [u for u in range(n) if is_ap[u]]


# === Main ===
if __name__ == "__main__":
    # Graph: 0-1-2-3-4 with 1-3 shortcut and isolated 5-6 bridge
    graph = {
        0: [1],
        1: [0, 2, 3],
        2: [1, 3],
        3: [1, 2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5],
    }
    aps = find_articulation_points(graph, 7)
    print(f"Articulation points: {aps}")
```

**Output:**
```
Articulation points: [3, 4, 5]
```

Vertex 3 is an articulation point because removing it disconnects vertex 4 (and beyond) from the rest. Vertices 4 and 5 are articulation points because they form a chain of [bridges](bridges.md) -- removing either splits the graph.

## Worked Example

Consider the graph with edges: $\{0\text{-}1,\ 1\text{-}2,\ 2\text{-}3,\ 3\text{-}1,\ 3\text{-}4,\ 4\text{-}5,\ 5\text{-}6\}$.

| Vertex | disc | low | Parent | children | AP? |
|---|---|---|---|---|---|
| 0 | 0 | 0 | None | 1 | No (root with 1 child) |
| 1 | 1 | 1 | 0 | 2 | No (low[2]=1, low[3]=1, both reach 1) |
| 2 | 2 | 1 | 1 | 1 | No (low[3]=1 < disc[2]=2) |
| 3 | 3 | 1 | 2 | 1 | Yes (low[4]=4 >= disc[3]=3) |
| 4 | 4 | 4 | 3 | 1 | Yes (low[5]=5 >= disc[4]=4) |
| 5 | 5 | 5 | 4 | 1 | Yes (low[6]=6 >= disc[5]=5) |
| 6 | 6 | 6 | 5 | 0 | No (leaf) |

Articulation points: $\{3, 4, 5\}$.

## Relationship to Bridges and Biconnected Components

- An edge whose both endpoints are articulation points is often (but not always) a [bridge](bridges.md).
- Every bridge has at least one endpoint that is an articulation point (unless the bridge connects two isolated vertices).
- Removing all articulation points decomposes the graph into its [biconnected components](components.md).
- The [block-cut tree](block_cut.md) provides a tree representation of how articulation points connect biconnected components.

## Reference

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372-378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
