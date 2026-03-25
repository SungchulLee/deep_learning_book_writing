# Bipartite Graphs

Many real-world relationships are naturally two-sided: students enroll in courses, workers are assigned to tasks, and applicants are matched to positions. In each case, the entities fall into two distinct groups, and connections only run between groups, never within them. Bipartite graphs formalize this structure, and their characterization through odd cycles provides one of the most elegant results in graph theory.

## Definition

A graph $G = (V, E)$ is **bipartite** if its vertex set $V$ can be partitioned into two disjoint, non-empty sets $L$ and $R$ such that every edge $e \in E$ connects a vertex in $L$ to a vertex in $R$. Formally,

$$
V = L \cup R, \quad L \cap R = \emptyset, \quad \text{and} \quad \forall (u, v) \in E: \; u \in L \Leftrightarrow v \in R
$$

No edge connects two vertices within the same partition. The pair $(L, R)$ is called a **bipartition** of $G$.

!!! example "A Simple Bipartite Graph"
    Consider $V = \{1, 2, 3, 4, 5, 6\}$ with edges $\{(1,4), (1,5), (2,5), (2,6), (3,6)\}$. Setting $L = \{1, 2, 3\}$ and $R = \{4, 5, 6\}$, every edge crosses between $L$ and $R$, so the graph is bipartite.

## Complete Bipartite Graphs

The **complete bipartite graph** $K_{m,n}$ has partition sizes $|L| = m$ and $|R| = n$, with every vertex in $L$ adjacent to every vertex in $R$. The total number of edges is

$$
|E(K_{m,n})| = m \cdot n
$$

The graph $K_{1,n}$ is called a **star graph** because one central vertex connects to all $n$ others.

## Odd-Cycle Characterization

The following theorem provides the fundamental characterization of bipartite graphs and connects the concept to cycle structure.

!!! tip "Theorem: Bipartite Characterization"
    A graph $G$ is bipartite if and only if $G$ contains no odd-length cycle.

**Proof sketch.**

$(\Rightarrow)$ Suppose $G$ is bipartite with partition $(L, R)$. Consider any cycle $v_0, v_1, \ldots, v_k = v_0$. Since adjacent vertices alternate between $L$ and $R$, the cycle must return to its starting partition after an even number of steps. Therefore $k$ is even, and no odd-length cycle exists.

$(\Leftarrow)$ Suppose $G$ has no odd-length cycle. Without loss of generality, assume $G$ is connected (otherwise, apply the argument to each connected component). Pick any vertex $s$ and run BFS from $s$. Define

$$
L = \{v \in V : d(s, v) \text{ is even}\}, \quad R = \{v \in V : d(s, v) \text{ is odd}\}
$$

where $d(s, v)$ is the shortest-path distance from $s$ to $v$. If some edge $(u, v)$ has both endpoints in $L$ (or both in $R$), then the BFS tree paths from $s$ to $u$ and $s$ to $v$, together with edge $(u, v)$, form an odd-length cycle -- contradicting the hypothesis. Therefore $(L, R)$ is a valid bipartition. $\square$

## Testing Bipartiteness with 2-Coloring

The proof above immediately yields a practical algorithm: assign colors based on BFS distance parity. If no edge connects two same-colored vertices, the graph is bipartite; otherwise, an odd cycle exists.

The algorithm runs in $O(V + E)$ time, the same cost as BFS.

```python
"""
Bipartite graph testing using BFS-based 2-coloring.

Determines whether an undirected graph is bipartite by attempting
to assign one of two colors to each vertex such that no adjacent
vertices share the same color.
"""

from collections import deque


# === Graph Representation ===

def build_adjacency_list(n, edges):
    """Build an adjacency list from a list of undirected edges."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# === Bipartiteness Test ===

def is_bipartite(adj, n):
    """
    Test whether the graph is bipartite using BFS 2-coloring.

    Returns (True, color) if bipartite, where color[v] in {0, 1}
    gives the partition assignment. Returns (False, color) if an
    odd cycle is detected.
    """
    color = [-1] * n

    for start in range(n):
        if color[start] != -1:
            continue
        # BFS from each unvisited component
        color[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False, color

    return True, color


# === Main ===

if __name__ == "__main__":
    # Example 1: bipartite graph
    edges1 = [(0, 3), (0, 4), (1, 4), (1, 5), (2, 5)]
    adj1 = build_adjacency_list(6, edges1)
    result1, colors1 = is_bipartite(adj1, 6)
    print(f"Graph 1 bipartite: {result1}")
    if result1:
        L = [v for v in range(6) if colors1[v] == 0]
        R = [v for v in range(6) if colors1[v] == 1]
        print(f"  Partition L: {L}, R: {R}")

    # Example 2: triangle (odd cycle) -> not bipartite
    edges2 = [(0, 1), (1, 2), (2, 0)]
    adj2 = build_adjacency_list(3, edges2)
    result2, _ = is_bipartite(adj2, 3)
    print(f"Graph 2 (triangle) bipartite: {result2}")
```

**Output:**
```
Graph 1 bipartite: True
  Partition L: [0, 1, 2], R: [3, 4, 5]
Graph 2 (triangle) bipartite: False
```

## Properties of Bipartite Graphs

Several useful properties follow directly from the definition and characterization theorem.

- **Maximum edges.** A bipartite graph on $n$ vertices has at most $\lfloor n^2 / 4 \rfloor$ edges, achieved by $K_{\lfloor n/2 \rfloor, \lceil n/2 \rceil}$.
- **Every tree is bipartite.** Trees are connected and acyclic, so they trivially contain no odd cycle.
- **Independent sets.** Each partition $L$ and $R$ is an independent set (no internal edges), so the independence number satisfies $\alpha(G) \geq \max(|L|, |R|)$.
- **Chromatic number.** A non-empty bipartite graph has chromatic number exactly 2, meaning it is 2-colorable.

!!! warning "Common Pitfall"
    A graph can be bipartite even if it is disconnected. The bipartition is applied independently to each connected component. The algorithm above handles this by iterating over all unvisited starting vertices.

## Applications

Bipartite graphs appear throughout computer science and operations research:

- **Matching problems.** The Hungarian algorithm and Hopcroft-Karp algorithm find maximum matchings in bipartite graphs, with applications to job assignment and resource allocation.
- **Network flow.** Many maximum-flow problems reduce to bipartite matching through the construction of flow networks.
- **Scheduling.** Tasks assigned to time slots or machines naturally form bipartite structures.
- **Recommendation systems.** User-item interaction graphs are bipartite, forming the basis for collaborative filtering.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
- West, D. B. (2001). *Introduction to Graph Theory* (2nd ed.). Prentice Hall. Section 1.2.
