# BFS Applications

Breadth-first search does more than simply visit every vertex. Because BFS explores vertices in order of increasing distance from the source, it naturally solves several fundamental graph problems. This page presents the most important applications: shortest paths in unweighted graphs, bipartiteness testing, connected-component discovery, and level-order traversal.

## Shortest Paths in Unweighted Graphs

Many graph problems reduce to finding the fewest edges between two vertices. In a social network, for example, the "degrees of separation" between two people is exactly the shortest unweighted path. BFS solves this problem optimally because it explores vertices level by level: the first time it reaches a vertex $v$, it has used the fewest possible edges.

In an unweighted graph every edge has the same cost, so the number of edges on a path equals the path length. Recording the predecessor of each vertex along the way lets us reconstruct the actual shortest path. The time complexity remains $O(V + E)$, where $V$ is the number of vertices and $E$ is the number of edges.

```python
"""
Shortest path in an unweighted graph using BFS.

Demonstrates how BFS naturally computes shortest distances
and how predecessor tracking enables path reconstruction.
"""

from collections import deque

# === Shortest path via BFS ===================================================

def bfs_shortest_path(graph, source, target):
    """Return the shortest path from source to target in an unweighted graph.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.
    source : int
        Starting vertex.
    target : int
        Destination vertex.

    Returns
    -------
    list[int] | None
        Shortest path as a list of vertices, or None if unreachable.
    """
    if source == target:
        return [source]

    visited = {source}
    predecessor = {source: None}
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                predecessor[neighbor] = node
                if neighbor == target:
                    # Reconstruct path by following predecessors
                    path = []
                    current = target
                    while current is not None:
                        path.append(current)
                        current = predecessor[current]
                    return path[::-1]
                queue.append(neighbor)
    return None


# === Main =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2], 4: [2]}
    path = bfs_shortest_path(graph, 0, 3)
    print(f"Shortest path from 0 to 3: {path}")
    print(f"Distance: {len(path) - 1} edges")
```

**Output:**
```
Shortest path from 0 to 3: [0, 1, 3]
Distance: 2 edges
```

The predecessor-based reconstruction uses $O(V)$ memory rather than storing full paths in the queue, which keeps the overall space complexity at $O(V)$.

## Bipartiteness Testing

Many practical problems require splitting objects into two groups with constraints between them. Job assignment, graph coloring with two colors, and conflict-free scheduling all depend on whether the underlying graph is bipartite. BFS provides a simple linear-time test for this property.

A graph is **bipartite** if its vertex set can be partitioned into two groups such that every edge connects a vertex in one group to a vertex in the other. Equivalently, a graph is bipartite if and only if it contains no odd-length cycle. The BFS-based algorithm assigns alternating colors as it traverses each level. If an edge ever connects two vertices of the same color, the graph is not bipartite.

```python
"""
Bipartiteness testing using BFS two-coloring.

If BFS can assign two colors so that no edge connects same-color
vertices, the graph is bipartite.
"""

from collections import deque

# === Bipartite check ==========================================================

def is_bipartite(graph):
    """Check whether an undirected graph is bipartite.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    bool
        True if the graph is bipartite, False otherwise.
    """
    color = {}
    for start in graph:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True


# === Main =====================================================================

if __name__ == "__main__":
    bipartite_graph = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    non_bipartite = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    print(f"Square cycle bipartite? {is_bipartite(bipartite_graph)}")
    print(f"Triangle bipartite? {is_bipartite(non_bipartite)}")
```

**Output:**
```
Square cycle bipartite? True
Triangle bipartite? False
```

The square cycle (4 vertices forming a cycle) is bipartite because the two color classes are $\{0, 2\}$ and $\{1, 3\}$. The triangle has three vertices all connected to each other, which forces an odd-length cycle, making it non-bipartite.

## Connected Components

Understanding which vertices can reach each other is often the first step in analyzing a graph. In a network of computers, for instance, connected components correspond to groups of machines that can communicate. BFS provides a direct way to discover these groups.

In an undirected graph, a **connected component** is a maximal set of vertices such that a path exists between every pair. BFS from any unvisited vertex discovers its entire component. By iterating over all vertices and launching BFS from each unvisited one, we enumerate every component in $O(V + E)$ total time.

```python
"""
Connected component discovery using BFS.

Each BFS call from an unvisited vertex discovers one full component.
"""

from collections import deque

# === Connected components =====================================================

def connected_components(graph):
    """Find all connected components of an undirected graph.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.

    Returns
    -------
    list[list[int]]
        Each inner list contains the vertices of one component.
    """
    visited = set()
    components = []

    for vertex in graph:
        if vertex not in visited:
            component = []
            queue = deque([vertex])
            visited.add(vertex)
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

    return components


# === Main =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1], 1: [0],
        2: [3, 4], 3: [2, 4], 4: [2, 3],
        5: [],
    }
    comps = connected_components(graph)
    for i, comp in enumerate(comps):
        print(f"Component {i}: {comp}")
```

**Output:**
```
Component 0: [0, 1]
Component 1: [2, 3, 4]
Component 2: [5]
```

## Level-Order Traversal

Beyond finding paths and components, BFS is the natural choice for any problem that requires grouping vertices by their distance from the source. Level-order traversal explicitly separates vertices into distance classes and is especially useful in tree algorithms where each level corresponds to a depth in the tree.

```python
"""
Level-order traversal using BFS.

Groups vertices by their distance (level) from the source.
"""

from collections import deque

# === Level-order traversal ====================================================

def level_order(graph, source):
    """Return vertices grouped by BFS level.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation.
    source : int
        Starting vertex.

    Returns
    -------
    list[list[int]]
        Each inner list holds the vertices at that distance from source.
    """
    visited = {source}
    queue = deque([source])
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        levels.append(current_level)

    return levels


# === Main =====================================================================

if __name__ == "__main__":
    graph = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1], 4: [1], 5: [2]}
    levels = level_order(graph, 0)
    for depth, lvl in enumerate(levels):
        print(f"Level {depth}: {lvl}")
```

**Output:**
```
Level 0: [0]
Level 1: [1, 2]
Level 2: [3, 4, 5]
```

## Summary of BFS Applications

| Application | Key Idea | Time |
|---|---|---|
| Shortest path (unweighted) | First visit = fewest edges | $O(V + E)$ |
| Bipartiteness testing | Two-color by BFS level | $O(V + E)$ |
| Connected components | BFS from each unvisited vertex | $O(V + E)$ |
| Level-order traversal | Group by distance from source | $O(V + E)$ |

All four applications run in linear time because each builds directly on a single BFS pass (or a constant number of passes over the graph).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
