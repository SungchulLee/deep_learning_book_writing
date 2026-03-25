# Kahn's Algorithm

While the [DFS-based approach](dfs.md) discovers topological order through finish times, Kahn's algorithm takes a more direct route: it repeatedly identifies vertices with no incoming edges, outputs them, and removes their outgoing edges. This "peel off the sources" strategy mirrors how we naturally reason about dependencies -- start with tasks that have no prerequisites, complete them, and unlock the next layer.

## Intuition

Consider a DAG representing course prerequisites. Some courses have no prerequisites at all -- these are the sources. Once a student completes all source courses, new courses become available (their prerequisites are now satisfied). Kahn's algorithm formalizes this layer-by-layer process.

The algorithm maintains the **in-degree** of each vertex: the count of incoming edges. A vertex with in-degree zero is a source that can be placed next in the topological order. When we "remove" a source, we decrement the in-degrees of its neighbors, potentially creating new sources.

## Algorithm

**Input:** A directed graph $G = (V, E)$ with $|V| = n$ and $|E| = m$.

**Output:** A topological ordering of $V$, or a report that $G$ contains a cycle.

1. Compute the in-degree of every vertex.
2. Initialize a queue $Q$ with all vertices having in-degree zero.
3. While $Q$ is not empty:
    - Dequeue a vertex $u$ and append it to the output list.
    - For each neighbor $v$ of $u$, decrement $\text{in-degree}(v)$ by 1.
    - If $\text{in-degree}(v)$ becomes zero, enqueue $v$.
4. If the output list contains all $n$ vertices, return it as the topological order. Otherwise, $G$ contains a cycle.

## Correctness

!!! note "Why Kahn's Algorithm Produces a Valid Topological Order"
    **Claim:** If $G$ is a DAG, Kahn's algorithm outputs a valid topological ordering.

    **Proof.** We show that for every edge $(u, v) \in E$, vertex $u$ appears before $v$ in the output. When $u$ is dequeued in step 3, the algorithm decrements $\text{in-degree}(v)$. Before this point, $v$ cannot have in-degree zero (since the edge from $u$ contributes to $v$'s in-degree), so $v$ has not yet been dequeued. Therefore $u$ precedes $v$ in the output. $\square$

!!! note "Cycle Detection"
    **Claim:** If $G$ contains a cycle, Kahn's algorithm processes fewer than $n$ vertices.

    **Proof.** Every vertex in a cycle always has at least one predecessor that is also in the cycle. Since no vertex in the cycle ever reaches in-degree zero, none of them are enqueued. The output list therefore omits at least the vertices in the cycle. $\square$

## Complexity

Each vertex is enqueued and dequeued exactly once, and each edge is examined exactly once (when its source is dequeued). Therefore:

$$
T(V, E) = O(V + E)
$$

The space complexity is $O(V)$ for the in-degree array and the queue.

## Implementation

```python
"""
Kahn's algorithm for topological sorting.

Uses iterative source removal (BFS on in-degree-zero vertices) to produce
a topological ordering of a directed acyclic graph, with built-in cycle
detection.
"""

from collections import deque


# === Kahn's Topological Sort ===
def kahn_topo_sort(graph, n):
    """
    Compute topological ordering using Kahn's algorithm.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a directed graph with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[int]
        Vertices in topological order. Empty list if a cycle exists.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph.get(u, []):
            in_degree[v] += 1

    queue = deque(v for v in range(n) if in_degree[v] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != n:
        return []  # cycle detected
    return order


# === Main ===
if __name__ == "__main__":
    # DAG: 0 -> 1 -> 3, 0 -> 2 -> 3 -> 4
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    result = kahn_topo_sort(dag, 5)
    print(f"Topological order: {result}")

    # Verify validity
    pos = {v: i for i, v in enumerate(result)}
    valid = all(pos[u] < pos[v] for u in dag for v in dag[u])
    print(f"Valid topological order: {valid}")

    # Graph with a cycle: 0 -> 1 -> 2 -> 0
    cyclic = {0: [1], 1: [2], 2: [0]}
    result_cyclic = kahn_topo_sort(cyclic, 3)
    print(f"Cyclic graph result: {result_cyclic}")
```

**Output:**
```
Topological order: [0, 1, 2, 3, 4]
Valid topological order: True
Cyclic graph result: []
```

## Lexicographically Smallest Topological Order

A useful variant replaces the queue with a min-heap (priority queue). This ensures that among all valid choices of the next source vertex, the algorithm always picks the smallest-labeled one. The result is the **lexicographically smallest** topological ordering.

```python
"""
Kahn's algorithm variant producing the lexicographically smallest
topological ordering by using a min-heap instead of a plain queue.
"""

import heapq


# === Lexicographic Kahn's Sort ===
def kahn_lex_smallest(graph, n):
    """
    Compute the lexicographically smallest topological ordering.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a DAG with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[int]
        Lexicographically smallest topological order.
    """
    in_degree = [0] * n
    for u in range(n):
        for v in graph.get(u, []):
            in_degree[v] += 1

    heap = [v for v in range(n) if in_degree[v] == 0]
    heapq.heapify(heap)
    order = []

    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(heap, v)

    return order


# === Main ===
if __name__ == "__main__":
    dag = {0: [2, 1], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"Lex smallest order: {kahn_lex_smallest(dag, 5)}")
```

**Output:**
```
Lex smallest order: [0, 1, 2, 3, 4]
```

The heap variant runs in $O((V + E) \log V)$ due to the heap operations.

## Comparison with DFS-Based Sort

| Property | Kahn's Algorithm | DFS-Based |
|---|---|---|
| Strategy | Iterative source removal | Reverse finish-time ordering |
| Data structure | Queue (or heap) | Recursion stack |
| Cycle detection | Incomplete output (fewer than $n$ vertices) | Back edge during traversal |
| Lexicographic order | Easy with a min-heap | Requires additional work |
| Parallelism | Sources in the same "layer" are independent | Sequential by nature |

For applications requiring a specific ordering among valid choices (such as the lexicographically smallest), Kahn's algorithm with a priority queue is the standard approach. For general topological sorting, both methods are equally efficient at $O(V + E)$.

## Reference

- Kahn, A. B. (1962). Topological sorting of large networks. *Communications of the ACM*, 5(11), 558-562.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
