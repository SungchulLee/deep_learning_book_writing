# K Shortest Paths

In many applications, knowing only the single shortest path is not enough. A navigation system might offer alternative routes to avoid traffic. A network may need backup paths for fault tolerance. The **$k$ shortest paths problem** asks for the $k$ shortest paths from a source $s$ to a target $t$, ranked by total cost. The paths may share edges (loopy variant) or be required to be edge-disjoint or vertex-simple depending on the variant.

## Problem Variants

- **$k$ shortest simple paths**: each path visits no vertex more than once. NP-hard in general, but tractable with Yen's algorithm for moderate $k$.
- **$k$ shortest walks**: paths may repeat vertices and edges. Solvable in polynomial time using Eppstein's algorithm.

Most practical algorithms focus on the simple-path variant.

## Yen's Algorithm

Yen's algorithm (1971) finds the $k$ shortest simple (loopless) paths from $s$ to $t$:

1. Find the shortest path $P_1$ using Dijkstra.
2. For $i = 2, 3, \ldots, k$:
    - For each vertex $v_j$ in $P_{i-1}$ (the spur node), temporarily remove edges used by previously found paths that share the same prefix up to $v_j$.
    - Find the shortest path from $v_j$ to $t$ in the modified graph (the spur path).
    - Concatenate the root path ($s$ to $v_j$) with the spur path to get a candidate.
    - Add all candidates to a min-heap.
    - Extract the minimum-cost candidate as $P_i$.

## Complexity of Yen's Algorithm

Each of the $k$ iterations runs Dijkstra at most $|P|$ times (where $|P|$ is the path length):

$$
O(kn(m + n \log n))
$$

where $n = |V|$ and $m = |E|$. This uses Dijkstra with a Fibonacci heap.

## Eppstein's Algorithm

Eppstein's algorithm (1998) finds $k$ shortest walks (allowing repeated vertices) in:

$$
O(m + n \log n + k \log k)
$$

It builds a compact implicit representation of all paths using a **path graph**, then extracts them in order. This is significantly faster than Yen's algorithm but does not guarantee simple paths.

## Implementation

```python
"""
K shortest simple paths using Yen's algorithm.

Finds the k shortest loopless paths from source to target
by iteratively deviating from previously found paths.
Uses Dijkstra's algorithm for single-source shortest paths.
"""

import heapq
from collections import defaultdict


# === Dijkstra's Algorithm ===

def dijkstra(graph: dict, source: int, target: int,
             blocked_edges: set = None,
             blocked_nodes: set = None) -> tuple:
    """Find shortest path from source to target.

    Returns (cost, path) or (float('inf'), []) if no path exists.
    """
    if blocked_edges is None:
        blocked_edges = set()
    if blocked_nodes is None:
        blocked_nodes = set()

    dist = {source: 0}
    prev = {source: None}
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        if u == target:
            break
        for v, w in graph.get(u, []):
            if v in blocked_nodes or (u, v) in blocked_edges:
                continue
            new_dist = d + w
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    if target not in dist:
        return float('inf'), []

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    return dist[target], path[::-1]


# === Yen's Algorithm ===

def yen_k_shortest(graph: dict, source: int, target: int,
                   k: int) -> list:
    """Find k shortest simple paths using Yen's algorithm.

    Returns a list of (cost, path) tuples sorted by cost.
    """
    # First shortest path
    cost, path = dijkstra(graph, source, target)
    if not path:
        return []

    a_paths = [(cost, path)]
    candidates = []
    candidate_set = set()

    for i in range(1, k):
        prev_path = a_paths[i - 1][1]

        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[:j + 1]
            root_cost = 0
            for idx in range(len(root_path) - 1):
                u, v = root_path[idx], root_path[idx + 1]
                for nb, w in graph.get(u, []):
                    if nb == v:
                        root_cost += w
                        break

            # Block edges of paths sharing the same root
            blocked_edges = set()
            for _, p in a_paths:
                if p[:j + 1] == root_path and j + 1 < len(p):
                    blocked_edges.add((p[j], p[j + 1]))

            blocked_nodes = set(root_path[:-1])

            spur_cost, spur_path = dijkstra(
                graph, spur_node, target,
                blocked_edges, blocked_nodes
            )

            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = root_cost + spur_cost
                path_tuple = tuple(total_path)
                if path_tuple not in candidate_set:
                    candidate_set.add(path_tuple)
                    heapq.heappush(candidates, (total_cost, total_path))

        if not candidates:
            break

        next_cost, next_path = heapq.heappop(candidates)
        a_paths.append((next_cost, next_path))

    return a_paths


# === Demonstration ===

if __name__ == "__main__":
    # Build a small weighted graph (adjacency list)
    graph = defaultdict(list)
    edges = [
        (0, 1, 1), (0, 2, 5), (1, 2, 2), (1, 3, 6),
        (2, 3, 2), (2, 4, 7), (3, 4, 1), (0, 3, 8)
    ]
    for u, v, w in edges:
        graph[u].append((v, w))

    print("Graph edges:", edges)
    print()

    k = 4
    paths = yen_k_shortest(graph, 0, 4, k)
    print(f"Top {k} shortest paths from 0 to 4:")
    for i, (cost, path) in enumerate(paths, 1):
        print(f"  #{i}: cost={cost}, path={path}")
```

**Output:**
```
Graph edges: [(0, 1, 1), (0, 2, 5), (1, 2, 2), (1, 3, 6), (2, 3, 2), (2, 4, 7), (3, 4, 1), (0, 3, 8)]

Top 4 shortest paths from 0 to 4:
  #1: cost=6, path=[0, 1, 2, 3, 4]
  #2: cost=8, path=[0, 1, 3, 4]
  #3: cost=8, path=[0, 2, 3, 4]
  #4: cost=9, path=[0, 3, 4]
```

## Comparison

| Algorithm | Paths Found | Time | Simple Paths? |
|-----------|-------------|------|---------------|
| Yen (1971) | $k$ shortest | $O(kn(m + n \log n))$ | Yes |
| Eppstein (1998) | $k$ shortest | $O(m + n \log n + k \log k)$ | No (walks) |
| Lawler (1972) | $k$ shortest | $O(kn(m + n \log n))$ | Yes |

## Reference

- Yen, J. Y. (1971). Finding the $K$ shortest loopless paths in a network. *Management Science*, 17(11), 712-716.
- Eppstein, D. (1998). Finding the $k$ shortest paths. *SIAM Journal on Computing*, 28(2), 652-673.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
