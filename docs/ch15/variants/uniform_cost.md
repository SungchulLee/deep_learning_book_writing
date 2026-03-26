# Uniform Cost Search

Breadth-first search finds the shortest path when all edge weights are equal, but fails on weighted graphs because it expands vertices by hop count rather than by accumulated cost. **Uniform cost search (UCS)** fixes this by expanding the vertex with the smallest cumulative path cost first, using a priority queue instead of a FIFO queue. UCS is equivalent to Dijkstra's algorithm but is typically presented in the AI/search literature with a goal test and an explored set. When all edge weights are equal, UCS reduces to BFS.

## Algorithm

UCS explores vertices in order of their path cost from the source $s$:

1. Initialize a priority queue with $(0, s)$ (cost, vertex).
2. Initialize an explored set (empty).
3. While the priority queue is not empty:
    - Extract the vertex $u$ with the smallest cost $g(u)$.
    - If $u$ is the goal, return $g(u)$ and the path.
    - If $u$ is in the explored set, skip it.
    - Add $u$ to the explored set.
    - For each neighbor $v$ of $u$ with edge weight $w(u,v)$:
        - If $v$ is not explored, insert $(g(u) + w(u,v),\, v)$ into the priority queue.
4. If the priority queue is exhausted, no path exists.

## Relationship to Dijkstra's Algorithm

UCS and Dijkstra's algorithm compute the same shortest paths. The differences are in framing:

| Property | UCS | Dijkstra |
|----------|-----|----------|
| Goal test | Stops at goal | Computes all distances |
| Literature | AI / search | Graph algorithms |
| Priority queue | Standard | Often with decrease-key |
| Heuristic | None ($h = 0$) | None |

UCS is also A* search with $h(n) = 0$ for all $n$, since the evaluation function becomes $f(n) = g(n) + 0 = g(n)$.

## Correctness

UCS is **optimal** (finds shortest paths) when all edge weights are non-negative, $w(u,v) \geq 0$. The proof mirrors Dijkstra's correctness proof: when a vertex is extracted from the priority queue, its distance is final because any alternative path would pass through a vertex with equal or greater cost that has not yet been expanded.

## Complexity

Let $C^*$ be the cost of the optimal solution and $\epsilon$ be the minimum edge weight:

$$
\text{Time and space: } O(b^{1 + \lfloor C^*/\epsilon \rfloor})
$$

where $b$ is the branching factor. With a binary heap:

$$
\text{Time: } O((|V| + |E|) \log |V|)
$$

## Implementation

```python
"""
Uniform cost search for shortest paths in weighted graphs.

Expands vertices in order of cumulative path cost using a
priority queue. Equivalent to Dijkstra's algorithm with
goal-directed termination.
"""

import heapq


# === Uniform Cost Search ===

def uniform_cost_search(graph: dict, source: int,
                        goal: int) -> tuple:
    """Find shortest path from source to goal using UCS.

    Args:
        graph: Adjacency list {u: [(v, weight), ...]}.
        source: Starting vertex.
        goal: Target vertex.

    Returns:
        (cost, path) tuple. Returns (float('inf'), []) if
        no path exists.
    """
    frontier = [(0, source, [source])]
    explored = set()

    while frontier:
        cost, u, path = heapq.heappop(frontier)

        if u == goal:
            return cost, path

        if u in explored:
            continue
        explored.add(u)

        for v, weight in graph.get(u, []):
            if v not in explored:
                new_cost = cost + weight
                heapq.heappush(frontier, (new_cost, v, path + [v]))

    return float('inf'), []


# === Demonstration ===

if __name__ == "__main__":
    # Weighted directed graph
    graph = {
        0: [(1, 4), (2, 2)],
        1: [(3, 5)],
        2: [(1, 1), (3, 8), (4, 10)],
        3: [(4, 2)],
        4: []
    }

    print("Graph:")
    for u, neighbors in sorted(graph.items()):
        for v, w in neighbors:
            print(f"  {u} -> {v} (weight {w})")
    print()

    # Find shortest path from 0 to 4
    cost, path = uniform_cost_search(graph, 0, 4)
    print(f"Shortest path 0 -> 4: cost={cost}, path={path}")
    print()

    # Find shortest path from 0 to 3
    cost, path = uniform_cost_search(graph, 0, 3)
    print(f"Shortest path 0 -> 3: cost={cost}, path={path}")
    print()

    # No path case
    graph_disconnected = {0: [(1, 1)], 1: [], 2: [(3, 1)], 3: []}
    cost, path = uniform_cost_search(graph_disconnected, 0, 3)
    print(f"Disconnected 0 -> 3: cost={cost}, path={path}")
```

**Output:**
```
Graph:
  0 -> 1 (weight 4)
  0 -> 2 (weight 2)
  1 -> 3 (weight 5)
  2 -> 1 (weight 1)
  2 -> 3 (weight 8)
  2 -> 4 (weight 10)
  3 -> 4 (weight 2)
  4 ->

Shortest path 0 -> 4: cost=10, path=[0, 2, 1, 3, 4]

Shortest path 0 -> 3: cost=8, path=[0, 2, 1, 3]

Disconnected 0 -> 3: cost=inf, path=[]
```

!!! warning "Negative Edge Weights"
    UCS (like Dijkstra) does not handle negative edge weights. If a graph contains negative-weight edges, use the Bellman-Ford algorithm instead, which detects negative cycles and correctly computes shortest paths in $O(|V| \cdot |E|)$ time.

## Reference

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.), Chapter 3. Pearson.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 24. MIT Press.
