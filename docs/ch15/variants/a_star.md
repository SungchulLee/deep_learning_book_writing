# A* Search

Dijkstra's algorithm explores vertices in all directions equally, which can be wasteful when the goal is to reach a specific target. A* search improves on Dijkstra by incorporating a **heuristic function** that estimates the remaining distance to the target, guiding the search toward the goal and often exploring far fewer vertices. A* is the standard algorithm for pathfinding in games, robotics, and route planning.

## The A* Evaluation Function

A* maintains a priority queue ordered by the evaluation function

$$
f(n) = g(n) + h(n)
$$

where $g(n)$ is the actual cost of the cheapest known path from the source $s$ to vertex $n$, and $h(n)$ is a heuristic estimate of the cost from $n$ to the target $t$. The algorithm always expands the vertex with the smallest $f$ value.

- When $h(n) = 0$ for all $n$, A* degenerates to Dijkstra's algorithm.
- When $h(n)$ accurately reflects the true remaining cost, A* heads straight toward the target.

## Admissibility and Optimality

A heuristic $h$ is **admissible** if it never overestimates the true cost to the goal:

$$
h(n) \leq h^*(n) \quad \text{for all } n
$$

where $h^*(n)$ is the true shortest-path cost from $n$ to $t$.

!!! tip "Admissibility guarantees optimality"
    If $h$ is admissible, A* is guaranteed to find an optimal (shortest) path from $s$ to $t$. This follows because any suboptimal path to $t$ has $f > f^*$ (the optimal cost), so A* will expand the optimal path first.

## Consistency (Monotonicity)

A heuristic $h$ is **consistent** (also called monotone) if for every edge $(u, v)$ with cost $w(u, v)$:

$$
h(u) \leq w(u, v) + h(v)
$$

Consistency implies admissibility (but not vice versa). With a consistent heuristic, the $f$ values along any path are non-decreasing, and A* never needs to re-open a closed vertex. This makes the algorithm more efficient in practice.

## Algorithm

1. Initialize: set $g(s) = 0$, $f(s) = h(s)$, and add $s$ to the priority queue.
2. While the priority queue is not empty:
    - Extract the vertex $n$ with the smallest $f(n)$.
    - If $n = t$, reconstruct and return the path.
    - For each neighbor $v$ of $n$: if $g(n) + w(n, v) < g(v)$, update $g(v)$ and $f(v) = g(v) + h(v)$, and add $v$ to the queue.
3. If the queue empties without reaching $t$, no path exists.

## Complexity

With a consistent heuristic, A* expands each vertex at most once. In the worst case (uninformative heuristic), A* expands all $V$ vertices and examines all $E$ edges, giving the same $O((V + E) \log V)$ complexity as Dijkstra with a binary heap. In practice, a good heuristic dramatically reduces the number of expanded vertices.

## Implementation

```python
"""
A* search algorithm for weighted graphs.

Uses a heuristic function to guide search toward the target,
reducing the number of vertices explored compared to Dijkstra.
"""

import heapq

# === A* search ================================================================

def a_star(graph, source, target, heuristic):
    """Find the shortest path using A* search.

    Parameters
    ----------
    graph : dict[int, list[tuple[int, float]]]
        Adjacency list with (neighbor, weight) pairs.
    source : int
        Starting vertex.
    target : int
        Goal vertex.
    heuristic : dict[int, float]
        Estimated cost from each vertex to the target.

    Returns
    -------
    tuple[list[int], float] | tuple[None, float]
        (path, cost) if found, (None, inf) otherwise.
    """
    g_cost = {source: 0.0}
    f_cost = {source: heuristic.get(source, 0.0)}
    predecessor = {source: None}
    open_set = [(f_cost[source], source)]
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = predecessor[node]
            return path[::-1], g_cost[target]

        if current in closed:
            continue
        closed.add(current)

        for neighbor, weight in graph[current]:
            if neighbor in closed:
                continue
            tentative_g = g_cost[current] + weight
            if tentative_g < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + heuristic.get(neighbor, 0.0)
                predecessor[neighbor] = current
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))

    return None, float("inf")


# === Main =====================================================================

if __name__ == "__main__":
    # Weighted graph: vertex -> [(neighbor, weight), ...]
    graph = {
        0: [(1, 1.0), (2, 4.0)],
        1: [(2, 2.0), (3, 5.0)],
        2: [(3, 1.0)],
        3: [],
    }

    # Heuristic: estimated distance from each vertex to target (vertex 3)
    h = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}

    path, cost = a_star(graph, 0, 3, h)
    print(f"A* path: {path}")
    print(f"Total cost: {cost}")
```

**Output:**
```
A* path: [0, 1, 2, 3]
Total cost: 4.0
```

The heuristic guides A* to find the optimal path $0 \to 1 \to 2 \to 3$ with cost 4.0 without exploring the expensive direct edge $0 \to 2$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 24-25. MIT Press.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.
