# TSP via Branch and Bound

The Traveling Salesman Problem (TSP) asks for the shortest tour visiting every city exactly once and returning to the start. With $n!$ possible tours, brute force is infeasible for even moderate $n$. Branch and bound systematically explores partial tours while pruning those whose lower bound exceeds the best known complete tour, often solving instances with tens of cities efficiently.

## Problem Formulation

Given $n$ cities and a distance matrix $d[i][j]$ (the cost of traveling from city $i$ to city $j$), find a permutation $\pi$ of $\{0, 1, \ldots, n-1\}$ that minimizes:

$$
\text{cost}(\pi) = \sum_{k=0}^{n-2} d[\pi(k)][\pi(k+1)] + d[\pi(n-1)][\pi(0)]
$$

## Lower Bound via Row-Column Reduction

A common bounding technique for TSP reduces the cost matrix:

1. **Row reduction**: For each row, subtract its minimum element from all entries. The sum of subtracted values contributes to the lower bound.
2. **Column reduction**: After row reduction, subtract the minimum of each column. Add these to the lower bound.

The resulting **reduced cost matrix** has at least one zero in every row and column. The total reduction gives a lower bound on the tour cost from any node:

$$
\text{LB} = \text{cost so far} + \text{reduction of remaining matrix}
$$

!!! tip "Why Reduction Works"
    Every tour must include exactly one element from each row and each column. Subtracting the row minimum from all entries does not change which tour is optimal, but it shifts the baseline cost. The accumulated shift is a valid lower bound.

## Algorithm

1. Start at a root node representing the start city with the fully reduced cost matrix.
2. For each unvisited city, create a child node:
    - Fix the edge to that city and compute the reduced cost matrix for the remaining subproblem.
    - Compute the lower bound for the child.
3. Prune children whose lower bound exceeds the best known tour.
4. Explore the most promising child (best-first) or use DFS.
5. At leaf nodes (complete tours), update the best tour if improved.

## Complexity

| Aspect | Value |
|---|---|
| Worst-case time | $O(n! \cdot n^2)$ |
| Practical time | Much less with good bounds |
| Space | $O(n^2)$ per node for the reduced matrix |

## Python Implementation

```python
"""
TSP — Branch and Bound with Matrix Reduction.

Uses row and column reduction to compute lower bounds, with DFS
exploration and pruning of unpromising branches.
"""

import math


# === Matrix Reduction ===

def reduce_matrix(matrix: list[list[float]]) -> tuple[list[list[float]], float]:
    """Reduce a cost matrix by row and column minimums.

    Returns (reduced_matrix, reduction_cost).
    """
    n = len(matrix)
    reduced = [row[:] for row in matrix]
    cost = 0.0

    # Row reduction
    for i in range(n):
        finite_vals = [reduced[i][j] for j in range(n) if reduced[i][j] < math.inf]
        if finite_vals:
            row_min = min(finite_vals)
            if row_min > 0:
                cost += row_min
                for j in range(n):
                    if reduced[i][j] < math.inf:
                        reduced[i][j] -= row_min

    # Column reduction
    for j in range(n):
        finite_vals = [reduced[i][j] for i in range(n) if reduced[i][j] < math.inf]
        if finite_vals:
            col_min = min(finite_vals)
            if col_min > 0:
                cost += col_min
                for i in range(n):
                    if reduced[i][j] < math.inf:
                        reduced[i][j] -= col_min

    return reduced, cost


# === Branch and Bound TSP ===

def tsp_branch_bound(dist: list[list[float]]) -> tuple[float, list[int]]:
    """Solve TSP using branch and bound with matrix reduction.

    Args:
        dist: n x n distance matrix.

    Returns:
        (min_cost, tour) where tour is a list of city indices.
    """
    n = len(dist)
    INF = math.inf

    # Initial reduction
    matrix, root_cost = reduce_matrix(dist)
    best_cost = INF
    best_tour: list[int] = []

    def dfs(
        matrix: list[list[float]], cost: float,
        path: list[int], visited: set[int]
    ) -> None:
        nonlocal best_cost, best_tour

        if len(path) == n:
            total = cost + matrix[path[-1]][path[0]]
            if total < best_cost:
                best_cost = total
                best_tour = path[:]
            return

        last = path[-1]
        for city in range(n):
            if city in visited:
                continue

            # Cost of edge from last to city
            edge_cost = matrix[last][city]
            if edge_cost >= INF:
                continue

            # Create reduced matrix for child node
            child_matrix = [row[:] for row in matrix]
            # Block row of last city and column of next city
            for j in range(n):
                child_matrix[last][j] = INF
            for i in range(n):
                child_matrix[i][city] = INF
            child_matrix[city][path[0]] = INF  # prevent premature return

            child_matrix, reduction = reduce_matrix(child_matrix)
            child_cost = cost + edge_cost + reduction

            if child_cost < best_cost:
                path.append(city)
                visited.add(city)
                dfs(child_matrix, child_cost, path, visited)
                path.pop()
                visited.discard(city)

    dfs(matrix, root_cost, [0], {0})
    return best_cost, best_tour


# === Main ===

if __name__ == "__main__":
    dist = [
        [math.inf, 10, 15, 20],
        [10, math.inf, 35, 25],
        [15, 35, math.inf, 30],
        [20, 25, 30, math.inf],
    ]

    cost, tour = tsp_branch_bound(dist)
    tour_str = " -> ".join(str(c) for c in tour) + f" -> {tour[0]}"
    print(f"Minimum cost: {cost}")
    print(f"Tour: {tour_str}")
    # Output:
    # Minimum cost: 80
    # Tour: 0 -> 1 -> 3 -> 2 -> 0
```

## Worked Example

For the 4-city distance matrix above:

1. **Root reduction**: Row mins = $[10, 10, 15, 20]$, column mins after row reduction = $[0, 0, 0, 0]$. Root LB = $55$.
2. **Branch from city 0**: Try edges $0 \to 1$, $0 \to 2$, $0 \to 3$.
3. **Edge $0 \to 1$** (cost 0 in reduced matrix): New LB = $55 + 0 + \text{reduction} = 55$. Promising.
4. Continue branching until complete tours are found. Best tour: $0 \to 1 \to 3 \to 2 \to 0$ with cost 80.

## Reference

- Little, J. D. C., Murty, K. G., Sweeney, D. W., & Karel, C. (1963). An algorithm for the traveling salesman problem. *Operations Research*, 11(6), 972-989.
- Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.), Chapter 9. Springer.
