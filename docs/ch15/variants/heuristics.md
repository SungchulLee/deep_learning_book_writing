# Heuristic Functions

Informed search algorithms like A* depend on a **heuristic function** $h(n)$ that estimates the cost from vertex $n$ to the goal. The quality of this estimate determines how efficiently the search proceeds: a perfect heuristic leads A* straight to the goal, while a trivial heuristic ($h = 0$) reduces A* to Dijkstra's algorithm. This page formalizes the key properties of heuristic functions and presents the most common heuristics used in practice.

## Admissibility

A heuristic $h$ is **admissible** if it never overestimates the true cost to reach the goal:

$$
0 \leq h(n) \leq h^*(n) \quad \text{for all vertices } n
$$

where $h^*(n)$ is the actual shortest-path cost from $n$ to the goal. Admissibility is the minimum requirement for A* to guarantee an optimal solution.

!!! tip "Why admissibility matters"
    If $h$ overestimates for some vertex $n$, A* might avoid paths through $n$ in favor of seemingly cheaper alternatives, potentially missing the true shortest path.

## Consistency (Monotonicity)

A heuristic $h$ is **consistent** if for every edge $(u, v)$ with cost $w(u, v)$:

$$
h(u) \leq w(u, v) + h(v)
$$

and $h(\text{goal}) = 0$. This is the triangle inequality applied to the heuristic. Consistency implies admissibility, but the converse is not always true.

With a consistent heuristic, the $f$ values along any optimal path are non-decreasing. This means that once A* expands a vertex, it has found the optimal cost to that vertex, so the vertex never needs to be re-opened. Consistency therefore makes A* more efficient and simpler to implement.

## Dominance

Given two admissible heuristics $h_1$ and $h_2$, we say $h_2$ **dominates** $h_1$ if

$$
h_1(n) \leq h_2(n) \leq h^*(n) \quad \text{for all } n
$$

A dominating heuristic is always preferable because it provides tighter lower bounds, causing A* to expand fewer vertices. When multiple admissible heuristics are available, using $h(n) = \max(h_1(n), h_2(n))$ produces a new admissible heuristic that dominates both.

## Common Heuristics for Grid Graphs

In grid-based pathfinding (games, robotics), the following heuristics are widely used.

**Manhattan distance** (4-directional movement). For positions $(x_1, y_1)$ and $(x_2, y_2)$:

$$
h_{\text{Manhattan}} = |x_1 - x_2| + |y_1 - y_2|
$$

This is admissible and consistent when movement is restricted to horizontal and vertical steps of uniform cost.

**Euclidean distance** (any-angle movement):

$$
h_{\text{Euclidean}} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

Admissible for any movement model where diagonal moves are allowed, since the straight-line distance is the shortest possible path.

**Chebyshev distance** (8-directional with uniform cost):

$$
h_{\text{Chebyshev}} = \max(|x_1 - x_2|, |y_1 - y_2|)
$$

Admissible and consistent when diagonal and cardinal moves have the same cost.

## Heuristic Quality and Search Performance

| Heuristic Quality | A* Behavior | Vertices Expanded |
|---|---|---|
| $h = 0$ | Reduces to Dijkstra | All reachable vertices |
| $h = h^*$ (perfect) | Expands only optimal path | Minimal |
| $0 < h < h^*$ (informative) | Guided toward goal | Between the two extremes |
| $h > h^*$ (inadmissible) | No optimality guarantee | May find suboptimal path |

## Implementation

```python
"""
Common heuristic functions for grid-based pathfinding.

Demonstrates Manhattan, Euclidean, and Chebyshev distances
and their use with A* search.
"""

import math

# === Heuristic Functions ===

def manhattan(a: tuple, b: tuple) -> int:
    """Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple, b: tuple) -> float:
    """Euclidean distance between two grid positions."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def chebyshev(a: tuple, b: tuple) -> int:
    """Chebyshev distance between two grid positions."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# === Demonstration ===

if __name__ == "__main__":
    start = (0, 0)
    goal = (3, 4)

    print(f"Start: {start}, Goal: {goal}")
    print(f"Manhattan distance:  {manhattan(start, goal)}")
    print(f"Euclidean distance:  {euclidean(start, goal):.2f}")
    print(f"Chebyshev distance:  {chebyshev(start, goal)}")
    print(f"\nManhattan >= Euclidean >= Chebyshev:")
    print(f"  {manhattan(start, goal)} >= {euclidean(start, goal):.2f} >= {chebyshev(start, goal)}")
```

**Output:**
```
Start: (0, 0), Goal: (3, 4)
Manhattan distance:  7
Euclidean distance:  5.00
Chebyshev distance:  4

Manhattan >= Euclidean >= Chebyshev:
  7 >= 5.00 >= 4
```

For this example, Manhattan distance gives the tightest estimate (assuming 4-directional movement), so it would cause A* to expand the fewest vertices in a 4-connected grid.

## Reference

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.), Chapter 3. Pearson.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 24-25. MIT Press.
