# Grid Paths

Grid path problems are a natural introduction to two-dimensional dynamic programming.  Given a grid of size $m \times n$, the task is to count or optimize paths from the top-left corner to the bottom-right corner, moving only right or down.  The two-dimensional table structure makes the recurrence and its dependencies visually intuitive, and the problem generalizes easily to handle obstacles, costs, and other constraints.

## Problem Statement: Counting Paths

Given an $m \times n$ grid, count the number of unique paths from position $(0, 0)$ to position $(m-1, n-1)$, where at each cell you can only move right or down.

**Example:** In a $3 \times 3$ grid, there are 6 unique paths.

## Recurrence

Let $dp[i][j]$ denote the number of unique paths from $(0, 0)$ to $(i, j)$.  Cell $(i, j)$ can be reached from either $(i-1, j)$ (moving down) or $(i, j-1)$ (moving right), so

$$
dp[i][j] = dp[i-1][j] + dp[i][j-1] \quad \text{for } i \ge 1, \; j \ge 1
$$

with base cases

$$
dp[i][0] = 1 \;\text{for all } i, \quad dp[0][j] = 1 \;\text{for all } j
$$

since there is exactly one way to reach any cell in the first row (all rights) or first column (all downs).

## Combinatorial Solution

The path from $(0,0)$ to $(m-1, n-1)$ consists of exactly $m - 1$ down moves and $n - 1$ right moves, in some order.  The number of distinct orderings is

$$
\binom{m + n - 2}{m - 1} = \frac{(m + n - 2)!}{(m - 1)!(n - 1)!}
$$

The DP approach and the combinatorial formula give the same answer, but DP generalizes to grids with obstacles or variable costs.

## Tabulation

```python
"""
Grid paths: count and optimize paths in 2D grids using DP.
"""


# ===================================================================
# Counting unique paths
# ===================================================================
def unique_paths(m: int, n: int) -> int:
    """Count unique paths in an m x n grid. Time: O(mn), Space: O(mn)."""
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]
```

## Grid with Obstacles

When some cells are blocked, set $dp[i][j] = 0$ for obstacle cells:

$$
dp[i][j] = \begin{cases} 0 & \text{if cell } (i,j) \text{ is an obstacle} \\ dp[i-1][j] + dp[i][j-1] & \text{otherwise} \end{cases}
$$

```python
# ===================================================================
# Counting paths with obstacles
# ===================================================================
def unique_paths_with_obstacles(grid: list[list[int]]) -> int:
    """Count paths in a grid where 1 = obstacle. Time: O(mn), Space: O(mn)."""
    m, n = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[m - 1][n - 1] == 1:
        return 0

    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                if i > 0:
                    dp[i][j] += dp[i - 1][j]
                if j > 0:
                    dp[i][j] += dp[i][j - 1]

    return dp[m - 1][n - 1]
```

## Minimum Cost Path

When each cell has an associated cost, find the path from $(0, 0)$ to $(m-1, n-1)$ with minimum total cost:

$$
dp[i][j] = \text{cost}[i][j] + \min\bigl(dp[i-1][j],\; dp[i][j-1]\bigr)
$$

with $dp[0][0] = \text{cost}[0][0]$.

```python
# ===================================================================
# Minimum cost path
# ===================================================================
def min_cost_path(cost: list[list[int]]) -> int:
    """Minimum cost from top-left to bottom-right. Time: O(mn), Space: O(mn)."""
    m, n = len(cost), len(cost[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = cost[0][0]

    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + cost[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + cost[0][j]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = cost[i][j] + min(dp[i - 1][j], dp[i][j - 1])

    return dp[m - 1][n - 1]
```

## Space Optimization

Since each row depends only on the current and previous rows, space reduces from $O(mn)$ to $O(n)$:

```python
# ===================================================================
# Space-optimized unique paths
# ===================================================================
def unique_paths_optimized(m: int, n: int) -> int:
    """Count unique paths with O(n) space."""
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[n - 1]
```

## Complexity

| Variant | Time | Space | Optimized Space |
|---------|------|-------|-----------------|
| Count paths | $O(mn)$ | $O(mn)$ | $O(n)$ |
| With obstacles | $O(mn)$ | $O(mn)$ | $O(n)$ |
| Minimum cost | $O(mn)$ | $O(mn)$ | $O(n)$ |

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    # Unique paths
    for m, n in [(3, 3), (3, 7), (7, 3)]:
        print(f"unique_paths({m},{n}) = {unique_paths(m, n)}")

    # With obstacles
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    print(f"paths with obstacle = {unique_paths_with_obstacles(grid)}")

    # Minimum cost
    cost = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
    print(f"min cost path = {min_cost_path(cost)}")
```

**Output:**
```
unique_paths(3,3) = 6
unique_paths(3,7) = 28
unique_paths(7,3) = 28
paths with obstacle = 2
min cost path = 7
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
