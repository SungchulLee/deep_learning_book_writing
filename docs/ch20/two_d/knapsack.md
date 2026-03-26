# 0/1 Knapsack

The 0/1 knapsack problem is one of the most important problems in combinatorial optimization. Given a set of items, each with a weight and a value, the task is to select a subset that maximizes total value without exceeding a weight capacity. The "0/1" refers to the constraint that each item is either taken entirely or left behind — no fractional selections are allowed. This problem illustrates two-dimensional dynamic programming with one dimension for items and another for weight capacity.

## Problem Statement

Given $n$ items where item $i$ has weight $w_i$ and value $v_i$, and a knapsack with weight capacity $W$, find

$$
\max \sum_{i=1}^{n} v_i x_i \quad \text{subject to} \quad \sum_{i=1}^{n} w_i x_i \le W, \quad x_i \in \{0, 1\}
$$

## Recurrence

Let $dp[i][w]$ denote the maximum value achievable using items $1$ through $i$ with weight capacity $w$. For each item $i$, there are two choices:

1. **Exclude** item $i$: the value is $dp[i-1][w]$.
2. **Include** item $i$ (only if $w_i \le w$): the value is $dp[i-1][w - w_i] + v_i$.

Taking the better option:

$$
dp[i][w] = \begin{cases} dp[i-1][w] & \text{if } w_i > w \\ \max\bigl(dp[i-1][w],\; dp[i-1][w - w_i] + v_i\bigr) & \text{if } w_i \le w \end{cases}
$$

with base cases $dp[0][w] = 0$ for all $w$ (no items means zero value).

## Worked Example

Consider items with weights $[2, 3, 4, 5]$, values $[3, 4, 5, 6]$, and capacity $W = 8$.

The DP table fills row by row. Each cell $dp[i][w]$ represents the best value using items $1 \ldots i$ with capacity $w$:

| $dp[i][w]$ | $w=0$ | $w=1$ | $w=2$ | $w=3$ | $w=4$ | $w=5$ | $w=6$ | $w=7$ | $w=8$ |
|---|---|---|---|---|---|---|---|---|---|
| $i=0$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| $i=1$ ($w_1\!=\!2, v_1\!=\!3$) | 0 | 0 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| $i=2$ ($w_2\!=\!3, v_2\!=\!4$) | 0 | 0 | 3 | 4 | 4 | 7 | 7 | 7 | 7 |
| $i=3$ ($w_3\!=\!4, v_3\!=\!5$) | 0 | 0 | 3 | 4 | 5 | 7 | 8 | 9 | 9 |
| $i=4$ ($w_4\!=\!5, v_4\!=\!6$) | 0 | 0 | 3 | 4 | 5 | 7 | 8 | 9 | **10** |

The optimal value is $dp[4][8] = 10$. Tracing back: $dp[4][8] \ne dp[3][8]$, so item 4 is selected ($w = 8 - 5 = 3$). Then $dp[2][3] \ne dp[1][3]$, so item 2 is selected. The optimal subset is items $\{2, 4\}$ with total weight $3 + 5 = 8$ and total value $4 + 6 = 10$.

## Tabulation

The following implementation builds the full 2D table, optimizes space with a 1D array, and provides solution reconstruction.

```python
"""
0/1 Knapsack — Dynamic Programming.

Three approaches: 2D tabulation, 1D space-optimized, and reconstruction.
"""


# === 2D Tabulation ===

def knapsack_2d(weights: list[int], values: list[int], capacity: int) -> int:
    """0/1 knapsack with 2D table. Time: O(nW), Space: O(nW)."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    return dp[n][capacity]


# === 1D Space Optimization ===

def knapsack_1d(weights: list[int], values: list[int], capacity: int) -> int:
    """0/1 knapsack with 1D array. Time: O(nW), Space: O(W)."""
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


# === Reconstruction ===

def knapsack_with_items(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """Return maximum value and indices of selected items."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # Trace back through table to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]

    return dp[n][capacity], list(reversed(selected))


# === Main ===

if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    max_val = knapsack_2d(weights, values, capacity)
    max_val_1d = knapsack_1d(weights, values, capacity)
    max_val_items, items = knapsack_with_items(weights, values, capacity)

    print(f"Weights: {weights}")
    print(f"Values:  {values}")
    print(f"Capacity: {capacity}")
    print(f"Max value (2D): {max_val}")
    print(f"Max value (1D): {max_val_1d}")
    print(f"Max value: {max_val_items}, items: {items}")
    # Output:
    # Weights: [2, 3, 4, 5]
    # Values:  [3, 4, 5, 6]
    # Capacity: 8
    # Max value (2D): 10
    # Max value (1D): 10
    # Max value: 10, items: [1, 3]
```

## Space Optimization

Since row $i$ depends only on row $i-1$, a single 1D array suffices. The key insight is to process weights in **reverse order** (from $W$ down to $w_i$). This prevents using an item twice: when computing $dp[w]$ for item $i$, the value $dp[w - w_i]$ still reflects the state *without* item $i$.

If weights were processed in forward order, $dp[w - w_i]$ might already include item $i$, effectively allowing unlimited copies (which solves the unbounded knapsack variant instead).

## Complexity

| Aspect | Value |
|---|---|
| Time | $O(nW)$ — pseudo-polynomial |
| Space (2D) | $O(nW)$ |
| Space (1D) | $O(W)$ |
| Subproblems | $(n+1)(W+1)$ |

!!! warning "Pseudo-polynomial Complexity"
    Although $O(nW)$ looks polynomial, the input size for $W$ is $\log W$ bits. The algorithm is exponential in the input size of $W$, which is why knapsack remains NP-hard. For large $W$, approximation algorithms or branch-and-bound may be more practical.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
