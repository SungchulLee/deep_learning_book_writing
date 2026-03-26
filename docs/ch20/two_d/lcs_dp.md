# LCS via Tabulation

The recursive LCS solution recomputes many overlapping subproblems. Tabulation eliminates this redundancy by filling a 2D table bottom-up, computing each subproblem exactly once. This page focuses on the bottom-up DP approach and its space-optimized variants, complementing the recursive and memoized approaches covered on sibling pages.

## Bottom-Up Strategy

Instead of recursion, allocate a table $c$ of size $(m+1) \times (n+1)$ and fill it row by row. Each cell $c[i][j]$ stores the LCS length of the prefixes $X_i = x_1 \cdots x_i$ and $Y_j = y_1 \cdots y_j$:

$$
c[i][j] = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \\ c[i-1][j-1] + 1 & \text{if } x_i = y_j \\ \max(c[i-1][j],\, c[i][j-1]) & \text{if } x_i \ne y_j \end{cases}
$$

The answer is $c[m][n]$.

## Fill Order and Dependencies

Each cell $c[i][j]$ depends on three neighbors:

- $c[i-1][j-1]$ (diagonal, above-left)
- $c[i-1][j]$ (directly above)
- $c[i][j-1]$ (directly left)

Processing rows from top to bottom and columns from left to right ensures all dependencies are satisfied before each cell is computed.

## Space Optimization

Since row $i$ depends only on row $i-1$, two rows suffice. We can alternate between a "previous" and "current" row, reducing space from $O(mn)$ to $O(\min(m, n))$.

!!! tip "Single-Row Trick"
    With careful bookkeeping, a single 1D array plus one extra variable (for the diagonal element) suffices. Process the shorter string as the column dimension to minimize array size.

## Printing the LCS

To reconstruct the actual subsequence (not just its length), maintain a direction table $b[i][j]$ during the fill phase:

- $b[i][j] = \text{DIAG}$ when $x_i = y_j$ (character is part of the LCS)
- $b[i][j] = \text{UP}$ when $c[i-1][j] \ge c[i][j-1]$
- $b[i][j] = \text{LEFT}$ otherwise

Trace from $b[m][n]$ back to a boundary to collect the LCS characters in reverse.

## Python Implementation

```python
"""
LCS — Bottom-Up Tabulation with Space Optimization.

Demonstrates the standard 2D tabulation, space-optimized 1D variant,
and subsequence reconstruction via a direction table.
"""


# === 2D Tabulation ===

def lcs_tabulation(x: str, y: str) -> int:
    """LCS length via bottom-up DP. Time: O(mn), Space: O(mn)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# === Space-Optimized (Two Rows) ===

def lcs_space_optimized(x: str, y: str) -> int:
    """LCS length using only two rows. Time: O(mn), Space: O(min(m,n))."""
    if len(x) < len(y):
        x, y = y, x
    m, n = len(x), len(y)

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


# === Reconstruction via Direction Table ===

def lcs_with_reconstruction(x: str, y: str) -> tuple[int, str]:
    """Return LCS length and the subsequence string."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    direction = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                direction[i][j] = "DIAG"
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                direction[i][j] = "UP"
            else:
                dp[i][j] = dp[i][j - 1]
                direction[i][j] = "LEFT"

    # Trace back to recover the LCS
    i, j = m, n
    result = []
    while i > 0 and j > 0:
        if direction[i][j] == "DIAG":
            result.append(x[i - 1])
            i -= 1
            j -= 1
        elif direction[i][j] == "UP":
            i -= 1
        else:
            j -= 1

    return dp[m][n], "".join(reversed(result))


# === Main ===

if __name__ == "__main__":
    x = "AGGTAB"
    y = "GXTXAYB"

    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"LCS length (2D):    {lcs_tabulation(x, y)}")
    print(f"LCS length (space): {lcs_space_optimized(x, y)}")

    length, subseq = lcs_with_reconstruction(x, y)
    print(f"LCS: '{subseq}' (length {length})")
    # Output:
    # X = AGGTAB
    # Y = GXTXAYB
    # LCS length (2D):    4
    # LCS length (space): 4
    # LCS: 'GTAB' (length 4)
```

## Worked Example

For $X = \text{AGGTAB}$ and $Y = \text{GXTXAYB}$:

| | $\varepsilon$ | G | X | T | X | A | Y | B |
|---|---|---|---|---|---|---|---|---|
| $\varepsilon$ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 0 | **1** | 1 | 1 |
| G | 0 | **1** | 1 | 1 | 1 | 1 | 1 | 1 |
| G | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| T | 0 | 1 | 1 | **2** | 2 | 2 | 2 | 2 |
| A | 0 | 1 | 1 | 2 | 2 | **3** | 3 | 3 |
| B | 0 | 1 | 1 | 2 | 2 | 3 | 3 | **4** |

The LCS has length 4. Tracing the diagonal entries yields $\text{GTAB}$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
