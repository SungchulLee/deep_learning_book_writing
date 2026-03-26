# Longest Common Subsequence

A **subsequence** of a string is obtained by deleting zero or more characters without changing the order of the remaining characters. The Longest Common Subsequence (LCS) problem asks: given two sequences $X$ and $Y$, what is the longest sequence that appears as a subsequence of both? LCS is a fundamental problem in dynamic programming, with applications in diff utilities, bioinformatics (DNA sequence alignment), and version control systems.

## Problem Definition

Given two sequences $X = x_1 x_2 \cdots x_m$ and $Y = y_1 y_2 \cdots y_n$, find the length of the longest sequence $Z$ that is a subsequence of both $X$ and $Y$.

**Example.** For $X = \text{ABCBDAB}$ and $Y = \text{BDCABA}$, one LCS is $\text{BCBA}$ with length 4.

## Optimal Substructure

The LCS problem exhibits optimal substructure. Let $X_i = x_1 \cdots x_i$ and $Y_j = y_1 \cdots y_j$ denote prefixes. Then:

- If $x_i = y_j$, this common character must be part of an LCS, and the remaining LCS comes from $X_{i-1}$ and $Y_{j-1}$.
- If $x_i \ne y_j$, at least one of $x_i$ or $y_j$ is not in the LCS, so we take the longer of $\text{LCS}(X_i, Y_{j-1})$ and $\text{LCS}(X_{i-1}, Y_j)$.

## Recurrence

Define $c[i][j]$ as the length of the LCS of $X_i$ and $Y_j$:

$$
c[i][j] = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \\ c[i-1][j-1] + 1 & \text{if } i,j > 0 \text{ and } x_i = y_j \\ \max(c[i][j-1],\, c[i-1][j]) & \text{if } i,j > 0 \text{ and } x_i \ne y_j \end{cases}
$$

The base case states that any sequence has an LCS of length 0 with the empty sequence.

## Worked Example

For $X = \text{ABCB}$ and $Y = \text{BDCB}$, the DP table is:

|  | $\varepsilon$ | B | D | C | B |
|---|---|---|---|---|---|
| $\varepsilon$ | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 0 |
| B | 0 | **1** | 1 | 1 | 1 |
| C | 0 | 1 | 1 | **2** | 2 |
| B | 0 | 1 | 1 | 2 | **3** |

Reading the diagonal matches: B at $(2,1)$, C at $(3,3)$, B at $(4,4)$ gives LCS = $\text{BCB}$ with length 3.

## Complexity

| Aspect | Value |
|---|---|
| Time | $O(mn)$ |
| Space (2D) | $O(mn)$ |
| Space (1D) | $O(\min(m,n))$ |
| Subproblems | $(m+1)(n+1)$ |

## Python Implementation

```python
"""
Longest Common Subsequence — Dynamic Programming.

Computes the LCS length and recovers the actual subsequence
using backtracking through the DP table.
"""


# === LCS Length (2D Tabulation) ===

def lcs_length(x: str, y: str) -> int:
    """Return the length of the LCS of x and y. Time: O(mn), Space: O(mn)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# === LCS with Reconstruction ===

def lcs_with_string(x: str, y: str) -> tuple[int, str]:
    """Return the LCS length and one LCS string."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Trace back to recover the subsequence
    i, j = m, n
    result = []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            result.append(x[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], "".join(reversed(result))


# === Main ===

if __name__ == "__main__":
    x = "ABCBDAB"
    y = "BDCABA"

    length = lcs_length(x, y)
    length2, subseq = lcs_with_string(x, y)

    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"LCS length: {length}")
    print(f"LCS string: {subseq} (length {length2})")
    # Output:
    # X = ABCBDAB
    # Y = BDCABA
    # LCS length: 4
    # LCS string: BCBA (length 4)
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
