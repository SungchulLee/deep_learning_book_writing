# LCS via Memoization

The naive recursive solution to the Longest Common Subsequence problem has exponential time complexity because it recomputes the same subproblems repeatedly. Memoization adds a cache to the recursive approach: each subproblem is solved once, and subsequent calls retrieve the stored result. This top-down strategy preserves the natural recursive structure while achieving the same $O(mn)$ time complexity as bottom-up tabulation.

## From Recursion to Memoization

The plain recursive LCS computes $c(i, j)$ — the LCS length of prefixes $X_i$ and $Y_j$ — using:

$$
c(i, j) = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \\ c(i-1, j-1) + 1 & \text{if } x_i = y_j \\ \max(c(i-1, j),\, c(i, j-1)) & \text{if } x_i \ne y_j \end{cases}
$$

Without memoization, the recursion tree has up to $2^{m+n}$ nodes because each call branches into two subproblems. However, there are only $(m+1)(n+1)$ distinct subproblems $(i, j)$. Memoization ensures each is computed exactly once.

## How Memoization Works

1. Initialize a 2D table `memo` of size $(m+1) \times (n+1)$ with sentinel values (e.g., $-1$).
2. Before computing $c(i, j)$, check if `memo[i][j]` already stores a result.
3. If cached, return the stored value immediately.
4. If not, compute the result recursively, store it in `memo[i][j]`, then return it.

!!! tip "Memoization vs Tabulation"
    Memoization computes only the subproblems reachable from the original call, which can be advantageous when many table entries are never needed. Tabulation fills the entire table regardless, but avoids recursion overhead and stack depth limits.

## Complexity

| Aspect | Value |
|---|---|
| Time | $O(mn)$ — each of $(m+1)(n+1)$ subproblems computed once |
| Space | $O(mn)$ for the memo table, plus $O(m+n)$ recursion stack |
| Stack depth | $O(m + n)$ in the worst case |

!!! warning "Recursion Depth Limit"
    For long sequences ($m + n > 1000$), the recursion depth may exceed Python's default limit. Use `sys.setrecursionlimit()` or switch to the bottom-up tabulation approach.

## Python Implementation

```python
"""
LCS — Memoized Recursive Solution.

Demonstrates top-down dynamic programming for the Longest Common
Subsequence problem, contrasting with the naive recursive approach.
"""


# === Naive Recursive LCS (Exponential) ===

def lcs_recursive(x: str, y: str, i: int, j: int) -> int:
    """Naive recursive LCS. Time: O(2^(m+n)), Space: O(m+n) stack."""
    if i == 0 or j == 0:
        return 0
    if x[i - 1] == y[j - 1]:
        return lcs_recursive(x, y, i - 1, j - 1) + 1
    return max(lcs_recursive(x, y, i - 1, j), lcs_recursive(x, y, i, j - 1))


# === Memoized LCS ===

def lcs_memo(x: str, y: str) -> int:
    """LCS length via memoization. Time: O(mn), Space: O(mn)."""
    m, n = len(x), len(y)
    memo = [[-1] * (n + 1) for _ in range(m + 1)]

    def helper(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0
        if memo[i][j] != -1:
            return memo[i][j]
        if x[i - 1] == y[j - 1]:
            memo[i][j] = helper(i - 1, j - 1) + 1
        else:
            memo[i][j] = max(helper(i - 1, j), helper(i, j - 1))
        return memo[i][j]

    return helper(m, n)


# === Memoized LCS with functools ===

def lcs_memo_functools(x: str, y: str) -> int:
    """LCS using Python's built-in lru_cache for memoization."""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def helper(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0
        if x[i - 1] == y[j - 1]:
            return helper(i - 1, j - 1) + 1
        return max(helper(i - 1, j), helper(i, j - 1))

    return helper(len(x), len(y))


# === Main ===

if __name__ == "__main__":
    x = "ABCBDAB"
    y = "BDCABA"

    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"LCS (naive):     {lcs_recursive(x, y, len(x), len(y))}")
    print(f"LCS (memo):      {lcs_memo(x, y)}")
    print(f"LCS (lru_cache): {lcs_memo_functools(x, y)}")
    # Output:
    # X = ABCBDAB
    # Y = BDCABA
    # LCS (naive):     4
    # LCS (memo):      4
    # LCS (lru_cache): 4
```

## Subproblem Reuse Analysis

For $X = \text{AB}$ and $Y = \text{BA}$, the naive recursion tree shows why memoization helps:

```
c(2,2)
├── c(1,2)             [x2 ≠ y2]
│   ├── c(0,2) = 0
│   └── c(1,1)         [x1 = y1 = 'A'? No: A ≠ B]
│       ├── c(0,1) = 0
│       └── c(1,0) = 0
└── c(2,1)             [x2 ≠ y1]
    ├── c(1,1)         ← RECOMPUTED without memo
    │   ├── c(0,1) = 0
    │   └── c(1,0) = 0
    └── c(2,0) = 0
```

With memoization, $c(1,1)$ is computed once and retrieved on the second call. For larger inputs, the savings grow from constant to exponential.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
