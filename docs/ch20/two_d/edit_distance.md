# Edit Distance

How similar are two strings? The **edit distance** (Levenshtein distance) counts the minimum number of single-character operations needed to transform one string into another. This metric is fundamental in spell checking, DNA sequence alignment, diff utilities, and natural language processing. The problem has a clean dynamic programming solution with optimal substructure and overlapping subproblems, making it a textbook example of 2D dynamic programming.

## Problem Statement

Given two strings $s_1$ of length $m$ and $s_2$ of length $n$, find the minimum number of operations to transform $s_1$ into $s_2$. The allowed operations, each costing 1, are:

- **Insert** a character into $s_1$.
- **Delete** a character from $s_1$.
- **Replace** a character in $s_1$ with a different character.

For example, transforming "kitten" into "sitting" requires 3 operations: replace 'k' with 's', replace 'e' with 'i', and insert 'g' at the end.

## Optimal Substructure

Let $d(i, j)$ denote the edit distance between the first $i$ characters of $s_1$ and the first $j$ characters of $s_2$. Consider the last characters $s_1[i]$ and $s_2[j]$:

- If $s_1[i] = s_2[j]$, the characters match and no operation is needed: $d(i, j) = d(i-1, j-1)$.
- If $s_1[i] \neq s_2[j]$, we take the minimum over three choices:
    - **Replace** $s_1[i]$ with $s_2[j]$: costs $1 + d(i-1, j-1)$.
    - **Delete** $s_1[i]$: costs $1 + d(i-1, j)$.
    - **Insert** $s_2[j]$ after $s_1[i]$: costs $1 + d(i, j-1)$.

## Recurrence

$$
d(i, j) = \begin{cases} j & \text{if } i = 0 \\ i & \text{if } j = 0 \\ d(i-1, j-1) & \text{if } s_1[i] = s_2[j] \\ 1 + \min\bigl(d(i-1, j),\; d(i, j-1),\; d(i-1, j-1)\bigr) & \text{otherwise} \end{cases}
$$

**Base cases.** Transforming a string of length $i$ into an empty string requires $i$ deletions, so $d(i, 0) = i$. Transforming an empty string into a string of length $j$ requires $j$ insertions, so $d(0, j) = j$.

## Tabulation

The DP table is filled row by row, left to right. Each cell $d(i, j)$ depends only on $d(i-1, j-1)$, $d(i-1, j)$, and $d(i, j-1)$, which are all computed before $d(i, j)$.

For "kitten" and "sitting", the table is:

|       |   | s | i | t | t | i | n | g |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| **k** | 1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| **i** | 2 | 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| **t** | 3 | 3 | 2 | 1 | 2 | 3 | 4 | 5 |
| **t** | 4 | 4 | 3 | 2 | 1 | 2 | 3 | 4 |
| **e** | 5 | 5 | 4 | 3 | 2 | 2 | 3 | 4 |
| **n** | 6 | 6 | 5 | 4 | 3 | 3 | 2 | 3 |

The answer $d(6, 7) = 3$ is in the bottom-right corner.

## Implementation

```python
"""
Edit distance (Levenshtein distance) via dynamic programming.

Computes the minimum number of insertions, deletions, and replacements
to transform one string into another. Includes operation backtrace
and space-optimized variant.
"""

# === Standard DP Solution ===

def edit_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings.

    Args:
        s1: Source string.
        s2: Target string.

    Returns:
        Minimum number of edit operations.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1]   # replace
                )

    return dp[m][n]


# === Backtrace to Recover Operations ===

def edit_operations(s1: str, s2: str) -> list[str]:
    """Recover the sequence of edit operations.

    Args:
        s1: Source string.
        s2: Target string.

    Returns:
        List of operation descriptions.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    # Backtrace
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"Replace '{s1[i-1]}' with '{s2[j-1]}' at position {i}")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(f"Delete '{s1[i-1]}' at position {i}")
            i -= 1
        else:
            ops.append(f"Insert '{s2[j-1]}' at position {i + 1}")
            j -= 1

    ops.reverse()
    return ops


# === Space-Optimized Version ===

def edit_distance_optimized(s1: str, s2: str) -> int:
    """Space-optimized edit distance using two rows.

    Args:
        s1: Source string.
        s2: Target string.

    Returns:
        Minimum number of edit operations.
    """
    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


# === Demonstration ===

if __name__ == "__main__":
    s1, s2 = "kitten", "sitting"

    dist = edit_distance(s1, s2)
    print(f"edit_distance('{s1}', '{s2}') = {dist}")

    ops = edit_operations(s1, s2)
    print(f"\nOperations ({len(ops)} total):")
    for op in ops:
        print(f"  {op}")

    dist_opt = edit_distance_optimized(s1, s2)
    print(f"\nSpace-optimized result: {dist_opt}")

    # Additional example
    a, b = "intention", "execution"
    print(f"\nedit_distance('{a}', '{b}') = {edit_distance(a, b)}")
```

**Output:**

```
edit_distance('kitten', 'sitting') = 3

Operations (3 total):
  Replace 'k' with 's' at position 1
  Replace 'e' with 'i' at position 5
  Insert 'g' at position 7

Space-optimized result: 3

edit_distance('intention', 'execution') = 5
```

## Complexity

| Aspect | Standard | Space-optimized |
|--------|:--------:|:---------------:|
| Time   | $O(mn)$  | $O(mn)$         |
| Space  | $O(mn)$  | $O(\min(m, n))$ |

The standard solution fills an $(m+1) \times (n+1)$ table, giving $O(mn)$ time and space. Since each row only depends on the previous row, the space-optimized version uses just two rows, reducing space to $O(\min(m, n))$. The trade-off is that backtrace (recovering the actual operations) requires the full table.

## Variants

- **Weighted edit distance.** Each operation has a different cost. The recurrence replaces the constant 1 with operation-specific costs $c_{\text{ins}}, c_{\text{del}}, c_{\text{rep}}$.
- **Longest common subsequence.** Setting the replacement cost to infinity (or 2, to model delete + insert) makes the edit distance equivalent to $m + n - 2 \cdot \text{LCS}(s_1, s_2)$.
- **Damerau-Levenshtein distance.** Adds a fourth operation: transposition of two adjacent characters.

## Applications

- **Spell checking.** Suggest corrections by finding dictionary words with small edit distance.
- **DNA sequence alignment.** Measure similarity between genetic sequences.
- **Diff utilities.** Compute minimal change sets between file versions.
- **Natural language processing.** Fuzzy string matching in search and information retrieval.

## Reference

- Wagner, R. A., & Fischer, M. J. (1974). The string-to-string correction problem. *Journal of the ACM*, 21(1), 168--173.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 14: Dynamic Programming.
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707--710.
