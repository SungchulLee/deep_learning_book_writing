# Palindrome Partitioning

A palindrome reads the same forwards and backwards. The **palindrome partitioning** problem asks: given a string, what is the minimum number of cuts needed to split it into substrings that are each palindromes? This problem combines two DP subproblems — precomputing which substrings are palindromes and then finding the optimal cut positions — making it a rich example of two-dimensional dynamic programming.

## Problem Statement

Given a string $s$ of length $n$, find the minimum number of cuts to partition $s$ into substrings $s_1, s_2, \ldots, s_k$ such that each $s_i$ is a palindrome.

**Example.** For $s = \text{``aab''}$, one cut after position 1 gives $\{\text{``aa''}, \text{``b''}\}$, both palindromes. The minimum number of cuts is 1.

## Step 1 — Palindrome Table

First, precompute a boolean table $P[i][j]$ that records whether the substring $s[i \ldots j]$ is a palindrome. A substring $s[i \ldots j]$ is a palindrome if and only if:

$$
P[i][j] = \begin{cases} \text{true} & \text{if } i = j \\ s[i] = s[j] & \text{if } j = i + 1 \\ s[i] = s[j] \text{ and } P[i+1][j-1] & \text{if } j > i + 1 \end{cases}
$$

Fill this table by increasing substring length so that $P[i+1][j-1]$ is always available when needed.

## Step 2 — Minimum Cuts

Let $\text{cuts}[j]$ be the minimum number of cuts to partition $s[0 \ldots j]$ into palindromes:

$$
\text{cuts}[j] = \begin{cases} 0 & \text{if } P[0][j] = \text{true} \\ \displaystyle \min_{0 \le i \le j,\, P[i][j]} \bigl\{ \text{cuts}[i-1] + 1 \bigr\} & \text{otherwise} \end{cases}
$$

The answer is $\text{cuts}[n-1]$.

!!! tip "Intuition"
    For each position $j$, check every possible last palindrome $s[i \ldots j]$. If it is a palindrome, the cost is one cut plus the optimal solution for $s[0 \ldots i-1]$.

## Complexity

| Aspect | Value |
|---|---|
| Time | $O(n^2)$ |
| Space | $O(n^2)$ for the palindrome table, $O(n)$ for the cuts array |
| Subproblems | $O(n^2)$ palindrome checks + $O(n)$ cut computations |

## Worked Example

For $s = \text{``abac''}$ (0-indexed):

**Palindrome table $P$:**

| $P[i][j]$ | a | b | a | c |
|---|---|---|---|---|
| a | T | F | T | F |
| b | | T | F | F |
| a | | | T | F |
| c | | | | T |

**Cuts array:**

- $\text{cuts}[0] = 0$ (``a'' is a palindrome)
- $\text{cuts}[1] = 1$ (``ab'' is not a palindrome; best: ``a'' | ``b'')
- $\text{cuts}[2] = 0$ (``aba'' is a palindrome)
- $\text{cuts}[3] = 1$ (``abac'' is not; best: ``aba'' | ``c'')

Minimum cuts: 1.

## Python Implementation

```python
"""
Palindrome Partitioning — Minimum Cuts via DP.

Precomputes a palindrome table, then finds the minimum number of cuts
to partition a string into palindromic substrings.
"""


# === Palindrome Partitioning ===

def min_palindrome_cuts(s: str) -> int:
    """Return the minimum number of cuts for palindrome partitioning.

    Time: O(n^2), Space: O(n^2).
    """
    n = len(s)
    if n <= 1:
        return 0

    # Step 1: Build palindrome table
    is_pal = [[False] * n for _ in range(n)]

    for i in range(n):
        is_pal[i][i] = True

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]

    # Step 2: Compute minimum cuts
    cuts = [0] * n
    for j in range(n):
        if is_pal[0][j]:
            cuts[j] = 0
        else:
            cuts[j] = j  # worst case: cut every character
            for i in range(1, j + 1):
                if is_pal[i][j]:
                    cuts[j] = min(cuts[j], cuts[i - 1] + 1)

    return cuts[n - 1]


# === Partition Reconstruction ===

def palindrome_partition(s: str) -> list[str]:
    """Return one optimal palindrome partition."""
    n = len(s)
    if n <= 1:
        return [s] if s else []

    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = (s[i] == s[j])
            else:
                is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]

    cuts = [0] * n
    split_at = [-1] * n
    for j in range(n):
        if is_pal[0][j]:
            cuts[j] = 0
            split_at[j] = 0
        else:
            cuts[j] = j
            split_at[j] = j
            for i in range(1, j + 1):
                if is_pal[i][j] and cuts[i - 1] + 1 < cuts[j]:
                    cuts[j] = cuts[i - 1] + 1
                    split_at[j] = i

    # Trace back to build partition
    parts = []
    j = n - 1
    while j >= 0:
        i = split_at[j]
        parts.append(s[i:j + 1])
        j = i - 1

    return list(reversed(parts))


# === Main ===

if __name__ == "__main__":
    test_cases = ["aab", "abac", "abcba", "abcdef"]
    for s in test_cases:
        num_cuts = min_palindrome_cuts(s)
        partition = palindrome_partition(s)
        print(f"'{s}' -> {num_cuts} cuts: {partition}")
    # Output:
    # 'aab' -> 1 cuts: ['aa', 'b']
    # 'abac' -> 1 cuts: ['aba', 'c']
    # 'abcba' -> 0 cuts: ['abcba']
    # 'abcdef' -> 5 cuts: ['a', 'b', 'c', 'd', 'e', 'f']
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
