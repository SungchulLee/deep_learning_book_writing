# Z-Algorithm

The Z-algorithm computes, for a string $S$ of length $n$, the **Z-array** $Z[0..n-1]$ where $Z[i]$ is the length of the longest substring starting at position $i$ that matches a prefix of $S$. By convention, $Z[0]$ is defined as 0 (or $n$). This array can be used for exact pattern matching in $O(n+m)$ time.

## Definition

$$

Z[i] = \max\{k \ge 0 : S[0..k-1] = S[i..i+k-1]\}

$$

For the string $S = \texttt{aabxaab}$, the Z-array is:

| $i$    | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|--------|---|---|---|---|---|---|---|
| $S[i]$ | a | a | b | x | a | a | b |
| $Z[i]$ | 0 | 1 | 0 | 0 | 3 | 1 | 0 |

## Algorithm

The Z-algorithm uses a window $[L, R]$ representing the interval with the rightmost endpoint $R$ such that $S[L..R]$ matches a prefix of $S$. We process positions left to right, and for each position $i$, we either use previously computed values or extend the window.

```python
def z_function(s: str) -> list[int]:
    """Compute the Z-array for string s."""
    n = len(s)
    if n == 0:
        return []
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z

# Example
print(z_function("aabxaab"))
# Output: [0, 1, 0, 0, 3, 1, 0]
```

## Application to Pattern Matching

To find all occurrences of pattern $P$ in text $T$, construct the string $S = P\$T$ where $\$$ is a character not in $P$ or $T$. Compute the Z-array of $S$. Position $i$ in $S$ corresponds to a match if $Z[i] = m$ (the length of $P$).

```python
def z_search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using the Z-algorithm."""
    m = len(pattern)
    if m == 0:
        return []
    concat = pattern + "$" + text
    z = z_function(concat)
    occurrences = []
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            occurrences.append(i - m - 1)
    return occurrences

# Example
text = "AABCAABXAAAZ"
pattern = "AAB"
print(z_search(text, pattern))
# Output: [0, 4]
```

## Complexity Analysis

- **Time:** $O(n)$. Each character is visited at most twice (once when it is inside the Z-box and once when extending the Z-box). The amortized cost per position is $O(1)$.
- **Space:** $O(n)$ for the Z-array.
- **For pattern matching:** $O(n + m)$ time and space, where the concatenated string has length $n + m + 1$.

## Relationship to KMP

The Z-algorithm and the KMP failure function are closely related. In fact, one can compute the failure function from the Z-array and vice versa in $O(n)$ time. However, the Z-algorithm is often considered simpler to implement and understand.

# Reference

[Z-algorithm - CP-Algorithms](https://cp-algorithms.com/string/z-function.html)

[Z Algorithm (Linear time pattern searching) - GeeksforGeeks](https://www.geeksforgeeks.org/z-algorithm-linear-time-pattern-searching-algorithm/)
