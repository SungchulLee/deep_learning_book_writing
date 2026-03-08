# Naive Algorithm

The naive (brute-force) string matching algorithm is the simplest approach to finding all occurrences of a pattern $P[0..m-1]$ in a text $T[0..n-1]$. It slides the pattern across the text one position at a time and checks for a match at each position.

## Algorithm

For every possible alignment $i = 0, 1, \ldots, n - m$, compare $P[0..m-1]$ with $T[i..i+m-1]$ character by character. If all $m$ characters match, report an occurrence at position $i$. If a mismatch occurs at any position, move to the next alignment $i+1$.

$$

\text{For each shift } s \in \{0, 1, \ldots, n-m\}: \quad \text{check if } T[s+j] = P[j] \;\; \forall \, j \in \{0, \ldots, m-1\}

$$

```python
def naive_search(text: str, pattern: str) -> list[int]:
    """Return all starting indices where pattern occurs in text."""
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    occurrences = []
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            occurrences.append(i)
    return occurrences

# Example
text = "AABAACAADAABAABA"
pattern = "AABA"
print(naive_search(text, pattern))
# Output: [0, 9, 12]
```

## Complexity Analysis

- **Worst case:** $O((n - m + 1) \cdot m) = O(nm)$. This occurs when many partial matches exist, e.g., $T = \texttt{AAAA\ldots A}$ and $P = \texttt{AAA\ldots AB}$.
- **Best case:** $O(n)$. When the first character of $P$ never appears in $T$, each alignment fails immediately.
- **Average case:** $O(n)$ for random text over a large alphabet, since most alignments fail quickly.
- **Space:** $O(1)$ auxiliary space (excluding the output list).

The naive algorithm serves as a baseline. Its simplicity makes it practical for short patterns or small texts, but for large-scale matching, algorithms like KMP or Boyer-Moore are preferred.

# Reference

[Introduction to Algorithms (CLRS), Section 32.1 - The naive string-matching algorithm](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)

[Naive Pattern Searching Algorithm](https://www.geeksforgeeks.org/naive-algorithm-for-pattern-searching/)
