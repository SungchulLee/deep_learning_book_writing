# Boyer-Moore

The Boyer-Moore algorithm is one of the most efficient practical algorithms for exact string matching. It scans the pattern from right to left against the text, and uses two heuristics -- the bad-character rule and the good-suffix rule -- to skip large portions of the text after a mismatch.

## Key Ideas

Unlike KMP, which scans the pattern left to right, Boyer-Moore compares the pattern from right to left at each alignment. When a mismatch occurs at position $j$ in the pattern (with text position $i + j$), the algorithm computes two possible shifts:

1. **Bad-character shift:** Based on the mismatching text character.
2. **Good-suffix shift:** Based on the matched suffix of the pattern.

The algorithm takes the maximum of the two shifts, guaranteeing progress.

$$
\text{shift} = \max(\text{bad\_character\_shift}, \text{good\_suffix\_shift})
$$

## Full Algorithm

```python
def boyer_moore(text: str, pattern: str) -> list[int]:
    """Boyer-Moore string matching with both heuristics."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    # Bad character table
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    # Good suffix table
    suffix = [0] * m
    suffix[m - 1] = m
    g = m - 1
    f = 0
    for i in range(m - 2, -1, -1):
        if i > g and suffix[i + m - 1 - f] < i - g:
            suffix[i] = suffix[i + m - 1 - f]
        else:
            g = min(g, i)
            f = i
            while g >= 0 and pattern[g] == pattern[g + m - 1 - f]:
                g -= 1
            suffix[i] = f - g

    good_suffix = [m] * m
    j = 0
    for i in range(m - 1, -1, -1):
        if suffix[i] == i + 1:
            while j < m - 1 - i:
                if good_suffix[j] == m:
                    good_suffix[j] = m - 1 - i
                j += 1
    for i in range(m - 2):
        good_suffix[m - 1 - suffix[i]] = m - 1 - i

    # Search
    occurrences = []
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            occurrences.append(i)
            i += good_suffix[0]
        else:
            bc_shift = j - bad_char.get(text[i + j], -1)
            gs_shift = good_suffix[j]
            i += max(bc_shift, gs_shift)
    return occurrences


# Example
text = "TRUSTHARDTOOTHBRUSHES"
pattern = "TOOTH"
print(boyer_moore(text, pattern))
# Output: [9]
```

## Complexity Analysis

- **Preprocessing:** $O(m + |\Sigma|)$ where $|\Sigma|$ is the alphabet size.
- **Best case:** $O(n/m)$. When the last character of the pattern does not appear in the text, the algorithm can skip $m$ positions at each step -- this is sublinear.
- **Worst case:** $O(nm)$ with the basic algorithm. With the Galil rule optimization, the worst case becomes $O(n + m)$.
- **Average case:** $O(n/m)$ for large alphabets, making it one of the fastest practical algorithms.

Boyer-Moore is the algorithm of choice in many text editors and the Unix `grep` utility.


# Reference

[Boyer, Moore - A Fast String Searching Algorithm (1977)](https://doi.org/10.1145/359842.359859)

[Boyer-Moore String Search Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm)
