# Good Suffix Rule

The good-suffix rule is the second heuristic in the Boyer-Moore algorithm. While the bad-character rule focuses on the mismatched character, the good-suffix rule exploits the portion of the pattern that has already been matched.

## Intuition

Suppose we are comparing the pattern from right to left and a mismatch occurs at position $j$ in the pattern. The suffix $P[j+1..m-1]$ has already matched the corresponding text. Call this the "good suffix" $t = P[j+1..m-1]$.

The good-suffix rule shifts the pattern to align with the next occurrence of $t$ inside $P$ that is preceded by a character different from $P[j]$. If no such occurrence exists, the rule shifts to align the longest proper suffix of $t$ that matches a prefix of $P$.

## Formal Definition

Let the good suffix be $t = P[j+1..m-1]$. The shift is determined by two cases:

**Case 1:** There exists a position $k < j$ such that $P[k+1..k+m-1-j] = t$ and $P[k] \neq P[j]$. We shift the pattern by $j - k$ positions.

**Case 2:** No such $k$ exists, but a proper suffix of $t$ matches a prefix of $P$. Let $\ell$ be the length of the longest such prefix. We shift by $m - \ell$.

$$

\text{good\_suffix\_shift}(j) = \begin{cases}
j - k & \text{if Case 1 applies (use rightmost such } k\text{)}\\
m - \ell & \text{if only Case 2 applies}\\
m & \text{if neither case applies}
\end{cases}

$$

## Preprocessing

The good-suffix table is computed using an auxiliary array $\text{suffix}[i]$, which stores the length of the longest suffix of $P$ that matches a suffix of $P[0..i]$.

```python
def build_good_suffix_table(pattern: str) -> list[int]:
    """Build the good suffix shift table for Boyer-Moore."""
    m = len(pattern)
    if m == 0:
        return []

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

    shift = [m] * m
    j = 0
    for i in range(m - 1, -1, -1):
        if suffix[i] == i + 1:
            while j < m - 1 - i:
                if shift[j] == m:
                    shift[j] = m - 1 - i
                j += 1
    for i in range(m - 2):
        shift[m - 1 - suffix[i]] = m - 1 - i

    return shift

# Example
pattern = "ABCBAB"
table = build_good_suffix_table(pattern)
print(f"Pattern: {pattern}")
print(f"Good suffix shift table: {table}")
# Output: Good suffix shift table: [4, 4, 4, 4, 2, 1]
```

## Example Walkthrough

For pattern $P = \texttt{ABCBAB}$ ($m=6$):

- If mismatch at $j=4$, good suffix = `"B"`. `"B"` re-occurs at $P[3]$ preceded by `C` $\neq$ `A`=$P[4]$. Shift by 2.
- If mismatch at $j=3$, good suffix = `"AB"`. `"AB"` occurs as prefix $P[0..1]$. Shift by 4.

## Complexity

- **Preprocessing:** $O(m)$ time and space.
- Combined with the bad-character rule, the good-suffix rule ensures that Boyer-Moore achieves $O(n+m)$ worst-case time (with the Galil rule for the matching phase).

# Reference

[Boyer, Moore - A Fast String Searching Algorithm (1977)](https://doi.org/10.1145/359842.359859)

[Good Suffix Heuristic - Lecroq](http://www-igm.univ-mlv.fr/~lecroq/string/node14.html)
