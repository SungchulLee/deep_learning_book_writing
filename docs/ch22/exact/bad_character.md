# Bad Character Rule

The bad-character rule is one of the two heuristics used in the Boyer-Moore algorithm to determine how far to shift the pattern when a mismatch occurs. It examines the character in the text that caused the mismatch and uses it to compute a shift.

## Intuition

When comparing the pattern right to left, suppose a mismatch occurs: $T[i+j] \neq P[j]$. The character $c = T[i+j]$ is the "bad character." The rule asks: where does $c$ last occur in $P[0..j-1]$?

- If $c$ occurs at position $k < j$ in the pattern, shift the pattern so that $P[k]$ aligns with $T[i+j]$. The shift amount is $j - k$.
- If $c$ does not occur in $P[0..j-1]$, shift the entire pattern past the mismatch position. The shift amount is $j + 1$.

$$

\text{bad\_char\_shift}(j, c) = j - \max\{k : k < j \text{ and } P[k] = c\}

$$

If no such $k$ exists, we use $k = -1$, giving a shift of $j + 1$.

## Preprocessing

We build a lookup table that, for each character in the alphabet, stores the rightmost position of that character in the pattern.

```python
def build_bad_character_table(pattern: str) -> dict[str, int]:
    """Build the bad character table for Boyer-Moore.

    Maps each character to its rightmost position in the pattern.
    """
    table = {}
    for i, ch in enumerate(pattern):
        table[ch] = i
    return table

def bad_character_search(text: str, pattern: str) -> list[int]:
    """Boyer-Moore using only the bad-character heuristic."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    bad_char = build_bad_character_table(pattern)
    occurrences = []
    i = 0

    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            occurrences.append(i)
            i += 1
        else:
            bc_pos = bad_char.get(text[i + j], -1)
            shift = j - bc_pos
            i += max(1, shift)
    return occurrences

# Example
text = "ABCABCABABC"
pattern = "ABABC"
print(bad_character_search(text, pattern))
# Output: [6]
```

## Extended Bad-Character Rule

The simple version only stores the rightmost occurrence of each character. The **extended** version stores, for each position $j$ in the pattern and each character $c$, the rightmost occurrence of $c$ in $P[0..j-1]$. This provides better shifts but requires $O(m \cdot |\Sigma|)$ space.

## Complexity

- **Preprocessing:** $O(m + |\Sigma|)$ for the simple table, or $O(m \cdot |\Sigma|)$ for the extended version.
- **The bad-character rule alone does not guarantee sublinear or linear worst-case performance.** It can degenerate to $O(nm)$. However, combined with the good-suffix rule, Boyer-Moore achieves $O(n+m)$ worst-case.

# Reference

[Boyer, Moore - A Fast String Searching Algorithm (1977)](https://doi.org/10.1145/359842.359859)

[Bad Character Heuristic - GeeksforGeeks](https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/)
