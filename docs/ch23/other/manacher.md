# Manacher's Algorithm

Finding the longest palindromic substring by brute force takes $O(n^2)$ time (expand around each center). **Manacher's algorithm** achieves $O(n)$ by exploiting the symmetric structure of palindromes: when processing a position within a known palindrome, previously computed radii can be reused, avoiding redundant character comparisons.

## Key Idea

Manacher's algorithm maintains the rightmost palindrome found so far, defined by its center $C$ and right boundary $R$. For a new center $i$:

- If $i < R$, the mirror position $i' = 2C - i$ has already been processed. The palindrome radius at $i$ is at least $\min(p[i'], R - i)$.
- Expand outward from this initial radius. Any expansion beyond $R$ discovers new territory.

This ensures that each character is examined as part of an expansion at most $O(1)$ times amortized.

## Handling Even-Length Palindromes

The classic trick inserts a separator character (e.g., `#`) between every pair of characters and at both ends:

$$
\text{``abba''} \rightarrow \text{``\#a\#b\#b\#a\#''}
$$

This transforms every palindrome (odd or even length) into an odd-length palindrome in the transformed string, unifying both cases.

## Algorithm

Let $T$ be the transformed string of length $2n + 1$. Compute an array $p$ where $p[i]$ is the radius of the longest palindrome centered at $i$ in $T$.

1. Initialize $C = 0$, $R = 0$ (center and right boundary of the rightmost palindrome).
2. For each position $i$ from 0 to $|T| - 1$:
    - Set the mirror $i' = 2C - i$.
    - Initialize $p[i] = \min(p[i'], R - i)$ if $i < R$, else $p[i] = 0$.
    - Expand: while $T[i + p[i] + 1] = T[i - p[i] - 1]$, increment $p[i]$.
    - If $i + p[i] > R$, update $C = i$ and $R = i + p[i]$.
3. The longest palindromic substring has length $\max(p)$, centered at $\text{argmax}(p)$.

## Complexity

| Aspect | Value |
|---|---|
| Time | $O(n)$ |
| Space | $O(n)$ |

!!! tip "Why O(n)?"
    Each expansion step advances $R$ to the right, and $R$ never decreases. Since $R$ can advance at most $O(n)$ times total, the total number of expansion steps across all positions is $O(n)$.

## Python Implementation

```python
"""
Manacher's Algorithm — Longest Palindromic Substring in O(n).

Uses the separator trick to handle both odd and even length palindromes
uniformly, then computes palindrome radii in linear time.
"""


# === Manacher's Algorithm ===

def manacher(s: str) -> tuple[str, int, int]:
    """Find the longest palindromic substring.

    Returns (palindrome, start_in_original, length).
    """
    if not s:
        return "", 0, 0

    # Transform: "abc" -> "^#a#b#c#$"
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    center = 0
    right = 0

    for i in range(1, n - 1):
        mirror = 2 * center - i

        if i < right:
            p[i] = min(p[mirror], right - i)

        # Expand around center i
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1

        # Update rightmost palindrome
        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Find the maximum radius
    max_radius = max(p)
    max_center = p.index(max_radius)

    # Map back to original string
    start = (max_center - max_radius) // 2
    return s[start:start + max_radius], start, max_radius


# === All Palindromic Substrings ===

def all_palindrome_radii(s: str) -> list[int]:
    """Return the palindrome radius for each center in the original string.

    Odd-length palindromes only (for simplicity).
    """
    n = len(s)
    p = [0] * n
    center = 0
    right = 0

    for i in range(n):
        mirror = 2 * center - i
        if i < right:
            p[i] = min(p[mirror], right - i)

        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0
               and s[i + p[i] + 1] == s[i - p[i] - 1]):
            p[i] += 1

        if i + p[i] > right:
            center = i
            right = i + p[i]

    return p


# === Main ===

if __name__ == "__main__":
    test_cases = ["babad", "cbbd", "abaaba", "abacaba"]
    for s in test_cases:
        palindrome, start, length = manacher(s)
        print(f"'{s}' -> '{palindrome}' (start={start}, length={length})")
    # Output:
    # 'babad' -> 'bab' (start=0, length=3)
    # 'cbbd' -> 'bb' (start=1, length=2)
    # 'abaaba' -> 'abaaba' (start=0, length=6)
    # 'abacaba' -> 'abacaba' (start=0, length=7)
```

## Worked Example

For $s = \text{``abacaba''}$:

Transformed: `^#a#b#a#c#a#b#a#$`

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $T$ | ^ | # | a | # | b | # | a | # | c | # | a | # | b | # | a | # | $ |
| $p$ | 0 | 0 | 1 | 0 | 3 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 3 | 0 | 1 | 0 | 0 |

The maximum radius is 7 at position 8 (character 'c'). Mapping back: start = $(8 - 7) / 2 = 0$, length = 7. The longest palindrome is "abacaba".

## Reference

- Manacher, G. (1975). A new linear-time "on-line" algorithm for finding the smallest initial palindrome of a string. *Journal of the ACM*, 22(3), 346-351.
