# KMP Algorithm

The naive string matching algorithm slides the pattern one position at a time, giving $O(nm)$ worst-case performance. The **Knuth-Morris-Pratt (KMP)** algorithm eliminates redundant comparisons by precomputing a **failure function** (also called the prefix function) that tells the algorithm how far to shift the pattern when a mismatch occurs. This achieves $O(n + m)$ time with $O(m)$ extra space.

## Key Insight

When a mismatch occurs at position $j$ of the pattern after matching $j$ characters, the naive approach restarts matching from the next text position. KMP observes that some prefix of the pattern may already match a suffix of the matched portion — so the pattern can be shifted forward without re-examining those characters.

## Failure Function

The failure function $\pi[j]$ stores the length of the longest proper prefix of the pattern $P[0 \ldots j]$ that is also a suffix. Formally:

$$
\pi[j] = \max\{k : 0 \le k < j \text{ and } P[0 \ldots k-1] = P[j-k+1 \ldots j]\}
$$

with $\pi[0] = 0$ (no proper prefix for a single character).

**Example.** For pattern $P = \text{``ABABAC''}$:

| $j$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $P[j]$ | A | B | A | B | A | C |
| $\pi[j]$ | 0 | 0 | 1 | 2 | 3 | 0 |

At $j = 4$, the longest proper prefix-suffix of "ABABA" is "ABA" (length 3).

## Building the Failure Function

The failure function is computed iteratively in $O(m)$ time using the observation that $\pi[j]$ extends $\pi[j-1]$:

1. Set $\pi[0] = 0$ and $k = 0$.
2. For $j = 1, 2, \ldots, m-1$:
    - While $k > 0$ and $P[k] \ne P[j]$, set $k = \pi[k-1]$ (fall back).
    - If $P[k] = P[j]$, increment $k$.
    - Set $\pi[j] = k$.

## Search Algorithm

With the failure function precomputed:

1. Initialize $j = 0$ (position in pattern).
2. For each character $T[i]$ in the text:
    - While $j > 0$ and $P[j] \ne T[i]$, set $j = \pi[j-1]$.
    - If $P[j] = T[i]$, increment $j$.
    - If $j = m$, a match is found at position $i - m + 1$. Set $j = \pi[j-1]$ to continue searching.

## Complexity

| Aspect | Value |
|---|---|
| Preprocessing | $O(m)$ |
| Searching | $O(n)$ |
| Total | $O(n + m)$ |
| Space | $O(m)$ for the failure function |

!!! tip "Why O(n + m)?"
    Each character of the text is compared at most a constant number of times. The value of $j$ increases by at most 1 per text character and can only decrease by following failure links, but the total number of decreases is bounded by the total number of increases.

## Python Implementation

```python
"""
Knuth-Morris-Pratt (KMP) String Matching Algorithm.

Preprocesses the pattern to build a failure function, then scans
the text in O(n + m) time to find all occurrences.
"""


# === Failure Function ===

def compute_failure(pattern: str) -> list[int]:
    """Build the KMP failure function (prefix function).

    failure[j] = length of longest proper prefix of pattern[0..j]
    that is also a suffix.
    """
    m = len(pattern)
    failure = [0] * m
    k = 0

    for j in range(1, m):
        while k > 0 and pattern[k] != pattern[j]:
            k = failure[k - 1]
        if pattern[k] == pattern[j]:
            k += 1
        failure[j] = k

    return failure


# === KMP Search ===

def kmp_search(text: str, pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using KMP.

    Returns list of starting indices of matches.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return []

    failure = compute_failure(pattern)
    matches = []
    j = 0  # position in pattern

    for i in range(n):
        while j > 0 and pattern[j] != text[i]:
            j = failure[j - 1]
        if pattern[j] == text[i]:
            j += 1
        if j == m:
            matches.append(i - m + 1)
            j = failure[j - 1]

    return matches


# === Main ===

if __name__ == "__main__":
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"

    failure = compute_failure(pattern)
    matches = kmp_search(text, pattern)

    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    print(f"Failure: {failure}")
    print(f"Matches at: {matches}")
    # Output:
    # Text:    ABABDABACDABABCABAB
    # Pattern: ABABCABAB
    # Failure: [0, 0, 1, 2, 0, 1, 2, 3, 4]
    # Matches at: [9]
```

## Worked Example

**Text:** `AABABAA`, **Pattern:** `AABA`

Failure function for "AABA": $\pi = [0, 1, 0, 1]$.

| Step | $i$ | $T[i]$ | $j$ before | Match? | $j$ after |
|---|---|---|---|---|---|
| 1 | 0 | A | 0 | Yes | 1 |
| 2 | 1 | A | 1 | Yes | 2 |
| 3 | 2 | B | 2 | No, fall to $\pi[1]=1$; $P[1]$=A vs B: No, fall to $\pi[0]=0$; $P[0]$=A vs B: No | 0 |
| 4 | 3 | A | 0 | Yes | 1 |
| 5 | 4 | B | 1 | No, fall to $\pi[0]=0$; $P[0]$=A vs B: No | 0 |
| 6 | 5 | A | 0 | Yes | 1 |
| 7 | 6 | A | 1 | Yes | 2 |

No complete match found (never reached $j = 4$).

## Reference

- Knuth, D. E., Morris, J. H., & Pratt, V. R. (1977). Fast pattern matching in strings. *SIAM Journal on Computing*, 6(2), 323-350.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 32. MIT Press.
