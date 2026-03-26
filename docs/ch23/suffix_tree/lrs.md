# Longest Repeated Substring

Given a string, finding the longest substring that occurs at least twice is a classic problem in string algorithms. This problem appears in data compression (identifying repeated patterns), bioinformatics (finding repeated motifs in DNA), and music analysis (detecting repeated musical phrases). While a brute-force approach that checks all pairs of substrings takes $O(n^3)$ time, suffix trees and suffix arrays solve this in $O(n)$ time. This section presents both approaches and their relationship.

## Problem Statement

Given a string $T[0..n-1]$ of length $n$, the **Longest Repeated Substring (LRS)** is the longest string $w$ that occurs at least twice as a contiguous substring of $T$. The two occurrences may overlap.

$$
\text{LRS}(T) = \arg\max_{w} |w| \quad \text{such that } w = T[i..i+|w|-1] = T[j..j+|w|-1] \text{ for some } i \neq j
$$

## Suffix Tree Solution

In the suffix tree of $T\$$, every internal node represents a repeated substring (since an internal node has at least two children, its path label occurs at least twice). The LRS is the path label of the **deepest internal node** — the one with the greatest string depth.

### Algorithm

1. Build the suffix tree of $T\$$ in $O(n)$ time
2. Traverse all internal nodes and find the one with maximum string depth
3. The path label of that node is the LRS

$$
\text{LRS} = \text{path}(v^*) \quad \text{where } v^* = \arg\max_{v \text{ internal}} \text{depth}(v)
$$

**Time complexity**: $O(n)$ for construction and $O(n)$ for the traversal.

??? example "LRS of 'banana'"
    The suffix tree of `banana$` has internal nodes with path labels:

    - `a` (depth 1): appears at positions 1, 3, 5
    - `ana` (depth 3): appears at positions 1, 3
    - `na` (depth 2): appears at positions 2, 4

    The deepest internal node has path label `ana` with depth 3, so $\text{LRS} = \texttt{ana}$.

## Suffix Array Solution

With the suffix array and LCP array, the LRS is found by taking the maximum value in the LCP array:

$$
\text{LRS length} = \max_{1 \leq k \leq n} \text{LCP}[k]
$$

The LRS itself is $T[\text{SA}[k^*] .. \text{SA}[k^*] + \text{LCP}[k^*] - 1]$, where $k^* = \arg\max_k \text{LCP}[k]$.

### Why This Works

Two adjacent suffixes in the sorted suffix array with LCP value $\ell$ share a common prefix of length $\ell$. This prefix appears at (at least) two positions in $T$: $\text{SA}[k-1]$ and $\text{SA}[k]$. The maximum such $\ell$ gives the LRS.

??? example "LRS of 'banana' via suffix array"
    For $T = \texttt{banana\$}$:

    | Rank | SA | Suffix | LCP |
    |------|----|--------|-----|
    | 0 | 6 | `$` | 0 |
    | 1 | 5 | `a$` | 0 |
    | 2 | 3 | `ana$` | 1 |
    | 3 | 1 | `anana$` | 3 |
    | 4 | 0 | `banana$` | 0 |
    | 5 | 4 | `na$` | 0 |
    | 6 | 2 | `nana$` | 2 |

    Maximum LCP is 3 at position $k = 3$. The LRS is $T[1..3] = \texttt{ana}$, occurring at positions 1 and 3.

## Variations

### Longest Substring Occurring at Least k Times

Generalize the problem to finding the longest substring that occurs at least $k$ times. With the suffix tree, find the deepest internal node with at least $k$ leaves in its subtree.

With the suffix array, use a sliding window of size $k$ on the LCP array. The answer is:

$$
\text{LRS}_k = \max_{1 \leq i \leq n - k + 1} \min_{i \leq j < i + k - 1} \text{LCP}[j + 1]
$$

This can be computed in $O(n)$ time using a deque-based sliding window minimum.

### Non-Overlapping Longest Repeated Substring

In some applications, overlapping occurrences are not useful. The **non-overlapping LRS** requires $|i - j| \geq |w|$.

With the suffix tree, for each internal node $v$ with depth $d$, check whether the leftmost and rightmost leaf positions differ by at least $d$. The deepest such node gives the non-overlapping LRS.

With the suffix array, for each LCP value $\text{LCP}[k]$, check whether $|\text{SA}[k] - \text{SA}[k-1]| \geq \text{LCP}[k]$.

## Python Implementation

```python
"""
Longest Repeated Substring using suffix array and LCP array.
"""


# === Suffix Array Construction ===

def build_suffix_array(text: str) -> list[int]:
    """Build suffix array using prefix doubling."""
    n = len(text)
    rank = [ord(c) for c in text]
    sa = list(range(n))
    k = 1
    while k < n:
        def key(i, _k=k, _r=rank[:]):
            return (_r[i], _r[i + _k] if i + _k < n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        for j in range(1, n):
            prev = (rank[sa[j - 1]],
                    rank[sa[j - 1] + k] if sa[j - 1] + k < n else -1)
            curr = (rank[sa[j]],
                    rank[sa[j] + k] if sa[j] + k < n else -1)
            new_rank[sa[j]] = new_rank[sa[j - 1]] + (1 if curr != prev else 0)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


# === Kasai's Algorithm ===

def build_lcp(text: str, sa: list[int]) -> list[int]:
    """Compute LCP array using Kasai's algorithm."""
    n = len(sa)
    rank = [0] * n
    for k in range(n):
        rank[sa[k]] = k
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r > 0:
            j = sa[r - 1]
            while i + h < n and j + h < n and text[i + h] == text[j + h]:
                h += 1
            lcp[r] = h
            h = max(h - 1, 0)
        else:
            h = 0
    return lcp


# === Longest Repeated Substring ===

def longest_repeated_substring(text: str) -> str:
    """Find the longest repeated substring.

    Parameters
    ----------
    text : str
        Input string (sentinel is appended if not present).

    Returns
    -------
    str
        The longest repeated substring, or empty string if none exists.
    """
    if not text.endswith("$"):
        text += "$"

    sa = build_suffix_array(text)
    lcp = build_lcp(text, sa)

    max_lcp = 0
    max_idx = 0
    for k in range(1, len(text)):
        if lcp[k] > max_lcp:
            max_lcp = lcp[k]
            max_idx = k

    if max_lcp == 0:
        return ""

    return text[sa[max_idx]:sa[max_idx] + max_lcp]


# === Main ===

if __name__ == "__main__":
    test_cases = [
        "banana",
        "abcabc",
        "aabaaab",
        "abcd",
    ]

    for s in test_cases:
        lrs = longest_repeated_substring(s)
        print(f"LRS of '{s}': '{lrs}' (length {len(lrs)})")
    # Output:
    # LRS of 'banana': 'ana' (length 3)
    # LRS of 'abcabc': 'abc' (length 3)
    # LRS of 'aabaaab': 'aab' (length 3)
    # LRS of 'abcd': '' (length 0)
```

## Complexity Comparison

| Method | Time | Space |
|--------|------|-------|
| Brute force (all pairs) | $O(n^3)$ | $O(1)$ |
| Suffix tree | $O(n)$ | $O(n)$ (large constant) |
| Suffix array + LCP | $O(n)$ | $O(n)$ (small constant) |

Both suffix-based approaches achieve optimal linear time. The suffix array approach is generally preferred in practice due to its lower memory usage.

## Reference

- Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press, Section 7.12.
