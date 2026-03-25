# Suffix Array Applications

The suffix array, especially when augmented with the LCP array, serves as a versatile foundation for a wide range of string processing problems. Many tasks that traditionally required suffix trees -- such as finding longest repeated substrings, longest common substrings between two strings, and counting distinct substrings -- can be solved just as efficiently with suffix arrays while using far less memory. This section surveys the most important applications and provides concrete algorithms for each.

## Pattern Matching

The most fundamental application of suffix arrays is searching for all occurrences of a pattern $P[0..m-1]$ in a text $T[0..n-1]$.

### Binary Search Approach

Since the suffix array stores suffixes in sorted order, all suffixes that begin with $P$ form a contiguous range $[\ell, r]$ in the suffix array. Two binary searches find this range:

1. **Lower bound**: Find the smallest $\ell$ such that suffix($\text{SA}[\ell]$) has $P$ as a prefix
2. **Upper bound**: Find the largest $r$ such that suffix($\text{SA}[r]$) has $P$ as a prefix

The number of occurrences is $r - \ell + 1$, and each occurrence position is $\text{SA}[k]$ for $k \in [\ell, r]$.

**Time complexity**: $O(m \log n)$ without the LCP array, or $O(m + \log n)$ with LCP-enhanced binary search.

??? example "Finding all occurrences of 'an' in 'banana'"
    With $T = \texttt{banana\$}$ and $\text{SA} = [6, 5, 3, 1, 0, 4, 2]$:

    - Lower bound binary search finds $\ell = 2$ (suffix `ana$` at $\text{SA}[2] = 3$)
    - Upper bound binary search finds $r = 3$ (suffix `anana$` at $\text{SA}[3] = 1$)
    - Pattern `an` occurs at positions 3 and 1

## Longest Repeated Substring

The **longest repeated substring (LRS)** is the longest string that appears at least twice in $T$. With the LCP array, this is simply the maximum value:

$$
\text{LRS length} = \max_{1 \leq k \leq n} \text{LCP}[k]
$$

The actual substring is $T[\text{SA}[k^*] .. \text{SA}[k^*] + \text{LCP}[k^*] - 1]$, where $k^* = \arg\max_k \text{LCP}[k]$.

**Time complexity**: $O(n)$ after the suffix array and LCP array are built.

??? example "LRS of 'banana'"
    For $T = \texttt{banana\$}$ with $\text{LCP} = [0, 0, 1, 3, 0, 0, 2]$:

    - Maximum LCP value is 3, occurring at position $k = 3$
    - $\text{SA}[3] = 1$, so the LRS is $T[1..3] = \texttt{ana}$
    - Indeed, `ana` appears at positions 1 and 3

## Longest Common Substring of Two Strings

Given two strings $S_1$ and $S_2$, their **longest common substring (LCS)** can be found by concatenating them with a separator:

$$
T = S_1 \cdot \texttt{\#} \cdot S_2 \cdot \texttt{\$}
$$

where $\texttt{\#}$ and $\texttt{\$}$ are distinct sentinel characters not in either string. Build the suffix array and LCP array of $T$, then scan for the maximum LCP value between adjacent suffixes that originate from different strings.

Formally, let $n_1 = |S_1|$. A suffix starting at position $i$ belongs to $S_1$ if $i < n_1$ and to $S_2$ if $i > n_1$. The LCS length is:

$$
\text{LCS length} = \max_{\substack{1 \leq k \leq |T| \\ \text{SA}[k] \text{ and } \text{SA}[k-1] \\ \text{from different strings}}} \text{LCP}[k]
$$

**Time complexity**: $O(n_1 + n_2)$ total.

## Counting Distinct Substrings

Every substring of $T$ is a prefix of some suffix. Suffix $\text{SA}[k]$ contributes $(n - \text{SA}[k])$ substrings (its prefixes of lengths 1 through $n - \text{SA}[k]$). However, $\text{LCP}[k]$ of these are shared with the previous suffix in sorted order. The total count of distinct substrings is:

$$
D = \sum_{k=0}^{n} (n - \text{SA}[k]) - \sum_{k=1}^{n} \text{LCP}[k]
$$

**Time complexity**: $O(n)$ after construction.

## Longest Common Prefix Queries

The LCP of any two suffixes (not just adjacent ones in the suffix array) can be computed using the **range minimum query (RMQ)** property:

$$
\text{lcp}(\text{suffix}(\text{SA}[i]),\; \text{suffix}(\text{SA}[j])) = \min_{i < k \leq j} \text{LCP}[k]
$$

By building a sparse table over the LCP array in $O(n \log n)$ preprocessing time, each query can be answered in $O(1)$ time.

!!! tip "RMQ with linear preprocessing"
    Using the Bender-Farach-Colton algorithm for the special case of $\pm 1$ RMQ (which the LCP array satisfies after a reduction), preprocessing takes only $O(n)$ time while maintaining $O(1)$ query time.

## Lexicographic Comparison of Substrings

Given two substrings $T[i..i+\ell_1-1]$ and $T[j..j+\ell_2-1]$, their lexicographic order can be determined in $O(1)$ time using:

1. Compute $L = \text{lcp}(\text{suffix}(i), \text{suffix}(j))$ via RMQ in $O(1)$
2. If $L \geq \min(\ell_1, \ell_2)$, the shorter substring is lexicographically smaller (or they are equal if $\ell_1 = \ell_2$)
3. Otherwise, compare $T[i + L]$ vs $T[j + L]$

This enables $O(1)$ comparison of arbitrary substrings after $O(n)$ preprocessing.

## Counting Occurrences of a Pattern

Beyond finding occurrences, we can **count** how many times a pattern $P$ occurs in $T$ by finding the range $[\ell, r]$ via binary search. The count is simply $r - \ell + 1$, without needing to enumerate all positions.

## Applications Summary

| Problem | Data Structure | Time |
|---------|---------------|------|
| Pattern matching | SA | $O(m \log n)$ |
| Pattern matching | SA + LCP | $O(m + \log n)$ |
| Longest repeated substring | SA + LCP | $O(n)$ |
| Longest common substring | SA + LCP | $O(n_1 + n_2)$ |
| Count distinct substrings | SA + LCP | $O(n)$ |
| LCP of any two suffixes | SA + LCP + RMQ | $O(1)$ query |
| Lexicographic substring comparison | SA + LCP + RMQ | $O(1)$ query |

## Implementation

```python
"""
Suffix array applications: pattern matching, longest repeated
substring, and counting distinct substrings.
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


# === Applications ===

def pattern_search(text: str, sa: list[int], pattern: str) -> list[int]:
    """Find all occurrences of pattern in text using binary search on SA."""
    n = len(text)
    m = len(pattern)

    # Lower bound
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        suffix = text[sa[mid]:sa[mid] + m]
        if suffix < pattern:
            lo = mid + 1
        else:
            hi = mid
    left = lo

    # Upper bound
    lo, hi = left, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        suffix = text[sa[mid]:sa[mid] + m]
        if suffix > pattern:
            hi = mid - 1
        else:
            lo = mid
    right = lo

    if text[sa[left]:sa[left] + m] != pattern:
        return []
    return sorted(sa[k] for k in range(left, right + 1))


def longest_repeated_substring(text: str, sa: list[int],
                                lcp: list[int]) -> str:
    """Find the longest repeated substring."""
    max_lcp = max(lcp)
    if max_lcp == 0:
        return ""
    k = lcp.index(max_lcp)
    return text[sa[k]:sa[k] + max_lcp]


def count_distinct_substrings(text: str, sa: list[int],
                               lcp: list[int]) -> int:
    """Count the number of distinct substrings."""
    n = len(text)
    total = sum(n - sa[k] for k in range(n))
    duplicates = sum(lcp[k] for k in range(1, n))
    return total - duplicates


# === Main ===

if __name__ == "__main__":
    text = "banana$"
    sa = build_suffix_array(text)
    lcp = build_lcp(text, sa)

    print(f"Text: {text}")
    print(f"SA:  {sa}")
    print(f"LCP: {lcp}")

    positions = pattern_search(text, sa, "ana")
    print(f"\n'ana' found at positions: {positions}")

    lrs = longest_repeated_substring(text, sa, lcp)
    print(f"Longest repeated substring: '{lrs}'")

    count = count_distinct_substrings(text, sa, lcp)
    print(f"Distinct substrings: {count}")
```

## Reference

- Manber, U. and Myers, G. (1993). *Suffix arrays: A new method for on-line string searches*. SIAM Journal on Computing, 22(5), 935-948.
- Abouelhoda, M. I., Kurtz, S., and Ohlebusch, E. (2004). *Replacing suffix trees with enhanced suffix arrays*. Journal of Discrete Algorithms, 2(1), 53-86.
