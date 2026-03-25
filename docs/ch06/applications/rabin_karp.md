# Rabin-Karp

Naive string matching compares a pattern of length $m$ against every position in a text of length $n$, taking $O(nm)$ time in the worst case. The Rabin-Karp algorithm replaces most character-by-character comparisons with hash comparisons, achieving $O(n + m)$ expected time by using a **rolling hash** that updates in $O(1)$ as the window slides across the text.

## Rolling Hash Concept

The idea is to compute a hash of each length-$m$ substring of the text and compare it to the hash of the pattern. If the hashes differ, the substring cannot match and no character comparison is needed. If the hashes agree, a character-by-character verification confirms or rejects the match (to handle hash collisions).

The critical insight is that when the window slides one position to the right, the new hash can be computed from the old hash in $O(1)$ time rather than recomputing from scratch in $O(m)$ time.

## Polynomial Rolling Hash

Treat each character as a digit in base $d$ (where $d = |\Sigma|$ is the alphabet size). The hash of the substring $T[s \ldots s+m-1]$ is

$$
H(s) = \left(\sum_{j=0}^{m-1} T[s+j] \cdot d^{m-1-j}\right) \bmod q
$$

where $q$ is a large prime chosen to reduce collisions.

When the window shifts from position $s$ to $s+1$, the new hash is computed by removing the leading character and adding the trailing character:

$$
H(s+1) = \bigl(d \cdot (H(s) - T[s] \cdot d^{m-1}) + T[s+m]\bigr) \bmod q
$$

The value $d^{m-1} \bmod q$ is precomputed once at the start, making each update $O(1)$.

## Algorithm

1. Compute the hash of the pattern: $H_P = H(\text{pattern})$.
2. Compute the hash of the first window: $H(0)$.
3. For each position $s = 0, 1, \ldots, n - m$:
    - If $H(s) = H_P$, verify by comparing characters $T[s \ldots s+m-1]$ with the pattern.
    - If $s < n - m$, compute $H(s+1)$ from $H(s)$ using the rolling update.

## Complexity Analysis

**Preprocessing**: computing $H_P$ and $H(0)$ takes $O(m)$ time.

**Matching phase**: computing each rolling hash update takes $O(1)$. There are $n - m + 1$ positions to check. If a hash match occurs, verification takes $O(m)$.

**Expected time** (with a good choice of $q$): the probability of a spurious hit (hash match without character match) is approximately $1/q$. The expected number of spurious hits is $(n - m + 1)/q$, each costing $O(m)$ for verification. With $q$ chosen to be at least $m$, the expected total time is

$$
O(n + m)
$$

**Worst-case time**: if every position produces a hash match (e.g., text = "aaa...a" and pattern = "aaa"), every position requires $O(m)$ verification, giving

$$
O(nm)
$$

This worst case can be mitigated by using multiple hash functions or choosing $q$ to be very large.

## Multi-Pattern Search

Rabin-Karp extends naturally to searching for multiple patterns simultaneously. Given $k$ patterns of the same length $m$, store all pattern hashes in a hash set. For each text window, check if the window hash appears in the set. The expected time is

$$
O(n + km)
$$

since the hash set lookup is $O(1)$ and preprocessing all patterns takes $O(km)$.

## Python Implementation

```python
"""
Rabin-Karp string matching algorithm.

Uses a polynomial rolling hash to achieve O(n + m) expected time
for single-pattern matching.
"""


# === Rolling Hash Parameters ===

BASE = 256       # alphabet size (extended ASCII)
MOD = 101        # prime modulus


# === Rabin-Karp Algorithm ===

def rabin_karp(text, pattern):
    """Find all occurrences of pattern in text using Rabin-Karp.

    Returns a list of starting indices where pattern occurs.
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Precompute d^(m-1) mod q
    h = pow(BASE, m - 1, MOD)

    # Compute initial hashes
    p_hash = 0  # pattern hash
    t_hash = 0  # text window hash
    for i in range(m):
        p_hash = (BASE * p_hash + ord(pattern[i])) % MOD
        t_hash = (BASE * t_hash + ord(text[i])) % MOD

    matches = []
    for s in range(n - m + 1):
        # Check hash match
        if t_hash == p_hash:
            # Verify character by character
            if text[s:s + m] == pattern:
                matches.append(s)

        # Compute rolling hash for next window
        if s < n - m:
            t_hash = (BASE * (t_hash - ord(text[s]) * h)
                       + ord(text[s + m])) % MOD
            if t_hash < 0:
                t_hash += MOD

    return matches


# === Demonstration ===

if __name__ == "__main__":
    text = "AABAACAADAABAABA"
    pattern = "AABA"
    result = rabin_karp(text, pattern)
    print(f"Text:    '{text}'")
    print(f"Pattern: '{pattern}'")
    print(f"Found at indices: {result}")

    # Multiple occurrences
    text2 = "abcabcabc"
    pattern2 = "abc"
    result2 = rabin_karp(text2, pattern2)
    print(f"\nText:    '{text2}'")
    print(f"Pattern: '{pattern2}'")
    print(f"Found at indices: {result2}")
```

**Output:**
```
Text:    'AABAACAADAABAABA'
Pattern: 'AABA'
Found at indices: [0, 9, 12]

Text:    'abcabcabc'
Pattern: 'abc'
Found at indices: [0, 3, 6]
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 32](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Karp, R. M. and Rabin, M. O. "Efficient Randomized Pattern-Matching Algorithms." *IBM Journal of Research and Development*, 31(2), 1987.
