# Rabin-Karp Algorithm

The naive approach to string matching compares every character of the pattern at every text position, costing $O(nm)$ in the worst case. The **Rabin-Karp algorithm** uses hashing to reduce the expected number of character comparisons: instead of comparing strings character by character, it compares hash values first and only performs a full comparison when hashes match. A **rolling hash** makes each hash update $O(1)$, yielding $O(n + m)$ expected time.

## Rolling Hash

A rolling hash computes the hash of each length-$m$ substring of the text in $O(1)$ time by updating the previous hash. Using a polynomial hash with base $d$ and modulus $q$:

$$
h(s[i \ldots i+m-1]) = \left(\sum_{k=0}^{m-1} s[i+k] \cdot d^{m-1-k}\right) \bmod q
$$

When sliding the window from position $i$ to $i+1$:

$$
h(s[i+1 \ldots i+m]) = \bigl(d \cdot (h(s[i \ldots i+m-1]) - s[i] \cdot d^{m-1}) + s[i+m]\bigr) \bmod q
$$

This update removes the contribution of the leftmost character and incorporates the new rightmost character.

## Algorithm

1. Compute the hash of the pattern $P$ and the hash of the first window $T[0 \ldots m-1]$.
2. Precompute $h = d^{m-1} \bmod q$.
3. For each position $i$ from 0 to $n - m$:
    - If the hashes match, perform a character-by-character comparison to confirm (avoiding false positives).
    - Compute the hash of the next window using the rolling update.

## Complexity

| Aspect | Value |
|---|---|
| Expected time | $O(n + m)$ |
| Worst-case time | $O(nm)$ (many hash collisions) |
| Space | $O(1)$ beyond input |
| Preprocessing | $O(m)$ |

!!! warning "Hash Collisions"
    When hashes match but strings differ (spurious hit), a full $O(m)$ comparison is needed. Choosing a large prime $q$ minimizes collisions. The expected number of spurious hits is $O(n/q)$, which is negligible for large $q$.

## Python Implementation

```python
"""
Rabin-Karp String Matching Algorithm.

Uses a polynomial rolling hash to find all occurrences of a pattern
in a text with O(n + m) expected time.
"""


# === Rabin-Karp Search ===

def rabin_karp(text: str, pattern: str, d: int = 256, q: int = 101) -> list[int]:
    """Find all occurrences of pattern in text using Rabin-Karp.

    Args:
        text: The text to search in.
        pattern: The pattern to search for.
        d: Base for the hash function (alphabet size).
        q: A prime modulus for the hash function.

    Returns:
        List of starting indices where pattern occurs.
    """
    n, m = len(text), len(pattern)
    if m > n or m == 0:
        return []

    matches = []
    h = pow(d, m - 1, q)  # d^(m-1) mod q

    # Compute initial hashes
    p_hash = 0  # pattern hash
    t_hash = 0  # text window hash
    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % q
        t_hash = (d * t_hash + ord(text[i])) % q

    # Slide the window
    for i in range(n - m + 1):
        if p_hash == t_hash:
            # Verify character by character (avoid spurious hits)
            if text[i:i + m] == pattern:
                matches.append(i)

        # Compute hash for next window
        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if t_hash < 0:
                t_hash += q

    return matches


# === Multi-Pattern Variant ===

def rabin_karp_multi(
    text: str, patterns: list[str], d: int = 256, q: int = 101
) -> dict[str, list[int]]:
    """Search for multiple patterns of the same length."""
    if not patterns:
        return {}

    m = len(patterns[0])
    results = {p: [] for p in patterns}

    # Compute pattern hashes
    p_hashes = {}
    for p in patterns:
        h_val = 0
        for ch in p:
            h_val = (d * h_val + ord(ch)) % q
        p_hashes.setdefault(h_val, []).append(p)

    n = len(text)
    if m > n:
        return results

    h = pow(d, m - 1, q)
    t_hash = 0
    for i in range(m):
        t_hash = (d * t_hash + ord(text[i])) % q

    for i in range(n - m + 1):
        if t_hash in p_hashes:
            for p in p_hashes[t_hash]:
                if text[i:i + m] == p:
                    results[p].append(i)

        if i < n - m:
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
            if t_hash < 0:
                t_hash += q

    return results


# === Main ===

if __name__ == "__main__":
    text = "AABAACAADAABAABA"
    pattern = "AABA"

    matches = rabin_karp(text, pattern)
    print(f"Text:    {text}")
    print(f"Pattern: {pattern}")
    print(f"Matches at: {matches}")

    # Multi-pattern example
    patterns = ["AABA", "AACA"]
    multi = rabin_karp_multi(text, patterns)
    print(f"\nMulti-pattern search:")
    for p, idx in multi.items():
        print(f"  '{p}': {idx}")
    # Output:
    # Text:    AABAACAADAABAABA
    # Pattern: AABA
    # Matches at: [0, 9, 12]
    #
    # Multi-pattern search:
    #   'AABA': [0, 9, 12]
    #   'AACA': [3]
```

## Worked Example

**Text:** `ABCABC`, **Pattern:** `ABC`, $d = 256$, $q = 101$.

1. Pattern hash: $(65 \cdot 256^2 + 66 \cdot 256 + 67) \bmod 101 = 4259907 \bmod 101 = 79$.
2. Window "ABC" hash: 79. **Match!** Verify: "ABC" = "ABC". Report position 0.
3. Roll: remove 'A', add 'A'. Window "BCA" hash: $(256 \cdot (79 - 65 \cdot 256^2 \bmod 101) + 65) \bmod 101$. Compute: no match.
4. Roll: window "CAB" — no match.
5. Roll: window "ABC" hash = 79. **Match!** Verify: "ABC" = "ABC". Report position 3.

## Reference

- Karp, R. M., & Rabin, M. O. (1987). Efficient randomized pattern-matching algorithms. *IBM Journal of Research and Development*, 31(2), 249-260.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 32. MIT Press.
