# Rolling Hash

A rolling hash (also called a sliding hash) is a hash function that can be efficiently updated as a fixed-size window slides over the input. It is the core technique behind the Rabin-Karp string matching algorithm, enabling average-case $O(n+m)$ pattern matching.

## Polynomial Rolling Hash

The most common rolling hash treats a string as a polynomial evaluated modulo a prime $p$. For a string $S[0..m-1]$ over an alphabet of size $d$:

$$

H(S[0..m-1]) = \left(\sum_{i=0}^{m-1} S[i] \cdot d^{m-1-i}\right) \bmod p

$$

When the window slides from $S[i..i+m-1]$ to $S[i+1..i+m]$, the hash is updated in $O(1)$:

$$

H(S[i+1..i+m]) = \left(d \cdot \bigl(H(S[i..i+m-1]) - S[i] \cdot d^{m-1}\bigr) + S[i+m]\right) \bmod p

$$

We precompute $d^{m-1} \bmod p$ once.

```python
def rabin_karp(text: str, pattern: str, d: int = 256, p: int = 101) -> list[int]:
    """Rabin-Karp string matching with rolling hash."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    occurrences = []
    h = pow(d, m - 1, p)  # d^(m-1) mod p

    # Compute initial hashes
    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (d * p_hash + ord(pattern[i])) % p
        t_hash = (d * t_hash + ord(text[i])) % p

    for i in range(n - m + 1):
        if p_hash == t_hash:
            # Verify character by character (avoid false positives)
            if text[i:i + m] == pattern:
                occurrences.append(i)
        if i < n - m:
            # Roll the hash forward
            t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % p
            if t_hash < 0:
                t_hash += p

    return occurrences

# Example
text = "GEEKS FOR GEEKS"
pattern = "GEEK"
print(rabin_karp(text, pattern))
# Output: [0, 10]
```

## Complexity Analysis

- **Preprocessing:** $O(m)$ to compute the hash of the pattern and the first window.
- **Expected search time:** $O(n + m)$. Each window update is $O(1)$. A spurious hit (hash collision) requires $O(m)$ verification. With a good hash function, the expected number of spurious hits is $O(n/p)$, which is small for large $p$.
- **Worst case:** $O(nm)$ when every window produces a hash collision (e.g., all characters identical and bad choice of $p$).
- **Space:** $O(1)$ auxiliary.

## Choosing Good Parameters

To minimize collisions, choose $p$ as a large prime and $d$ equal to the alphabet size. Using two independent hash functions (double hashing) reduces the collision probability to approximately $1/p^2$, making false positives negligible in practice.

# Reference

[Rabin-Karp Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm)

[Introduction to Algorithms (CLRS), Section 32.2 - The Rabin-Karp algorithm](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)
