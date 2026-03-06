# Failure Function

The failure function (also called the prefix function or partial match table) is the key preprocessing step for the KMP algorithm. For a pattern $P[0..m-1]$, the failure function $\pi[i]$ gives the length of the longest proper prefix of $P[0..i]$ that is also a suffix of $P[0..i]$.

## Definition

$$

\pi[i] = \max\{k : 0 \le k < i+1 \;\text{and}\; P[0..k-1] = P[i-k+1..i]\}

$$

In words, $\pi[i]$ is the length of the longest string that is both a proper prefix and a suffix of the substring $P[0..i]$.

## Example

For pattern $P = \texttt{ABABAC}$:

| $i$     | 0 | 1 | 2 | 3 | 4 | 5 |
|---------|---|---|---|---|---|---|
| $P[i]$  | A | B | A | B | A | C |
| $\pi[i]$| 0 | 0 | 1 | 2 | 3 | 0 |

At $i=4$, the prefix $\texttt{ABA}$ (length 3) equals the suffix $P[2..4]=\texttt{ABA}$, so $\pi[4]=3$.

## Algorithm

The failure function is computed in $O(m)$ time using the observation that $\pi[i]$ can be found by extending $\pi[i-1]$. If $P[\pi[i-1]] = P[i]$, then $\pi[i] = \pi[i-1]+1$. Otherwise, we follow the chain $\pi[\pi[i-1]-1], \pi[\pi[\pi[i-1]-1]-1], \ldots$ until we find a match or reach 0.

```python
def compute_failure(pattern: str) -> list[int]:
    """Compute the failure (prefix) function for a pattern."""
    m = len(pattern)
    pi = [0] * m
    k = 0  # length of current longest prefix-suffix
    for i in range(1, m):
        while k > 0 and pattern[k] != pattern[i]:
            k = pi[k - 1]  # fall back
        if pattern[k] == pattern[i]:
            k += 1
        pi[i] = k
    return pi

# Example
pattern = "ABABAC"
print(compute_failure(pattern))
# Output: [0, 0, 1, 2, 3, 0]

pattern2 = "AABAAAB"
print(compute_failure(pattern2))
# Output: [0, 1, 0, 1, 2, 2, 3]
```

## Complexity Analysis

- **Time:** $O(m)$. Although there is a while loop inside the for loop, the total number of times $k$ is decremented across the entire computation is at most $m-1$, since each increment of $k$ happens at most once per iteration.
- **Space:** $O(m)$ for the $\pi$ array.

## Why It Works

The key insight is that when a mismatch occurs during pattern matching at position $j$ in the pattern, the failure function tells us the longest prefix of $P$ that still matches the text. This means we can skip ahead by $j - \pi[j-1]$ positions in the text alignment without missing any potential match, because any shorter shift would require a prefix-suffix overlap longer than $\pi[j-1]$, which contradicts the maximality of $\pi$.

# Reference

[Introduction to Algorithms (CLRS), Section 32.4 - The Knuth-Morris-Pratt algorithm](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)

[Prefix function - CP-Algorithms](https://cp-algorithms.com/string/prefix-function.html)
