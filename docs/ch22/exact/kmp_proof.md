# KMP Correctness

The Knuth-Morris-Pratt (KMP) algorithm finds all occurrences of a pattern $P[0..m-1]$ in a text $T[0..n-1]$ in $O(n+m)$ time. This page presents a formal proof of its correctness and running time.

## Algorithm Recap

KMP maintains a pointer $i$ into the text and a pointer $j$ into the pattern. When $T[i] = P[j]$, both advance. On a mismatch (or after a full match), $j$ is reset to $\pi[j-1]$ (or $\pi[m-1]$), where $\pi$ is the failure function. The text pointer $i$ never moves backward.

```python
def kmp_search(text: str, pattern: str) -> list[int]:
    """KMP string matching algorithm."""
    n, m = len(text), len(pattern)
    if m == 0:
        return []

    # Build failure function
    pi = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and pattern[k] != pattern[i]:
            k = pi[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        pi[i] = k

    # Search
    occurrences = []
    j = 0
    for i in range(n):
        while j > 0 and pattern[j] != text[i]:
            j = pi[j - 1]
        if pattern[j] == text[i]:
            j += 1
        if j == m:
            occurrences.append(i - m + 1)
            j = pi[j - 1]
    return occurrences


# Example
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
print(kmp_search(text, pattern))
# Output: [9]
```

## Correctness Proof

**Theorem.** KMP reports position $s$ if and only if $T[s..s+m-1] = P[0..m-1]$.

**Proof.** We prove the following loop invariant:

*At the start of each iteration with text index $i$, the value $j$ equals the length of the longest proper prefix of $P$ that matches a suffix of $T[0..i-1]$.*

**Initialization:** Before the loop, $i = 0$ and $j = 0$. No characters have been compared, so the longest matching prefix has length 0. The invariant holds.

**Maintenance:** Suppose the invariant holds at the start of iteration $i$. We have $P[0..j-1] = T[i-j..i-1]$.

- If $T[i] = P[j]$, then $P[0..j] = T[i-j..i]$, so after incrementing $j$ the invariant holds for $i+1$.
- If $T[i] \neq P[j]$ and $j > 0$, we set $j = \pi[j-1]$. By definition of $\pi$, $P[0..\pi[j-1]-1]$ is the longest proper prefix of $P[0..j-1]$ that is also a suffix. Since $P[0..j-1] = T[i-j..i-1]$, we have $P[0..\pi[j-1]-1] = T[i-\pi[j-1]..i-1]$, preserving the invariant. We repeat until either a match is found or $j = 0$.
- If $j = 0$ and $T[i] \neq P[0]$, no prefix of $P$ matches any suffix of $T[0..i]$, and $j$ remains 0.

**Termination:** When $j = m$, the invariant gives $P[0..m-1] = T[i-m+1..i]$, so a valid match at position $i - m + 1$ is reported. Setting $j = \pi[m-1]$ correctly handles overlapping matches. $\square$

## Time Complexity Proof

**Theorem.** KMP runs in $O(n + m)$ time.

**Proof.** Define the potential $\Phi = j$ (the current position in the pattern). In each iteration of the for loop:

- Each character comparison that leads to $j \gets j + 1$ increases $\Phi$ by 1.
- Each fallback $j \gets \pi[j-1]$ strictly decreases $\Phi$ (since $\pi[j-1] < j$).
- $\Phi$ is always non-negative and increases by at most 1 per iteration of the for loop.

Over all $n$ iterations, $\Phi$ increases at most $n$ times. Since $\Phi \ge 0$, the total number of decreases (fallback operations) is also at most $n$. Thus the total work in the search phase is $O(n)$. The preprocessing phase takes $O(m)$ by an identical argument. Total: $O(n + m)$. $\square$


# Reference

[Introduction to Algorithms (CLRS), Section 32.4 - The Knuth-Morris-Pratt algorithm](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)

[Knuth, Morris, Pratt - Fast Pattern Matching in Strings (1977)](https://doi.org/10.1137/0206024)
