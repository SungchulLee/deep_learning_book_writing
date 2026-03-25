# Kasai's Algorithm

Computing the LCP array by naively comparing each pair of adjacent suffixes in the suffix array takes $O(n^2)$ time in the worst case, since each comparison may need to scan up to $n$ characters. Kasai, Lee, Arimura, Arikawa, and Park (2001) discovered a remarkably simple algorithm that computes the LCP array in $O(n)$ time by exploiting a key observation: if we know that two adjacent suffixes share a long common prefix, then removing the first character from each suffix reduces the common prefix by at most one. This section presents the algorithm, proves its linear-time guarantee, and traces through a complete example.

## The Key Lemma

The efficiency of Kasai's algorithm rests on a single lemma about how LCP values change when we step through suffixes in text order rather than sorted order.

!!! note "Lemma (LCP decrease bound)"
    Let $\text{SA}$ be the suffix array and $\text{SA}^{-1}$ its inverse. If suffix($i$) has rank $r = \text{SA}^{-1}[i]$ and the LCP between suffix($i$) and its predecessor in sorted order is $h$, then the LCP between suffix($i+1$) and its predecessor in sorted order is at least $h - 1$.

    Formally: if $\text{LCP}[\text{SA}^{-1}[i]] = h \geq 1$, then $\text{LCP}[\text{SA}^{-1}[i+1]] \geq h - 1$.

**Intuition**: Suffix($i$) and its sorted predecessor share a common prefix of length $h$. Removing the first character from both gives suffix($i+1$) and some suffix that must appear near suffix($i+1$) in sorted order, sharing at least $h-1$ characters.

**Proof sketch**: Let suffix($i$) have sorted predecessor suffix($j$), so $\text{lcp}(\text{suffix}(i), \text{suffix}(j)) = h$. This means $T[i..i+h-1] = T[j..j+h-1]$, which implies $T[i+1..i+h-1] = T[j+1..j+h-1]$. So suffix($i+1$) and suffix($j+1$) share at least $h-1$ characters. Since the LCP of suffix($i+1$) with its sorted predecessor is at least as large as its LCP with any other suffix that precedes it in sorted order, we get $\text{LCP}[\text{SA}^{-1}[i+1]] \geq h - 1$. $\square$

## Algorithm

Kasai's algorithm processes suffixes in **text order** ($i = 0, 1, 2, \ldots, n$) rather than sorted order. It maintains a variable $h$ that tracks the current LCP value. By the lemma, $h$ decreases by at most 1 between consecutive iterations, so the total work is bounded.

```
KASAI(T, SA):
    n = length(T) - 1
    rank = inverse of SA          // rank[SA[k]] = k for all k
    LCP = array of size n+1
    LCP[0] = 0
    h = 0
    for i = 0 to n:
        r = rank[i]
        if r > 0:
            j = SA[r - 1]        // sorted predecessor of suffix(i)
            while T[i + h] == T[j + h]:
                h = h + 1
            LCP[r] = h
            h = max(h - 1, 0)    // decrease by at most 1
        else:
            h = 0                // suffix(i) is first in sorted order
    return LCP
```

The critical insight is in the line `h = max(h - 1, 0)`: rather than resetting $h$ to zero for each suffix, we start the comparison from position $h - 1$, skipping characters we already know match.

## Worked Example

For $T = \texttt{banana\$}$ with $\text{SA} = [6, 5, 3, 1, 0, 4, 2]$:

First compute the inverse: $\text{SA}^{-1} = [4, 3, 6, 2, 5, 1, 0]$.

| Step | $i$ | $r = \text{SA}^{-1}[i]$ | $j = \text{SA}[r-1]$ | Start $h$ | Compare | $\text{LCP}[r]$ | End $h$ |
|------|-----|--------------------------|----------------------|-----------|---------|------------------|---------|
| 1 | 0 | 4 | 1 | 0 | `b` vs `a` | 0 | 0 |
| 2 | 1 | 3 | 3 | 0 | `ana..` vs `ana..` → match 3 | 3 | 2 |
| 3 | 2 | 6 | 4 | 2 | `na$` vs `na$` → match 2 | 2 | 1 |
| 4 | 3 | 2 | 5 | 1 | `a` matches, then `n` vs `$` | 1 | 0 |
| 5 | 4 | 5 | 0 | 0 | `n` vs `b` | 0 | 0 |
| 6 | 5 | 1 | 6 | 0 | `a` vs `$` | 0 | 0 |
| 7 | 6 | 0 | — | 0 | (first in order) | 0 | 0 |

Result: $\text{LCP} = [0, 0, 1, 3, 0, 0, 2]$, matching the expected values.

## Amortized Analysis

**Claim**: Kasai's algorithm runs in $O(n)$ time.

**Proof**: The variable $h$ acts as a potential function. Observe two facts:

1. In each iteration, $h$ can **increase** by some amount $\delta_i \geq 0$ during the while-loop comparisons.
2. Between iterations, $h$ **decreases** by at most 1 (the `h = max(h - 1, 0)` line).

The total number of character comparisons across all iterations equals the total increase in $h$:

$$
\text{Total comparisons} = \sum_{i=0}^{n} \delta_i
$$

Since $h$ starts at 0, ends at some value $\geq 0$, and decreases by at most 1 per iteration (at most $n+1$ decreases total), the total increase is bounded by:

$$
\sum_{i=0}^{n} \delta_i \leq (n+1) + h_{\text{final}} \leq 2(n+1)
$$

because $h$ can never exceed $n$ and each decrease of 1 must have been preceded by an increase of at least 1. Therefore the total work is $O(n)$. $\square$

**Space complexity**: The algorithm uses $O(n)$ space for the rank array and the LCP array.

## Implementation

```python
"""
Kasai's algorithm for O(n) LCP array construction.
"""


# === Kasai's Algorithm ===

def kasai(text: str, sa: list[int]) -> list[int]:
    """Compute the LCP array in O(n) time using Kasai's algorithm.

    Parameters
    ----------
    text : str
        The input string (with sentinel).
    sa : list[int]
        The suffix array of text.

    Returns
    -------
    list[int]
        The LCP array where lcp[k] is the length of the longest
        common prefix between suffix(sa[k-1]) and suffix(sa[k]).
    """
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


# === Main ===

if __name__ == "__main__":
    text = "banana$"
    sa = [6, 5, 3, 1, 0, 4, 2]
    lcp = kasai(text, sa)
    print(f"Text: {text}")
    print(f"SA:   {sa}")
    print(f"LCP:  {lcp}")
    print("\nVerification:")
    for k in range(len(sa)):
        suffix = text[sa[k]:]
        prev = text[sa[k - 1]:] if k > 0 else ""
        print(f"  LCP[{k}] = {lcp[k]:2d}  suffix: {suffix}")
```

## Reference

- Kasai, T., Lee, G., Arimura, H., Arikawa, S., and Park, K. (2001). *Linear-time longest-common-prefix computation in suffix arrays and its applications*. CPM 2001, LNCS 2089, pp. 181-192.
