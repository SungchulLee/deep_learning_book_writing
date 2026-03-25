# Efficient Suffix Array Construction

Sorting all suffixes naively by comparing them as strings takes $O(n^2 \log n)$ time in the worst case: there are $n$ suffixes to sort, comparison-based sorting needs $O(n \log n)$ comparisons, and each string comparison costs $O(n)$. For texts with millions of characters, this is prohibitively slow. The **prefix-doubling** technique, introduced by Karp, Miller, and Rosenberg and later refined by Manber and Myers, constructs the suffix array in $O(n \log n)$ time by exploiting a key insight: if we know how suffixes compare by their first $k$ characters, we can determine how they compare by their first $2k$ characters in $O(n)$ time using radix sort.

## Naive Construction and Its Cost

The simplest approach to building a suffix array is to generate all $n$ suffixes and sort them with a standard comparison-based sort. Each comparison of two suffixes of length up to $n$ takes $O(n)$ time, and sorting $n$ elements requires $O(n \log n)$ comparisons, giving:

$$
T_{\text{naive}} = O(n^2 \log n)
$$

This is impractical for large inputs. For example, the human genome has roughly $3 \times 10^9$ characters, making $n^2 \log n$ operations infeasible.

## Prefix Doubling Strategy

The prefix-doubling algorithm assigns a **rank** to each suffix based on increasingly long prefixes. In each round, the prefix length doubles from $k$ to $2k$. The key observation is that the rank of suffix($i$) by its first $2k$ characters can be determined from just two values: the rank of suffix($i$) and the rank of suffix($i + k$) by their first $k$ characters.

Formally, define the **rank array** $R_k$ at step $k$ so that $R_k[i]$ represents the rank of suffix($i$) when suffixes are compared only by their first $k$ characters. The initial step uses only the first character:

$$
R_1[i] = \text{rank of } T[i] \text{ in } \Sigma
$$

At each doubling step, suffix($i$) is sorted by the key pair $(R_k[i],\; R_k[i + k])$, which captures the first $2k$ characters:

$$
\text{suffix}(i)[0..2k-1] = \text{suffix}(i)[0..k-1] \cdot \text{suffix}(i+k)[0..k-1]
$$

If $i + k > n$, we use a sentinel rank of $-1$ (smaller than all other ranks) for the second component.

## Algorithm

The prefix-doubling algorithm proceeds as follows:

**Step 1 (Initialize).** Set $k = 1$. Compute $R_1[i]$ by ranking all characters $T[i]$ alphabetically. If the alphabet is small, counting sort achieves $O(n)$ time.

**Step 2 (Double and sort).** For each suffix $i$, form the pair $(R_k[i],\; R_k[i+k])$. Sort all suffixes by these pairs using **radix sort** on the two components. This takes $O(n)$ time per round because each component has values in $\{-1, 0, 1, \ldots, n-1\}$.

**Step 3 (Update ranks).** After sorting, assign new ranks $R_{2k}$ based on the sorted order. Equal pairs receive the same rank. If all ranks are distinct, all suffixes are fully resolved and the algorithm terminates.

**Step 4 (Iterate).** Set $k \leftarrow 2k$ and repeat from Step 2. Since the prefix length doubles each round, at most $\lceil \log_2 n \rceil$ rounds are needed before every suffix is uniquely ranked (i.e., $k \geq n$).

```
PREFIX-DOUBLING(T[0..n]):
    k = 1
    R = initial ranks from single characters
    while not all ranks distinct and k < n+1:
        for each i: key[i] = (R[i], R[i+k] if i+k <= n else -1)
        sort suffixes by key using radix sort
        update R from sorted order (assign same rank to equal keys)
        k = 2 * k
    SA = indices sorted by final R
    return SA
```

## Worked Example

Consider $T = \texttt{aab\$}$ with $n = 3$ (so $T[0..3]$ has length 4 including the sentinel).

**Round $k = 1$** (rank by first character):

| $i$ | suffix($i$) | $T[i]$ | $R_1[i]$ |
|-----|------------|--------|-----------|
| 0 | `aab$` | `a` | 1 |
| 1 | `ab$` | `a` | 1 |
| 2 | `b$` | `b` | 2 |
| 3 | `$` | `$` | 0 |

Ranks are not all distinct ($R_1[0] = R_1[1] = 1$), so we continue.

**Round $k = 2$** (rank by first 2 characters):

| $i$ | Key $(R_1[i], R_1[i+1])$ | First 2 chars |
|-----|--------------------------|---------------|
| 0 | $(1, 1)$ | `aa` |
| 1 | $(1, 2)$ | `ab` |
| 2 | $(2, 0)$ | `b$` |
| 3 | $(0, -1)$ | `$` |

Sorting by keys: $(0,-1) < (1,1) < (1,2) < (2,0)$, giving $R_2 = [1, 2, 3, 0]$. All ranks are distinct, so the algorithm terminates.

**Result**: $\text{SA} = [3, 0, 1, 2]$, corresponding to the sorted suffixes `$`, `aab$`, `ab$`, `b$`.

## Complexity Analysis

**Time complexity**: Each round performs a radix sort in $O(n)$ time and updates ranks in $O(n)$ time. There are $O(\log n)$ rounds because the prefix length doubles each iteration. The total time is:

$$
T(n) = O(n \log n)
$$

**Space complexity**: The algorithm stores the rank array $R$ (size $n$), the key pairs (size $2n$), and auxiliary arrays for radix sort. The total space is:

$$
S(n) = O(n)
$$

!!! tip "Practical optimization"
    If after any round all ranks are distinct, the suffixes are fully resolved and the algorithm can terminate early. For random strings over a large alphabet, this often happens after $O(\log \log n)$ rounds in practice, though the worst case remains $O(\log n)$ rounds.

## Using Comparison Sort Instead

If radix sort is replaced by a comparison-based sort (e.g., quicksort), each round takes $O(n \log n)$ instead of $O(n)$, yielding an overall time of:

$$
T(n) = O(n \log^2 n)
$$

This simpler variant is often used in practice because it is easier to implement and performs well on typical inputs despite the extra $\log n$ factor.

```python
"""
O(n log^2 n) suffix array construction using prefix doubling
with comparison-based sorting.
"""


# === Suffix Array Construction ===

def build_suffix_array(text: str) -> list[int]:
    """Build a suffix array using prefix doubling with comparison sort.

    Parameters
    ----------
    text : str
        Input string (a sentinel '$' is appended if not present).

    Returns
    -------
    list[int]
        The suffix array as a list of starting indices.
    """
    if not text.endswith("$"):
        text += "$"

    n = len(text)
    # Initial ranks from character ordinals
    rank = [ord(c) for c in text]
    sa = list(range(n))
    k = 1

    while k < n:
        def compare_key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)

        sa.sort(key=compare_key)

        # Update ranks
        new_rank = [0] * n
        new_rank[sa[0]] = 0
        for j in range(1, n):
            prev_key = (rank[sa[j - 1]],
                        rank[sa[j - 1] + k] if sa[j - 1] + k < n else -1)
            curr_key = (rank[sa[j]],
                        rank[sa[j] + k] if sa[j] + k < n else -1)
            new_rank[sa[j]] = new_rank[sa[j - 1]] + (1 if curr_key != prev_key else 0)

        rank = new_rank

        # Early termination if all ranks are unique
        if rank[sa[-1]] == n - 1:
            break

        k *= 2

    return sa


# === Main ===

if __name__ == "__main__":
    text = "banana$"
    sa = build_suffix_array(text)
    print(f"Text: {text}")
    print(f"Suffix Array: {sa}")
    print("\nSorted suffixes:")
    for i, idx in enumerate(sa):
        print(f"  SA[{i}] = {idx}: {text[idx:]}")
```

## Reference

- Karp, R. M., Miller, R. E., and Rosenberg, A. L. (1972). *Rapid identification of repeated patterns in strings, trees and arrays*. ACM Symposium on Theory of Computing.
- Manber, U. and Myers, G. (1993). *Suffix arrays: A new method for on-line string searches*. SIAM Journal on Computing, 22(5), 935-948.
