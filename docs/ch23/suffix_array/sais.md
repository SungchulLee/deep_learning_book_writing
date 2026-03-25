# SA-IS Algorithm

While prefix doubling constructs the suffix array in $O(n \log n)$ time, the theoretical optimal is $O(n)$. The **SA-IS** (Suffix Array by Induced Sorting) algorithm, introduced by Nong, Zhang, and Chan (2009), achieves this optimal bound. SA-IS is notable not only for its theoretical efficiency but also for its practical speed and elegant design. The algorithm classifies each suffix as either S-type or L-type, identifies special **LMS (Leftmost S-type)** positions, recursively sorts the LMS suffixes on a reduced problem of half the size, and then induces the full sorted order in two linear scans. This section explains each step in detail.

## S-Type and L-Type Classification

The first step of SA-IS classifies each position $i$ in the string $T[0..n]$ (where $T[n] = \$$ is the sentinel) based on how suffix($i$) compares to suffix($i+1$).

A position $i$ is **S-type** (smaller) if suffix($i$) is lexicographically smaller than suffix($i+1$):

$$
\text{suffix}(i) < \text{suffix}(i+1)
$$

A position $i$ is **L-type** (larger) if suffix($i$) is lexicographically larger than suffix($i+1$):

$$
\text{suffix}(i) > \text{suffix}(i+1)
$$

The sentinel position $n$ is always S-type by convention (it is the lexicographically smallest suffix).

!!! tip "Efficient classification"
    The type of each position can be determined in a single right-to-left scan:

    - $T[n]$ is S-type
    - If $T[i] < T[i+1]$, then position $i$ is S-type
    - If $T[i] > T[i+1]$, then position $i$ is L-type
    - If $T[i] = T[i+1]$, then position $i$ has the same type as position $i+1$

## LMS Positions

A position $i$ is a **Leftmost S-type (LMS)** position if $i$ is S-type and $i - 1$ is L-type (or $i = n$ for the sentinel). LMS positions mark the boundaries of critical substrings that the algorithm processes recursively.

An **LMS substring** is the substring $T[i..j]$ where $i$ and $j$ are consecutive LMS positions (including the sentinel). The key insight of SA-IS is that there are at most $n/2$ LMS positions, enabling a recursive reduction by half.

??? example "Classification for 'mmiissiissiippii$'"
    For $T = \texttt{mmiissiissiippii\$}$ (length 17):

    | Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
    |----------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|-----|
    | Char | m | m | i | i | s | s | i | i | s | s | i  | i  | p  | p  | i  | i  | $   |
    | Type | L | L | S | S | L | L | S | S | L | L | S  | S  | L  | L  | S  | S  | S   |
    | LMS? |   |   | * |   |   |   | * |   |   |   | *  |    |    |    | *  |    | *   |

    LMS positions: 2, 6, 10, 14, 16

## Bucket Sorting

SA-IS uses **bucket sorting** based on the first character of each suffix. All suffixes starting with the same character $c$ go into the bucket for $c$. Within each bucket, L-type suffixes come before S-type suffixes (because an L-type suffix starting with $c$ is larger than $c$'s S-type successors, and the second character determines the rest).

Each bucket has a defined range $[\text{bkt\_start}(c), \text{bkt\_end}(c)]$ in the suffix array. L-type suffixes are placed from the **left** (head) of the bucket, and S-type suffixes are placed from the **right** (tail).

## The Induced Sorting Procedure

The core of SA-IS is **induced sorting**: given the sorted order of LMS suffixes, the algorithm deduces the sorted order of all suffixes in two linear scans.

**Step 1: Place LMS suffixes.** Insert the LMS suffixes into their correct buckets (at the tail of each bucket) in their sorted order.

**Step 2: Induce L-type suffixes (left-to-right scan).** Scan the suffix array from left to right. For each filled position $\text{SA}[k] = i$, if position $i - 1$ is L-type, place suffix($i - 1$) at the head of its bucket. Since L-type suffixes are larger than the following character, this scan correctly orders them.

**Step 3: Induce S-type suffixes (right-to-left scan).** Scan the suffix array from right to left. For each filled position $\text{SA}[k] = i$, if position $i - 1$ is S-type, place suffix($i - 1$) at the tail of its bucket.

After these two scans, the entire suffix array is correctly sorted.

## Recursive Reduction

The LMS suffixes themselves must be sorted before the induced sorting procedure can begin. SA-IS achieves this recursively:

1. **Name LMS substrings**: Perform a preliminary induced sort to determine the relative order of LMS substrings. Assign integer names based on this order.

2. **Create reduced string**: Build a new string $T_1$ of length at most $n/2$, where each character is the name of an LMS substring.

3. **Recurse or direct sort**: If all names in $T_1$ are distinct, the suffix array of $T_1$ is determined directly. Otherwise, recursively apply SA-IS to $T_1$.

4. **Map back**: Convert the suffix array of $T_1$ back to the sorted order of LMS suffixes in the original string.

The recursion depth is $O(\log n)$ in the worst case, but the total work across all levels is $O(n)$ because the problem size halves at each level:

$$
T(n) = T(n/2) + O(n) = O(n)
$$

## Complete Algorithm Summary

```
SA-IS(T[0..n]):
    1. Classify each position as S-type or L-type (right-to-left scan)
    2. Identify LMS positions
    3. Place LMS suffixes into buckets (approximate sort)
    4. Induce-sort all L-type suffixes (left-to-right scan)
    5. Induce-sort all S-type suffixes (right-to-left scan)
       → This gives the sorted order of LMS substrings
    6. Assign integer names to LMS substrings by their sorted order
    7. Create reduced string T1 from the names
    8. If not all names unique:
         SA1 = SA-IS(T1)          // recurse
       Else:
         SA1 = directly compute from unique names
    9. From SA1, determine sorted order of LMS suffixes
   10. Clear SA, place sorted LMS suffixes into buckets
   11. Induce-sort all L-type suffixes (left-to-right)
   12. Induce-sort all S-type suffixes (right-to-left)
   13. Return SA
```

## Complexity Analysis

**Time complexity**: Each level of recursion processes a string of at most half the previous length and does $O(n)$ work (classification, bucket sorting, and two linear scans). The total time satisfies:

$$
T(n) \leq T(n/2) + cn
$$

By the master theorem (or by telescoping), $T(n) = O(n)$.

**Space complexity**: The algorithm requires $O(n)$ auxiliary space for the type array, bucket pointers, and the reduced string. With careful implementation, the constant factor can be made small (approximately $6n$ bytes for byte-alphabet strings).

!!! note "Practical performance"
    Despite its theoretical optimality, SA-IS is also one of the fastest suffix array construction algorithms in practice. Its cache-friendly linear scans and small constant factors make it competitive with or faster than many $O(n \log n)$ alternatives on real-world data.

## Comparison with Other Linear-Time Algorithms

| Algorithm | Year | Time | Space | Practical Speed |
|-----------|------|------|-------|-----------------|
| DC3/Skew | 2003 | $O(n)$ | $O(n)$ | Moderate |
| KA (Ko-Aluru) | 2005 | $O(n)$ | $O(n)$ | Moderate |
| SA-IS | 2009 | $O(n)$ | $O(n)$ | Fast |

SA-IS is generally preferred because it has the smallest constant factor and the simplest implementation among linear-time algorithms.

## Reference

- Nong, G., Zhang, S., and Chan, W. H. (2009). *Two efficient algorithms for linear time suffix array construction*. IEEE Transactions on Computers, 60(10), 1471-1484.
- Karkkainen, J., Sanders, P., and Burkhardt, S. (2006). *Linear work suffix array construction*. Journal of the ACM, 53(6), 918-936.
